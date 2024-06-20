"""Class to chat with an LLM Endpoint."""
from typing import List, Dict
from pydantic import BaseModel
from .clients import ApiClient, ApiClientSpec
from .response_handlers import (
    response_validation_handler, execute_function_call, 
    parse_objects_from_response, parse_tool_call_from_response
)
from .prompt_handlers import PromptManager
from .types import (
    UserMessage, SystemMessage, ApiResponse,
    AssistantMessage, ResponseSpec, ToolMessage
)
from .types.responses import OpenaiResponseFormat

class LlmSpec(BaseModel):
    """Specifications to init an Llm Client
    api_spec: ApiClientSpec
    """
    api_spec: ApiClientSpec = ApiClientSpec()
    system_prompt: None | str = None
    response_format: None | OpenaiResponseFormat = None
    tools: None | List = None

def get_system_prompt(llm_spec: LlmSpec):
    """Get a system prompt from the Model Specifications"""
    if not llm_spec.system_prompt:
        return PromptManager().generate_system_prompt(llm_spec.tools)
    if isinstance(llm_spec.system_prompt, str):
        return llm_spec.system_prompt
    if isinstance(llm_spec.system_prompt, PromptManager):
        return llm_spec.system_prompt.generate_system_prompt(llm_spec.tools)
    return "You are an intelligent AI Agent."

class LlmClient:
    """Base Client to chat with an Llm."""
    spec: LlmSpec

    def __init__(self, llm_spec: LlmSpec = None, inference_logger = None, **kwargs):
        if llm_spec:
            self.llm_spec = llm_spec
        else:
            self.llm_spec = LlmSpec(**kwargs)

        if inference_logger:
            self.inference_logger = inference_logger
        else:
            from .inference_logger import inference_logger
            self.inference_logger = inference_logger
        self.inference_logger.info(self.llm_spec)

        self.system_prompt = get_system_prompt(self.llm_spec)
        if self.system_prompt:
            self.messages = [SystemMessage(content = self.system_prompt)]
        self.api = ApiClient(self.llm_spec.api_spec, inference_logger=self.inference_logger)
        self.response_spec = ResponseSpec(
            response_format=self.llm_spec.response_format,
            tools = self.llm_spec.tools
        )

    def get_response(self, message: str) -> ApiResponse:
        """Get response from the self's ApiClient"""
        self.messages.append(UserMessage(content = message))
        validated = False
        n_tries = 5
        while not validated:
            # Format prompt and get response
            api_response = self.api.run_unvalidated_inference(
                messages = self.messages,
                response_format= self.llm_spec.response_format,
                tools=self.llm_spec.tools
            )
            self.inference_logger.info(f"""\n Got Response Object: {api_response}.
                Processing Completion and Validating. \n""")
            self.messages.append(AssistantMessage(content = api_response.choices[0].text))
            # Validate Completion according to response type
            completion_validation, _ = response_validation_handler(
                completion = api_response.choices[0].text,
                response_spec = self.response_spec
            )
            self.inference_logger.info(f"***Got validation object {completion_validation}")
            n_tries -= 1

            # If validated, return response
            if completion_validation.validation and not completion_validation.get_feedback:
                validated = True
                self.inference_logger.info(f"\n Got Validation: {completion_validation}. Returning \n")
                return completion_validation#, api_response, tool_calls
            # If not validated, continue loop
            #print(f"Object validation failed. {n_tries} tries left. Trying again.\n\n\n\n\n\n {completion_validation.assistant_message} \n\n\n\n")
            self.messages.append(ToolMessage(content = completion_validation.assistant_message))
            continue
        # If n_tries is exceeded, returns the last available response
        return completion_validation#.validation, completion_validation.error_message, None

    def chat(self, message: str) -> str:
        """Chat with an LlmClient with a message string"""
        response = self.get_response(message)
        return response


class AgentClient:
    """A different approach.
    listen -> speak -> reflect -> act -> react"""
    def __init__(
            self,
            system_prompt: str | None = None,
            response_format: Dict | None = None,
            tools : List | None = None,
            inference_logger = None
        ):
        self.llm_spec = LlmSpec(
            system_prompt =system_prompt,
            response_format = response_format,
            tools = tools
        )
        self.system_prompt = get_system_prompt(self.llm_spec)
        if self.system_prompt:
            self.messages = [SystemMessage(content = self.system_prompt)]
        
        self.api = ApiClient(self.llm_spec.api_spec)
        
        self.response_spec = ResponseSpec(
            response_format=self.llm_spec.response_format,
            tools = self.llm_spec.tools
        )
        self.validated = False
        
        if not inference_logger:
            from .inference_logger import inference_logger
        self.inference_logger = inference_logger
        self.inference_count = 0
        self.inference_logger.info(self.llm_spec)

    def listen(self, message:str):
        self.messages.append(UserMessage(content = message))
        self.validated = False

    def speak(self):
        self.inference_logger.info(f"Speaker says: Speaking, Inference Count: {self.inference_count}")
        api_response = self.api.run_unvalidated_inference(
            messages = self.messages,
            response_format= self.llm_spec.response_format,
            tools=self.llm_spec.tools
        )
        self.inference_logger.info(f"""Speaker says:
        Got Response Object: {api_response}.
        Processing Completion and Validating.
        """)
        self.messages.append(AssistantMessage(content = api_response.choices[0].text))
        self.inference_count += 1

    def reflect(self):
        n_tries = 5
        self.inference_logger.info("Reflection says: Reflecting")
        while not self.validated:
            self.speak()
            completion_validation, _ = response_validation_handler(
                completion = self.messages[-1].content,
                response_spec = self.response_spec
            )
            self.inference_logger.info(f"""Reflection says: tried: {5-n_tries} ntry left: {n_tries}
            Got validation object {completion_validation}
            """)
            n_tries -= 1
            if (completion_validation.validation) and (not completion_validation.get_feedback):
                self.validated = True
                self.inference_logger.info(f"""Reflection says:
                Got Validated Response: {completion_validation}.
                Returning
                """)
                return completion_validation
            self.messages.append(ToolMessage(content = completion_validation.assistant_message))
            continue
        return completion_validation

    def act(self, completion_validation):
        self.inference_logger.info("Actor says: Acting")
        if not self.validated:
            self.inference_logger.error("Response not validated. Why is the actor being called?")
            return ValueError()
        if completion_validation.get_feedback:
            self.inference_logger.error("Feedback requested. Why is the actor being called?")
            return ValueError()
        if completion_validation.tool_calls:
            tool_message = ""
            for tool_call in completion_validation.tool_calls:
                tool_results = execute_function_call(tool_call=tool_call.dict())
                self.inference_logger.info(f"Tool returned Results: {tool_results}")
                tool_message += f"<tool_response> {tool_results} </tool_response>"
                self.messages.append(ToolMessage(content = tool_message))
            self.validated = False
            completion_validation = self.reflect()
        if completion_validation.objects:
            objects = []
            for obj in completion_validation.objects:
                objects.append(self.response_spec.response_format.schema(**obj))
            completion_validation.objects = objects
        return completion_validation

    def chat(self, message):
        self.listen(message)
        reflection = self.reflect()
        response = self.act(reflection)
        return response