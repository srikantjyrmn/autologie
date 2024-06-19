"""Class to connect to LanguageModel APIEndpoints and extract structured / unstructured responses.
"""

import json
from datetime import datetime
from typing import Union, List, Dict, Generator
from functools import partial
import requests

from .types import (
    ApiClientSpec, ResponseSpec,
    ChatMessage, ToolMessage,
    CompletionApiResponse, ChatApiResponse,
    LlamacppCompletionPayload,
    OllamaCompletionPayload,
    OpenaiChatPayload
)
from .response_handlers import (
    response_object_handler,
    response_validation_handler
)

from .prompt_handlers import(
    chat_template_handler, grammar_handler,
    payload_handler
)

class ApiConnector:
    """ Class to make requests to an API endpoint. Can return response, or stream of responses.
    
    Args:
        base_url: str = Base URL of the Endpoint
        api_key: str: API Key
        
    Methods:
        get_response
        
    """
    def __init__(self, base_url : str, api_key : str = None, inference_logger = None):
        """Initialize a connection to an API endpoint.

        Args:
            base_url (str): URL to the API endpoint.
            api_key (str, optional): API key to use for Authentication. Defaults to None.
        """
        self.base_url = base_url
        self.key = api_key
        if not inference_logger:
            from .inference_logger import inference_logger
            self.inference_logger = inference_logger
        else:
            self.inference_logger = inference_logger

        # Initialize headers with API Key if provided
        if api_key:
            self.headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.key}",  # Add API key as bearer token
            }
        else:
            self.headers = {"Content-Type": "application/json"}

        self.inference_logger.info(f"""{__name__}:
        Initializing Connection to {self.base_url} with header:{json.dumps(
            self.headers,sort_keys=True,
                indent=4,
                separators=(',', ': '))}
        """
        )

    def generate_text_chunks(self, response) -> Generator[Dict, Dict, Dict]:
        """Generate text chunks. Helper for stream responses.
        #TODO."""
        # Define a generator function to yield text chunks
        try:
            for chunk in response.iter_lines():
                decoded_chunk = ""
                if chunk:
                    decoded_chunk += chunk.decode("utf-8")
                    decoded_json = json.loads(decoded_chunk.replace("data: ", ""))
                    #returned_data = {"choices": [{"text": new_data["text"]}]}
                    yield decoded_json
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")

    def get_response(self,
                    payload: Dict,
                    stream : bool = False
                ) -> Union[Dict, Generator[Dict, Dict, Dict]]:
        """Get response from an API endpoint.
        Args:
            payload (dict): The Data dictionary to pass into the request.
        Returns:
            json: JSON object of the Api Response
        """
        self.inference_logger.info(
            f"""{__name__} {datetime.now()}: ApiConnector Says:
            Getting response from {self.base_url} for payload :
                {json.dumps(payload,sort_keys=True,
                    indent=4,
                    separators=(',', ': '))
                }
            with headers 
                {self.headers}
            \n"""
        )
        try:
            response = requests.post(
            self.base_url,
            headers = self.headers,
            json = payload,
            stream = stream,
            timeout = 60*10 # 10 minutes
            )
            #print(response)
            if stream:
                return self.generate_text_chunks(response)
            else:
                return response.json()
        except Exception as e:
            self.inference_logger.error(f"{e} No Response")
            return None

class ApiClient:
    """Main Class for an API Client
    Args:
        ApiSpec: Specifications for the API.
        **kwargs: Any args to provide ApiSpec
        By default returns a completion endpoint on llamacpp.
    """
    api_spec : ApiClientSpec
    payload_handler: Union[LlamacppCompletionPayload, OllamaCompletionPayload, OpenaiChatPayload]
    response_object_handler: Union[CompletionApiResponse, ChatApiResponse]
    def __init__(
        self,
        api_spec: None | ApiClientSpec = None,
        inference_logger = None,
        **kwargs
    ):
        if api_spec:
            self.api_spec = api_spec
        else:
            self.api_spec = ApiClientSpec(**kwargs)
        if inference_logger:
            self.inference_logger = inference_logger
        else:
            from .inference_logger import inference_logger
            self.inference_logger = inference_logger
        self.inference_logger.info(f"""Initializing an API Endpoint with specs:
              {json.dumps(self.api_spec.__dict__,
              sort_keys=True,
                indent=4,
                separators=(',', ': '))}
        """)
        self.api = ApiConnector(
                    base_url = self.api_spec.base_url,
                    api_key = self.api_spec.api_key,
                    inference_logger=self.inference_logger
                )
        self.prompt_formatter = chat_template_handler(
            endpoint_type=self.api_spec.endpoint_type,
            chat_template=self.api_spec.chat_template,
            hf_model_name=self.api_spec.hf_repo_id
        )
        self.payload_handler = payload_handler(
            self.api_spec.endpoint_type,
            self.api_spec.server
        )
        self.response_object_handler = partial(response_object_handler,
            endpoint_type = self.api_spec.endpoint_type,
            server = self.api_spec.server
        )
        self.grammar_handler = partial(grammar_handler, server = self.api_spec.server)

    def get_api_response(self, payload) -> CompletionApiResponse | ChatApiResponse:
        """Gets a response from an API and returns a standard ApiResponse object"""
        api_response = self.api.get_response(
            payload = payload
        )
        self.inference_logger.info(f"Got Response: {api_response}")
        return self.response_object_handler(api_response)

    def run_unvalidated_inference(self,
                    messages: List[ChatMessage],
                    response_format : None|str| Dict = None,
                    tools : None | List = None,
                    tool_choice : None | str = None,
                    available_functions = None,
                    **kwargs
    ) -> ChatApiResponse:
        response_spec = ResponseSpec(
                response_format = response_format,
                tools = tools,
                tool_call = tool_choice,
                available_functions=available_functions
            )
        self.inference_logger.info(f"Running Inference with ResponseMode: {response_spec.response_mode}")
        if self.api_spec.endpoint_type == 'completion':
            payload = self.payload_handler(
                    model = self.api_spec.model,
                    prompt = self.prompt_formatter.format_conversation(
                        [x.message_dict for x in messages]
                    ),
                    grammar = self.grammar_handler(response_spec = response_spec),
                    response_format = response_spec.response_format,
                    tools = response_spec.tools
                ).payload_dict
            api_response = self.get_api_response(payload)
            self.inference_logger.info(f"""\n Got Response Object: {api_response}.
                Not Processing Completion and Validating. \n""")
        else:
            api_response = self.api.get_response(
                payload = self.payload_handler(
                    messages = [x.message_dict for x in messages],
                    model = self.api_spec.model,
                    **kwargs
                ).payload_dict
            )
            self.inference_logger.info(api_response)
        return api_response#self.response_object_handler(**api_response)

    def run_inference(self,
                    messages: List[ChatMessage],
                    response_format : None|str| Dict = None,
                    tools : None | List = None,
                    tool_choice : None | str = None,
                    available_functions = None,
                    **kwargs) -> ChatApiResponse:
        """Get responses from an API Endpoint"""
        try:
            response_spec = ResponseSpec(
                response_format = response_format,
                tools = tools,
                tool_call = tool_choice,
                available_functions=available_functions
            )
            self.inference_logger.info(f"Running Inference with ResponseMode: {response_spec.response_mode}")

            if self.api_spec.endpoint_type == 'completion':
                validated = False
                n_tries = 5
                while not validated:
                    # Format prompt and get response
                    payload = self.payload_handler(
                            model = self.api_spec.model,
                            prompt = self.prompt_formatter.format_conversation(
                                [x.message_dict for x in messages]
                            ),
                            grammar = self.grammar_handler(response_spec = response_spec),
                            response_format = response_spec.response_format,
                            tools = response_spec.tools
                        ).payload_dict
                    api_response = self.get_api_response(payload)
                    self.inference_logger.info(f"""\n Got Response Object: {api_response}.
                        Processing Completion and Validating. \n""")

                    # Validate Completion according to response type
                    completion_validation, tool_calls = response_validation_handler(
                        completion = api_response.choices[0].text,
                        response_spec = response_spec
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
                    messages.append(ToolMessage(content = completion_validation.assistant_message))
                    continue
                # If n_tries is exceeded, returns the last available response
                return completion_validation#.validation, completion_validation.error_message, None

            # In case it's a chat endpoint, we're assuming the responses will already be validated
            #TODO: Handle this case as well. Can still run validation on teh tools returned.
            #TODO: JSON will still need to be validated
            api_response = self.api.get_response(
                payload = self.payload_handler(
                    messages = [x.message_dict for x in messages],
                    model = self.api_spec.model,
                    **kwargs
                ).payload_dict
            )
            self.inference_logger.info(api_response)
            return self.response_object_handler(**api_response)
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None
