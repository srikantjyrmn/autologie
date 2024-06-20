"""Handlers for Endpoints, and corresponding Types."""
import os
import json
import re
from typing import List, Dict
from datetime import datetime

import ast
import yaml
from jinja2 import Template
from transformers import AutoTokenizer
import xml.etree.ElementTree as ET
from jsonschema import validate
from pydantic import ValidationError

from .types import (
    PromptSchema, ChatTemplate,
    ResponseSpec, ApiResponse,
    Payload,
    CompletionApiResponse, ApiResponseValidation,
    LlmServers, LlmEndpointTypes, OllamaCompletionResponse,
    ClientHandlerFactory,
    FunctionCallMessage, FunctionSignature
)

from .types.helpers.gbnf_grammar_from_pydantic_models import generate_gbnf_grammar_from_pydantic_models

from .utils import extract_json_from_markdown
from ..tools import nous_functions as available_functions
from .inference_logger import inference_logger



def get_fewshot_examples(num_fewshot):
    """return a list of few shot examples"""
    example_path = os.path.join(script_dir, 'prompt_assets', 'few_shot.json')
    with open(example_path, 'r') as file:
        examples = json.load(file)  # Use json.load with the file object, not the file path
    if num_fewshot > len(examples):
        raise ValueError(f"Not enough examples (got {num_fewshot}, but there are only {len(examples)} examples).")
    return examples[:num_fewshot]


def get_assistant_message(completion, chat_template, eos_token):
    """define and match pattern to find the assistant message"""
    completion = completion.strip()

    if chat_template == "zephyr":
        assistant_pattern = re.compile(r'<\|assistant\|>((?:(?!<\|assistant\|>).)*)$', re.DOTALL)
    elif chat_template == "chatml":
        assistant_pattern = re.compile(r'<\|im_start\|>\s*assistant((?:(?!<\|im_start\|>\s*assistant).)*)$', re.DOTALL)

    elif chat_template == "vicuna":
        assistant_pattern = re.compile(r'ASSISTANT:\s*((?:(?!ASSISTANT:).)*)$', re.DOTALL)
    else:
        raise NotImplementedError(f"Handling for chat_template '{chat_template}' is not implemented.")
    
    assistant_match = assistant_pattern.search(completion)
    if assistant_match:
        assistant_content = assistant_match.group(1).strip()
        if chat_template == "vicuna":
            eos_token = f"</s>{eos_token}"
        return assistant_content.replace(eos_token, "")
    else:
        assistant_content = None
        inference_logger.info("No match found for the assistant pattern")
        return assistant_content

class PromptManager:
    """A class to manage prompts for a function calling model."""
    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

    def format_yaml_prompt(self, prompt_schema: PromptSchema, variables: Dict) -> str:
        """Given a PromptSchema with ROle, Instruction etc, format it into a prompt."""
        formatted_prompt = ""
        for field, value in prompt_schema.dict().items():
            if field == "Examples" and variables.get("examples") is None:
                continue
            formatted_value = value.format(**variables)
            if field == "Instructions":
                formatted_prompt += f"{formatted_value}"
            else:
                formatted_value = formatted_value.replace("\n", " ")
                formatted_prompt += f"{formatted_value}"
        return formatted_prompt

    def read_yaml_file(self, file_path: str) -> PromptSchema:
        """Read a yaml file containing a PromptSchema for an agent."""
        with open(file_path, 'rb') as file:
            yaml_content = yaml.safe_load(file)

        prompt_schema = PromptSchema(
            Role=yaml_content.get('Role', ''),
            Objective=yaml_content.get('Objective', ''),
            Tools=yaml_content.get('Tools', ''),
            Examples=yaml_content.get('Examples', ''),
            Schema=yaml_content.get('Schema', ''),
            Instructions=yaml_content.get('Instructions', ''),
        )
        return prompt_schema

    def generate_prompt(self, user_prompt, tools, num_fewshot=None):
        """Generate the system prompt for a given PromptSchema.
        #Deprecated."""
        prompt_path = os.path.join(self.script_dir, 'prompt_assets', 'sys_prompt.yml')
        prompt_schema = self.read_yaml_file(prompt_path)

        if num_fewshot is not None:
            examples = get_fewshot_examples(num_fewshot)
        else:
            examples = None

        schema_json = json.loads(FunctionCallMessage.schema_json())
        #schema = schema_json.get("properties", {})

        variables = {
            "date": datetime.today(),
            "tools": tools,
            "examples": examples,
            "schema": schema_json
        }
        sys_prompt = self.format_yaml_prompt(prompt_schema, variables)

        prompt = [
                {'content': sys_prompt, 'role': 'system'}
            ]
        prompt.extend(user_prompt)
        return prompt

    def generate_system_prompt(self, tools:List,  system_prompt: str = "", num_fewshot:int =None):
        """Generate the system prompt for a given PromptSchema.
        #TODO"""
        prompt_path = os.path.join(self.script_dir, 'prompt_assets', 'sys_prompt.yml')
        prompt_schema = self.read_yaml_file(prompt_path)

        if num_fewshot is not None:
            examples = get_fewshot_examples(num_fewshot)
        else:
            examples = None

        schema_json = json.loads(FunctionCallMessage.schema_json())
        #schema = schema_json.get("properties", {})

        variables = {
            "date": datetime.today(),
            "tools": tools,
            "examples": examples,
            "schema": schema_json
        }
        sys_prompt = self.format_yaml_prompt(prompt_schema, variables)
        prompt = sys_prompt
        return prompt

def format_chatml(messages):
    """Format a ChatML prompt"""
    x = ChatTemplate(name = 'chatml')
    t = Template(x.template)
    return t.render(messages = messages, add_generation_prompt =True )

class PromptFormatter:
    """Prompt Formatter class takes in a list of messages and 
    returns a string formatted according the ChatTemplate conventions.
    Uses Jinja2.
    Can also be initialized from a HuggingFace AutoTokenizer.
    
    Args:
        chat_template: str | ChatTemplate : The ChatTemplate to use. If string, the obj is init.
        
    Methods:
        format_conversation: Returns a formatted prompt from a list of ChatMessages
    """
    def __init__(self,
                chat_template: None | str | ChatTemplate = None,
                model_name: None | str = None
            ):
        if (model_name is None) and (chat_template is None):
            print("No Formatting requested. Returning None")
        elif model_name:
            self.type = 'hf_tokenizer'
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
            self.response_function = self.apply_hf_template
        elif isinstance(chat_template, str):
            print(f"Getting Jinja Template for {chat_template}")
            self.chat_template  = ChatTemplate(name = chat_template)
            print(f"Got template {self.chat_template}")
            self.template = Template(self.chat_template.template)
            self.response_function = self.apply_jinja_template
        else:
            print("TODO: Chat Template not defined")

    def apply_jinja_template(self, messages_list):
        """Apply jinja template to a list of messages."""
        formatted_prompt = self.template.render(
            messages = messages_list,
            add_generation_prompt=True
        )
        return formatted_prompt
    def apply_hf_template(self, messages_list):
        """Apply HF template to a list of messages."""
        formatted_prompt = self.tokenizer.apply(
            conversation = messages_list,
            add_generation_prompt=True,
            tokenize=False
        )
        return formatted_prompt
    def format_conversation(self, messages: List) -> str:
        """Convert messages into a formatted prompt."""
        print(f"FORMATTING CONVERSATION: {messages}")
        formatted_prompt = self.response_function(messages_list = messages)
        return formatted_prompt


def chat_template_handler(endpoint_type: LlmEndpointTypes, 
                        chat_template: str, hf_model_name : None | str = None):
    """Gets a PromptFormatter for a given EndPointSpec parameters."""
    print(f"Getting a chat template for {endpoint_type} {chat_template}")
    if endpoint_type in ['chat', 'openai']:
        print("Prompt Formatting not required for these endpoints/ Skipping, Returning None.")
        return None
    if endpoint_type == 'completion':
        return PromptFormatter(chat_template=chat_template, model_name= hf_model_name)
    return None

def grammar_handler(server: LlmServers, response_spec: ResponseSpec):
    """Returns a grammar string based on ResponseSpec and Server."""
    if (server == 'llamacpp') and (response_spec.response_mode == 'json'):
        print("Generating Grammar for {response_spec.response_format.schema}")
        return generate_gbnf_grammar_from_pydantic_models([response_spec.response_format.schema])
    if (server == 'llamacpp') and (
        response_spec.response_mode == 'function_calling') and (response_spec.use_grammar):
        print("Generating Grammar for {response_spec.tools}")
        return generate_gbnf_grammar_from_pydantic_models(response_spec.tools)
    if server != 'llamacpp':
        print(f"No Grammar for {server}")
    print("No grammar Initialized")
    return None

def get_settings_for_endpoint(server: LlmServers, endpoint_type: LlmEndpointTypes):
    """Helper to get pre-configured settings and objects for a given server endpoint"""
    return ClientHandlerFactory(server = server, endpoint_type = endpoint_type).get_handlers()

def payload_handler(endpoint_type: LlmEndpointTypes, server: LlmServers) -> Payload:
    """Function to handle construction of the payload to send for a given ChatInput"""
    print(f"Getting a payload handler for {server} {endpoint_type}")
    return get_settings_for_endpoint(server, endpoint_type).payload_object

def response_object_handler(response_object, endpoint_type: str, server: str) -> ApiResponse:
    """A handler to take in an ServerAPIResponse, validate the expected response format, 
    and return an appropriate output and change it to provide APIResponse
    """
    print(f"Getting a response object handler for {server} {endpoint_type}")
    response = get_settings_for_endpoint(server, endpoint_type).response_object(**response_object)
    print(f"Respone bject hndler: {response}")
    if isinstance(response, OllamaCompletionResponse):
        return CompletionApiResponse(**response.response_object)
    return response

def response_validation_handler(
    completion: CompletionApiResponse,
    response_spec: ResponseSpec
) -> ApiResponseValidation:
    """Takes in a completion response and parses it as per the ResponseSpec."""
    print(f"\n Getting a completion handler for {response_spec}")

    if response_spec.response_mode == 'default':
        return ApiResponseValidation(
            assistant_message = completion
        ), None
    validation, get_feedback, obj, tool_calls, message = process_completion_and_validate(
        completion = completion,
        response_spec = response_spec
    )
    print(f"***Got tool call {tool_calls}")
    return ApiResponseValidation(
            validation = validation,
            get_feedback = get_feedback,
            assistant_message = message,
            tool_calls = tool_calls,
            error_message = message if not validation else None,
            objects = [obj] if obj else None
        ), tool_calls

def execute_function_call(tool_call, available_functions = available_functions):
    "Function to execute a Llm generated function call"
    function_name = tool_call.get("name")
    function_to_call = getattr(available_functions, function_name, None)
    function_args = tool_call.get("arguments", {})

    print(f"Invoking function call {function_name} ...")
    function_response = function_to_call(*function_args.values())
    results_dict = f'{{"name": "{function_name}", "content": {function_response}}}'
    print(f""" {results_dict}
    """)
    return results_dict

def parse_objects_from_response(response, expected_schema):
    """Parse objects from an API response"""
    print("Validating JSON")
    validation, obj, error_message = validate_json_data(
        json_object = response,
        json_schema = json.loads(expected_schema)
    )
    print(f"\n\n\n Got JSON Validation: {validation} {obj} {error_message} \n\n\n")
    if validation:
        print(f"parsed json objects:\n{json.dumps(obj, indent=2)}")
        tool_message = response
    else:
        tool_message = f"""<tool_response>
        Json schema validation failed
        Here's the error stacktrace: {error_message}
        Please return correct json object
    <tool_response>"""
    return validation, obj, tool_message

def parse_tool_call_from_response(
    response: str, 
    tool_signatures: List[FunctionSignature],
    available_functions = available_functions):
    """Parses tool call from an api response"""
    get_feedback=True
    validation, tool_calls, error_message = validate_and_extract_tool_calls(response)
    if validation:
        if (tool_calls is not None) and len(tool_calls)>0:
            #tool_message = completion
            print(f"parsed tool calls:\n{json.dumps(tool_calls, indent=2)}")
            print("Validating function call schema")
            #tool_message = f"Agent iteration to assist with user query: {response}\n"   
            tool_message = ""
            for tool_call in tool_calls:
                validation, message = validate_function_call_schema(tool_call, tool_signatures)
                tool_message = message
            print(tool_message)
            get_feedback = False
            return validation, get_feedback, tool_calls, tool_message
        get_feedback=False
        return validation, get_feedback, tool_calls, response
    tool_message = f"""<tool_response>
    There was an error parsing function calls.
    Here's the error stack trace: {error_message}
    Please call the function again with correct syntax
<tool_response>"""
    return validation, get_feedback, tool_calls, tool_message

def process_completion_and_validate(response_spec: ResponseSpec, completion: str):
    """Handles response according to response mode."""
    response_mode = response_spec.response_mode
    get_feedback = False
    print(f"processing completion for response mode {response_mode} for text {completion}")
    if not completion:
        print("Assistant message is None")
        raise ValueError("Assistant message is None")

    if response_mode == 'default':
        print("Preparing Text Response")
        validation, obj, tool_calls, tool_message = True, None, None, completion
    
    elif response_mode == 'json':
        tool_calls = None
        validation, obj, tool_message = parse_objects_from_response(
            response = completion,
            expected_schema = response_spec.response_format.schema.schema_json()
        )

    elif response_mode == 'function_calling':
        print("Parsing response for function calls.")
        obj = None
        validation, get_feedback, tool_calls, tool_message = parse_tool_call_from_response(
            response = completion,
            tool_signatures = [x.__dict__ for x in response_spec.tools]
        )
    else:
        obj, tool_calls, tool_message = None, None, completion
    return validation, get_feedback, obj, tool_calls, tool_message

def validate_function_call_schema(call, signatures):
    try:
        print(f"Trying to valdiate functionc all schema {call} {signatures}")
        call_data = FunctionCallMessage(**call)
    except ValidationError as e:
        return False, str(e)

    for signature in signatures:
        try:
            signature_data = FunctionSignature(**signature)
            if signature_data.function.name == call_data.name:
                # Validate types in function arguments
                for arg_name, arg_schema in signature_data.function.parameters.get('properties', {}).items():
                    if arg_name in call_data.arguments:
                        call_arg_value = call_data.arguments[arg_name]
                        if call_arg_value:
                            try:
                                validate_argument_type(arg_name, call_arg_value, arg_schema)
                            except Exception as arg_validation_error:
                                return False, str(arg_validation_error)

                # Check if all required arguments are present
                required_arguments = signature_data.function.parameters.get('required', [])
                result, missing_arguments = check_required_arguments(call_data.arguments, required_arguments)
                if not result:
                    return False, f"Missing required arguments: {missing_arguments}"

                return True, None
        except Exception as e:
            # Handle validation errors for the function signature
            return False, str(e)

    # No matching function signature found
    return False, f"No matching function signature found for function: {call_data.name}"

def check_required_arguments(call_arguments, required_arguments):
    missing_arguments = [arg for arg in required_arguments if arg not in call_arguments]
    return not bool(missing_arguments), missing_arguments

def validate_enum_value(arg_name, arg_value, enum_values):
    if arg_value not in enum_values:
        raise Exception(
            f"Invalid value '{arg_value}' for parameter {arg_name}. Expected one of {', '.join(map(str, enum_values))}"
        )

def validate_argument_type(arg_name, arg_value, arg_schema):
    arg_type = arg_schema.get('type', None)
    if arg_type:
        if arg_type == 'string' and 'enum' in arg_schema:
            enum_values = arg_schema['enum']
            if None not in enum_values and enum_values != []:
                try:
                    validate_enum_value(arg_name, arg_value, enum_values)
                except Exception as e:
                    # Propagate the validation error message
                    raise Exception(f"Error validating function call: {e}")

        python_type = get_python_type(arg_type)
        if not isinstance(arg_value, python_type):
            raise Exception(f"Type mismatch for parameter {arg_name}. Expected: {arg_type}, Got: {type(arg_value)}")

def get_python_type(json_type):
    type_mapping = {
        'string': str,
        'number': (int, float),
        'integer': int,
        'boolean': bool,
        'array': list,
        'object': dict,
        'null': type(None),
    }
    return type_mapping[json_type]

def validate_and_extract_tool_calls(assistant_content):
    validation_result = False
    tool_calls = []
    error_message = None

    try:
        # wrap content in root element
        xml_root_element = f"<root>{assistant_content}</root>"
        root = ET.fromstring(xml_root_element)

        # extract JSON data
        tool_call_elements = root.findall(".//tool_call")
        if len(tool_call_elements) == 0:
            return True, None, None
        for element in tool_call_elements:
            json_data = None
            try:
                json_text = element.text.strip()

                try:
                    # Prioritize json.loads for better error handling
                    json_data = json.loads(json_text)
                except json.JSONDecodeError as json_err:
                    try:
                        # Fallback to ast.literal_eval if json.loads fails
                        json_data = ast.literal_eval(json_text)
                    except (SyntaxError, ValueError) as eval_err:
                        error_message = f"JSON parsing failed with both json.loads and ast.literal_eval:\n"\
                                        f"- JSON Decode Error: {json_err}\n"\
                                        f"- Fallback Syntax/Value Error: {eval_err}\n"\
                                        f"- Problematic JSON text: {json_text}"
                        inference_logger.error(error_message)
                        continue
            except Exception as e:
                error_message = f"Cannot strip text: {e}"
                inference_logger.error(error_message)

            if json_data is not None:
                tool_calls.append(json_data)
                validation_result = True

    except ET.ParseError as err:
        error_message = f"XML Parse Error: {err}"
        inference_logger.error(f"XML Parse Error: {err}")

    # Return default values if no valid data is extracted
    return validation_result, tool_calls, error_message

def validate_json_data(json_object, json_schema):
    valid = False
    error_message = None
    result_json = None
    print(type(json_object), type(json_schema))
    try:
        # Attempt to load JSON using json.loads
        try:
            result_json = json.loads(json_object)
        except json.decoder.JSONDecodeError:
            # If json.loads fails, try ast.literal_eval
            try:
                result_json = ast.literal_eval(json_object)
            except (SyntaxError, ValueError) as e:
                try:
                    result_json = extract_json_from_markdown(json_object)
                except Exception as e:
                    error_message = f"JSON decoding error: {e}"
                    inference_logger.info(f"Validation failed for JSON data: {error_message}")
                    return valid, result_json, error_message

        # Return early if both json.loads and ast.literal_eval fail
        if result_json is None:
            error_message = "Failed to decode JSON data"
            inference_logger.info(f"Validation failed for JSON data: {error_message}")
            return valid, result_json, error_message

        # Validate each item in the list against schema if it's a list
        if isinstance(result_json, list):
            for index, item in enumerate(result_json):
                try:
                    validate(instance=item, schema=json_schema)
                    inference_logger.info(f"Item {index+1} is valid against the schema.")
                except ValidationError as e:
                    error_message = f"Validation failed for item {index+1}: {e}"
                    break
        else:
            # Default to validation without list
            try:
                validate(instance=result_json, schema=json_schema)
            except ValidationError as e:
                error_message = f"Validation failed: {e}"

    except Exception as e:
        error_message = f"Error occurred: {e}"

    if error_message is None:
        valid = True
        inference_logger.info("JSON data is valid against the schema.")
    else:
        inference_logger.info(f"Validation failed for JSON data: {error_message}")

    return valid, result_json, error_message