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
    def __init__(self, file_name):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.file_name = file_name

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
        prompt_path = os.path.join(self.script_dir, 'prompt_assets', self.file_name)
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
        prompt_path = os.path.join(self.script_dir, 'prompt_assets', self.file_name)
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
