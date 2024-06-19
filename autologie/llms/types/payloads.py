"""Payloads"""
from functools import cached_property
from typing import List, Dict, Optional, Literal, Union
from pydantic import BaseModel, computed_field, Field
from .generation_settings import (
    LlamacppGenerationSettings, OpenaiGenerationSettings, OllamaGenerationSettings
)
from .messages import ChatMessage, OpenaiToolFormat, FunctionSignature
from .responses import OpenaiResponseFormat

class LlamacppCompletionPayload(BaseModel):
    """Llamacpp Completion Payload"""
    prompt: str
    model: str
    generation_settings: LlamacppGenerationSettings = LlamacppGenerationSettings()
    grammar: str | None = None
    stream: bool = False
    raw: bool = True

    @computed_field
    @cached_property
    def payload_dict(self) -> Dict:
        self_dict = self.__dict__.copy()
        self_dict.update(self_dict.pop('generation_settings'))
        return self_dict

class LlamacppChatPayload(BaseModel):
    """Llamacpp Completion Payload"""
    messages: List[ChatMessage]
    model: str
    generation_settings: LlamacppGenerationSettings = LlamacppGenerationSettings()
    grammar: str | None = None
    stream: bool = False
    raw: bool = True

    @computed_field
    @cached_property
    def payload_dict(self) -> Dict:
        self_dict = self.__dict__.copy()
        self_dict.update(self_dict.pop('generation_settings'))
        return self_dict

class OpenaiPayload(BaseModel):
    """OpenAI type CLass for a payload sent to a CompletionEndpoint"""
    messages: List
    model: str = Field(alias = "model",
        description = "Name of the model to get the response from."
    )
    response_format: OpenaiResponseFormat | None
    tools: OpenaiToolFormat | None
    options: OpenaiGenerationSettings = Field(
        alias = 'generation_settings',
        default = OpenaiGenerationSettings()
    )
    stream: bool = False

    @computed_field
    @cached_property
    def payload_dict(self) -> Dict:
        """Method to get Dictionary of generation settings for inference.
        This method will handle aliases for endpoints. Default: Ollama. 
        LlamaCpp has the aliases."""
        self_dict = self.__dict__.copy()
        self_dict.update(self_dict.pop('options').__dict__)
        self_dict.update(self_dict.pop('response_format').__dict__)
        self_dict.update(self_dict.pop('tools').__dict__)
        return self_dict


class OpenaiChatPayload(BaseModel):
    """The chat payload that goes into OpenAI API endpoints"""
    messages: List[ChatMessage]
    model: str
    #functions: List[FunctionDefinition]
    #function_call: Literal["none", "auto"]
    response_format : Optional[OpenaiResponseFormat] = None
    tools: Optional[List[FunctionSignature]] = None
    tool_choice: Union[None, Literal["none", "auto"]] = None
    generation_settings: OpenaiGenerationSettings = OpenaiGenerationSettings()

    @computed_field
    @cached_property
    def payload_dict(self) -> Dict:
        """Payload"""
        self_dict = self.__dict__.copy()
        self_dict.update(self.generation_settings.__dict__)
        return self_dict


class OllamaCompletionPayload(BaseModel):
    """Pydantic class for an Ollama Completion Payload.
    .payload_dict contains the payload that has to be sent to Ollama."""
    prompt: str
    model: str
    options: OllamaGenerationSettings = OllamaGenerationSettings()
    stream: bool = False
    raw: bool = True

    @computed_field
    @cached_property
    def payload_dict(self) -> Dict:
        self_dict = self.__dict__.copy()
        self_dict['options'] = self_dict["options"].__dict__
        return self_dict

class OllamaChatPayload(BaseModel):
    """Ollama type CLass for a payload sent to a CompletionEndpoint
    #WhyKeep?"""
    messages: List
    model: str = Field(alias = "model",
        description = "Name of the model to get the response from."
    )
    options: OllamaGenerationSettings = Field(
        alias = 'generation_settings',
        default = OllamaGenerationSettings()
    )
    stream: bool = False

    @computed_field
    @cached_property
    def payload_dict(self) -> Dict:
        """Method to get Dictionary of generation settings for inference.
        This method will handle aliases for endpoints. Default: Ollama. 
        LlamaCpp has the aliases."""
        self_dict = self.__dict__.copy()
        self_dict['options'] = self_dict.pop('options').__dict__
        return self_dict

Payload = Union[LlamacppChatPayload,
                LlamacppCompletionPayload,
                OllamaChatPayload,
                OllamaCompletionPayload,
                OpenaiChatPayload]