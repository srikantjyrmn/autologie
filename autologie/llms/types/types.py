"""Pydantic Classes for API interface objects like Response and Prompt. 
Classes and Methods to generate a properly formatted ChatMessage, 
according to pre-defined ChatTemplate
#TODO.
"""

from typing import Optional, Union, Literal, TypedDict
from enum import auto
from pydantic import BaseModel, Field
from strenum import LowercaseStrEnum

from .payloads import (
    Payload, OllamaCompletionPayload, LlamacppCompletionPayload, OpenaiChatPayload
)
from .responses import CompletionApiResponse, ChatApiResponse, OllamaCompletionResponse
from .generation_settings import (
    LlamacppGenerationSettings, OllamaGenerationSettings, OpenaiGenerationSettings)

class LlmEndpointTypes(LowercaseStrEnum):
    """Types of API endpoints. 
    Values: [openai, completion, chat]
    Determines the structure of payload going in, 
    and the format of responses coming out. 
    Both are fully determined by (Server, Endpoint).
    """
    OPENAI_CLIENT = auto()
    OPENAI_API = auto()
    CHAT = auto()
    COMPLETION = auto()

class LlmServers(LowercaseStrEnum):
    """LLM Servers being used.
    Values: [llamacpp, ollama, openai]
    """
    LLAMACPP = auto()
    OLLAMA = auto()
    OPENAI = auto()

class ApiClientSpec(BaseModel):
    """Base class to define a model endpoint to get responses from.
    """
    base_url: str = 'http://localhost:8000/v1/completions'
    api_key: str = ''
    endpoint_type: Literal['openai_api', 'openai_client', 'completion', 'chat'] = 'completion'
    server: LlmServers = 'llamacpp'

    model: str = 'nous_hermes'
    chat_template: Union[None, Literal['default'], str] = Field(
        alias = 'chat_format',
        default = 'chatml')
    hf_repo_id: Optional[str] = None
    hf_local_path: Optional[str] = None
    system_prompt: Optional[str] = None

class ApiClientHandlers(TypedDict):
    """A dictionary defining the handlers and object types for a given server and endpoint."""
    server: LlmServers
    endpoint_type: LlmEndpointTypes
    payload_object: Payload
    response_object: Union[CompletionApiResponse, ChatApiResponse, OllamaCompletionResponse]
    generation_settings: LlamacppGenerationSettings = LlamacppGenerationSettings

class LlamacppCompletionHandlers(ApiClientHandlers):
    """Handlers for a llamacpp completion endpoint"""
    server: LlmServers = 'llamacpp'
    endpoint_type: LlmEndpointTypes = 'completion'
    payload_object: Payload = LlamacppCompletionPayload
    response_object = CompletionApiResponse
    generation_settings = LlamacppGenerationSettings

class OllamaCompletionHandlers(ApiClientHandlers):
    """Handlers for an Ollama completion client"""
    server: LlmServers = 'ollama'
    endpoint_type: LlmEndpointTypes = 'completion'
    payload_object: Payload = OllamaCompletionPayload
    response_object = OllamaCompletionResponse
    generation_settings = OllamaGenerationSettings

class OpenaiApiHandlers(ApiClientHandlers):
    """Handlers for an Openai Api compatible endpoint via ApiClientConnector. #TODO"""
    server: LlmServers
    endpoint_type: LlmEndpointTypes = 'openai_api'
    payload_object: Payload = OpenaiChatPayload
    response_object = ChatApiResponse
    generation_settings = OpenaiGenerationSettings

class OpenaiClientHandlers(ApiClientHandlers):
    """Handlers for an Openai Api compatible endpoint via Openai python client."""
    server: LlmServers = 'llamacpp'
    endpoint_type: LlmEndpointTypes = 'completion'
    payload_object: Payload = LlamacppCompletionPayload
    response_object = CompletionApiResponse
    generation_settings = LlamacppGenerationSettings

class ClientHandlerFactory(BaseModel):
    """Factory for client handlers. Initialized with server and endpoint.
    For custom types, one can also pass in argument."""
    server: LlmServers
    endpoint_type: LlmEndpointTypes

    def get_handlers(self,
                    **kwargs
                ) -> ApiClientHandlers:
        """Return a set of handlers for a given server endpoint."""
        factories = {
            'llamacpp/completion': LlamacppCompletionHandlers,
            'llamacpp/openai_api': OpenaiApiHandlers,
            'ollama/completion': OllamaCompletionHandlers,
        }
        server_endpoint = f"{self.server}/{self.endpoint_type}"
        if server_endpoint in factories:
            return factories[server_endpoint]
        return ClientHandlerFactory(
            server = self.server,
            endpoint_type = self.endpoint_type,
            **kwargs
        )
