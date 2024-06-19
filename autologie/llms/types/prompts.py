import os
from enum import auto
from strenum import LowercaseStrEnum
from pydantic import BaseModel, Field, computed_field
from functools import cached_property

class ChatTemplateEnum(LowercaseStrEnum):
    """Enums for Chat  Templates. #TODO: We must be able to associate HF metadata to chat templates.
    #TODO: Supoort more ChatTemplates, as needed by models used."""
    CHATML = auto()
    VICUNA = auto()
    ZEPHYR = auto()

def get_chat_template(chat_template):
    """read chat template from jinja file"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(script_dir, 'chat_templates', f"{chat_template}.j2")
    print(f"Reading Chat template from {template_path}")
    if not os.path.exists(template_path):
        print((f"Template file not found: {chat_template}"))
        return None
    try:
        with open(template_path, 'r') as file:
            template = file.read()
        return template
    except Exception as e:
        print(f"Error loading template: {e}")
        return None

class ChatTemplate(BaseModel):
    """Base Class for a ChatTemplate. 
    From the StringEnum defined in name we get a template by reading 
    the template file located in chat_templates."""
    name: ChatTemplateEnum
    @computed_field
    @cached_property
    def template(self) -> str:
        """The Jinja Template for the ChatTemplate."""
        return get_chat_template(self.name)


    def get_dict(self):
        return {
            'role' : self.role,
            'content': self.content
        }

class PromptFormat(LowercaseStrEnum):
    """Deprecated"""
    CHATML_FUNCTION_CALLING = auto()
    CHATML = auto()
    MISTRAL = auto()
    VICUNA = auto()
    LLAMA_2 = auto()
    SYNTHIA = auto()
    NEURAL_CHAT = auto()
    SOLAR = auto()
    OPEN_CHAT= auto()
    ALPACA = auto()
    LLAMA_3 = auto()
    PHI_3 = auto()
    B22 = auto()
    CODE_DS = auto()

#%% Types for more complex Prompts
class PromptSchema(BaseModel):
    """A prompt schema to be input for a function calling agent."""
    Role: str
    Objective: str
    Tools: str
    Examples: str
    Schema: str
    Instructions: str