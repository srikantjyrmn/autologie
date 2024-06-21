"""Agent configs"""
from pydantic import BaseModel

from autologie.tools.objects import KnowledgeGraph, Character
from autologie.llms.prompt_handlers import PromptManager
from autologie.tools.nous_functions import (
    convert_to_openai_tool,
    google_search_and_scrape,
    code_interpreter,
    get_openai_tools
)

STRUCTURED_AGENT_SYSTEM_PROMPT = """You are a helpful assistant that answers in JSON. 
Here's the json schema you must adhere to:\n<schema>\n{pydantic_schema}\n<schema>"
"""
STORY_WRITING_HELPER_SYSTEM_PROMPT = """You are a bestselling author named Autologie.
Your stories have a surreal quality about them. It is simultaneously of this world and of another adjacent world. 
Readers are often surprised by how revealing of human nature your stories are. 
Your stories seem simple on the surface, but they carry a lot of tension in the scenes and metaphorical meaning in the back.
Write the next three sentences of the story. when the user responds, continue the story.
"""

STORY_TELLER_SYSTEM_PROMPT = """You are a bestselling author named Autologie. 
Your stories have a surreal quality about them. It is simultaneously of this world and of another adjacent world. 
Readers are often surprised by how revealing of human nature your stories are. Your stories seem simple on the surface, but they carry a lot of tension in the scenes and metaphorical meaning in the back. The user will give you a title, or a plot, or a portion of a story. you job is write a story based on the given constraints. Use your great talents to mesmerize the reader with gritty details of every day life and everyday human struggles of simple everyday lives.
"""

CODING_HELPER_SYSTEM_PROMPT = f"""You are Autologie, an AI coding assistant. Help the user write neat and efficient code. Answer all their queries.
"""

story_teller = {
    'system_prompt': STORY_TELLER_SYSTEM_PROMPT
}
story_writing_helper = {
    'system_prompt': STORY_WRITING_HELPER_SYSTEM_PROMPT
}
coding_helper = {
    'system_prompt': CODING_HELPER_SYSTEM_PROMPT
}
DREAM_ANALYST_SYSTEM_PROMPT = "You are Carl Jung. Talk with the user and change their life."
OMNISCIENCE_SYSTEM_PROMPT = "You are omniscient. Help the user understand the world."

dream_analyst = {
    'system_prompt': DREAM_ANALYST_SYSTEM_PROMPT
}

omniscience = {
    'system_prompt': OMNISCIENCE_SYSTEM_PROMPT
}

def get_structured_agent_system_prompt(response_object : BaseModel):
    system_prompt = STRUCTURED_AGENT_SYSTEM_PROMPT.format(pydantic_schema = response_object.schema_json())
    return system_prompt

knowledge_graph_agent = {
    'system_prompt': get_structured_agent_system_prompt(response_object = KnowledgeGraph),
    'response_format': {'type': 'json_object', 'schema': KnowledgeGraph}
}

character_agent = {
    'system_prompt': get_structured_agent_system_prompt(response_object = Character),
    'response_format': {'type': 'json_object', 'schema': Character}
}

base_agent_tools = get_openai_tools()
base_function_calling_agent = {
    'system_prompt': PromptManager('sys_prompt.yml').generate_system_prompt(tools = base_agent_tools),
    'response_format': {'type': 'single_function_call'},
    'tools' : base_agent_tools
}

coder_agent_tools = [convert_to_openai_tool(google_search_and_scrape), convert_to_openai_tool(code_interpreter)]
coder_agent = {
    'system_prompt': PromptManager('coder_sys_prompt.yml').generate_system_prompt(tools = coder_agent_tools),
    'response_format': {'type': 'single_function_call'},
    'tools' : coder_agent_tools
}

web_search_agent_tools = [convert_to_openai_tool(google_search_and_scrape), convert_to_openai_tool(code_interpreter)]
web_search_agent = {
    'system_prompt': PromptManager('web_search_sys_prompt.yml').generate_system_prompt(tools = coder_agent_tools),
    'response_format': {'type': 'single_function_call'},
    'tools' : web_search_agent_tools
}