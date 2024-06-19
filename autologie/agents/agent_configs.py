from autologie.tools.objects import KnowledgeGraph, Character
from autologie.llms.response_handlers import PromptManager
from autologie.tools.nous_functions import convert_to_openai_tool, google_search_and_scrape, get_current_stock_price
from pydantic import BaseModel
STRUCTURED_AGENT_SYSTEM_PROMPT = """You are a helpful assistant that answers in JSON. Here's the json schema you must adhere to:\n<schema>\n{pydantic_schema}\n<schema>"
"""

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

tools = [convert_to_openai_tool(google_search_and_scrape), convert_to_openai_tool(get_current_stock_price)]
function_calling_agent = {
    'system_prompt': PromptManager().generate_system_prompt(tools = tools),
    'response_format': {'type': 'single_function_call'},
    'tools' : tools
}