"""Interface"""
from typing import TypedDict
import gradio as gr
from autologie.agents.agent_configs import (
    dream_analyst, omniscience,
    knowledge_graph_agent, character_agent,
    story_writing_helper, story_teller, coding_helper,
    coder_agent, base_function_calling_agent, web_search_agent
)
from autologie.llms.llms import AgentClient

# Initialize LlmClient instance globally
class State(TypedDict):
    system_prompt:str
    agent_type:str
    llm: AgentClient

agent_config_map = {
    'ChatAgent_OmniscientSuperIntelligence': omniscience,
    'ChatAgent_DreamAnalyst': dream_analyst,
    'ChatAgent_StoryTeller':story_teller,
    'ChatAgent_StoryWritingHelper': story_writing_helper,
    'ChatAgent_CodingHelper': coding_helper,

    'ObjAgent_Character': character_agent,
    'ObjAgent_KnowledgeGraph': knowledge_graph_agent,

    'AgentCodeyBanks': coder_agent,
    'AgentWebSearch': web_search_agent,
    'AgentNous': base_function_calling_agent
}

default_agent = 'AgentWebSearch'
default_config = agent_config_map[default_agent]
def read_markdown_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return content

def initialize_llm(agent_type = default_agent, system_prompt = default_config['system_prompt']):
    agent_spec = agent_config_map.get(agent_type, story_teller)
    llm = AgentClient(**agent_spec)
    state = {
        'agent_type': agent_type,
        'system_prompt': system_prompt,
        'llm': llm
    }
    print(state)
    return State(**state)

state = initialize_llm()
def chat(message, history, agent_type, system_prompt, t, m):
    global state

    if (state['llm'] is None) or (state['agent_type'] != agent_type) or (state['system_prompt'] != system_prompt):
        state = initialize_llm(agent_type, system_prompt)

    print(message)
    prompt = message['text']
    num_files = len(message["files"])
    print(f"You uploaded {num_files} files")

    response = state['llm'].chat(prompt)
    response_str = response.assistant_message
    response = f"{response_str}"
    return response

# Gradio interface setup
system_prompt_input = gr.Textbox(state['system_prompt'], label="System Prompt", lines=10, interactive=True)
model_name_selector = gr.Textbox("llamacpp/nous_hermes", label="Model Name")
slider = gr.Slider(10, 100, render=False)
agent_type_selector = gr.Dropdown(
    label = "Select Agent",
    choices = list(agent_config_map.keys()),
    value = state['agent_type']
)

def dropdown_selector(sys_prompt_input):
    
    return sys_prompt_input
agent_type_selector.change()
gradio_chat_box = gr.ChatInterface(
    chat,
    chatbot=gr.Chatbot(show_copy_button=True),#height=300),
    title="Autologie Chat Interface",
    description = f"You are chatting with {state['agent_type']}",
    theme="soft",
    cache_examples=False,
    #retry_btn=None,
    #undo_btn="Delete Previous",
    #clear_btn="Clear",
    multimodal=True,
    #examples=[{'text': "Hello", 'files': []}],
    additional_inputs = [agent_type_selector, system_prompt_input,
                        slider, model_name_selector],
    autofocus=True,
    fill_height=True
)

if __name__ == '__main__':
    gradio_chat_box.launch()
    #pass