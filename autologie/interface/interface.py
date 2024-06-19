from autologie.agents.agent_configs import dream_analyst, omniscience, function_calling_agent, knowledge_graph_agent, character_agent
from autologie.llms.llms import AgentClient
import gradio as gr
from typing import TypedDict
# Initialize LlmClient instance globally
class State(TypedDict):
    system_prompt:str
    agent_type:str
    llm: AgentClient

agent_config_map = {
    'OmniscientSuperIntelligence': omniscience,
    'DreamAnalyst': dream_analyst,
    'CharacterAgent': character_agent,
    'KnowledgeGraphAgent': knowledge_graph_agent,
    'functionCaller': function_calling_agent
}

def read_markdown_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return content

def initialize_llm(agent_type = "", system_prompt = ""):
    agent_spec = agent_config_map.get(agent_type, omniscience)
    llm = AgentClient(**agent_spec)
    state = {
        'agent_type': agent_type,
        'system_prompt': system_prompt,
        'llm': llm
    }
    return State(**state)

state = initialize_llm()
def chat(message, history, agent_type, system_prompt, t, m):
    global state
    
    if state['llm'] is None or state['agent_type'] != agent_type or state['system_prompt'] != system_prompt:
        state = initialize_llm(agent_type, system_prompt)
    
    print(message)
    prompt = message['text']  # + read_markdown_file(message['files'][0])  # You can uncomment this if needed
    num_files = len(message["files"])
    print(f"You uploaded {num_files} files")
    
    response = state['llm'].chat(prompt)
    response_str = response.assistant_message
    response = f"{response_str} You uploaded {num_files} files\n\nSystem prompt: {system_prompt}\nMessage: {prompt}." + f"Agent config: {state['llm'].response_spec}"
    
    return response

# Gradio interface setup
system_prompt_input = gr.Textbox("", label="System Prompt", lines=10, interactive=True)
model_name_selector = gr.Textbox("llamacpp/nous_hermes", label="Model Name")
slider = gr.Slider(10, 100, render=False)
agent_type_selector = gr.Dropdown(
    choices=['OmniscientSuperIntelligence', 'DreamAnalyst', 'functionCaller', 'StructuredSchema'],
    value="DreamAnalyst"
)

gradio_chat_box = gr.ChatInterface(
    chat,
    chatbot=gr.Chatbot(height=300),
    title="Model Chat Interface",
    description="Standard",
    theme="soft",
    cache_examples=False,
    retry_btn=None,
    undo_btn="Delete Previous",
    clear_btn="Clear",
    multimodal=True,
    #examples=[{'text': "Hello", 'files': []}],
    additional_inputs=[agent_type_selector, system_prompt_input, slider, model_name_selector],
    autofocus=False
)

if __name__ == '__main__':
    gradio_chat_box.launch()
