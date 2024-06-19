"""#TODO"""
import streamlit as st
import gradio as gr
from autologie.llms.llms import LlmClient

# Function to initialize LlmClient
def initialize_llm(agent_type, system_prompt):
    system_prompt_text = system_prompt_map.get(agent_type, "You are AI.")
    return LlmClient(system_prompt=system_prompt_text)

# Function to handle chat interactions
def chat(message, history, agent_type, system_prompt, t, m, llm):
    print(message)
    prompt = message['text']
    num_files = len(message["files"])
    print(f"You uploaded {num_files} files")
    
    response = llm.chat(prompt)
    response_str = response.assistant_message
    response = f"{response_str} You uploaded {num_files} files\n\nSystem prompt: {system_prompt}\nMessage: {prompt}."
    
    return response

# Global variables or constants
system_prompt_map = {
    'OmniscientSuperIntelligence': "You are OmniscientSuperIntelligence.",
    'DreamAnalyst': "You are DreamAnalyst."
}

# Initial setup for LlmClient
initial_agent_type = "DreamAnalyst"  # Initial values can be set here
initial_system_prompt = system_prompt_map.get(initial_agent_type, "You are AI.")
llm_instance = initialize_llm(initial_agent_type, initial_system_prompt)

# Gradio interface setup
system_prompt_input = gr.Textbox("", label="System Prompt", lines=10, interactive=True)
model_name_selector = gr.Textbox("llamacpp/nous_hermes", label="Model Name")
slider = gr.Slider(10, 100, render=False)
agent_type_selector = gr.Dropdown(
    choices=['OmniscientSuperIntelligence', 'DreamAnalyst'],
    value=initial_agent_type
)

# Define chat interface with the `chat` function as the handler
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

# Function to embed Gradio into Streamlit using HTML
def embed_gradio_into_streamlit():
    gradio_interface_html = gradio_chat_box.launch()
    st.markdown(gradio_interface_html, unsafe_allow_html=True)

# Streamlit application code
def main():
    st.title('Streamlit + Gradio Chat Interface')
    st.write("Streamlit application content goes here.")
    embed_gradio_into_streamlit()

if __name__ == '__main__':
    main()
