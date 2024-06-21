# Activate virtual environment
source ./server/llama_cpp_server_env/bin/activate
pip install -r ./server/requirements.txt
# Start the server
python3 -m llama_cpp.server --config_file ./server/model_config.json
