#!/bin/bash

# Start the server
echo "Starting the LlamaCpp Server..."
source ./server/start_llama_cpp_server.sh &

# Start the interface
echo "Starting the Autologie Interface..."
python3 -m autologie.interface.interface &

wait