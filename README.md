# autologie

Python repository to build and work with function calling AI Agents.

To use the package, first clone the repository:
```
git clone 
```

Move into the directory, then activate the poetry environment:
```
poetry shell
poetry install
```

Start the Interface
```
python3 -m autologie.interface
```

Use the package
```
from autologie import AgentClient
from autologie.agent_configs import omniscience, character_agent, function_calling_agent
chat_agent = AgentClient(**omniscience)
json_agent = AgentClient(**character_agent)
function_calling_agent = AgentClient(**function_calling_agent)
```