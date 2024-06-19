from pydantic import BaseModel
from typing import List, Optional
class Node(BaseModel):
    """A node in the Knowledge Graph. COmposed of an id, a unique identifier and name: a string identifier."""
    id: int
    name: str

class Edge(BaseModel):
    """An edge in the knowledge graph. COMposed of 
        start_node_idLint the node_id of the start node, 
        end_node_id
        and relaitonship_type

    Args:
        BaseModel (_type_): _description_
    """
    start_node_id: int
    end_node_id:int
    relationship_type:str
    
class KnowledgeGraph(BaseModel):
    """"A knowledge graph, composed of nodes and edges."""
    nodes: List[Node]
    edges: List[Edge]

class Character(BaseModel):
    """An example pydantic object to use for structured output generation.
    """
    name: str
    species: str
    role: str
    personality_traits: Optional[List[str]]
    special_attacks: Optional[List[str]]

    class Config:
        schema_extra = {
            "additionalProperties": False
        }