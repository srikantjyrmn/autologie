"""Tables t store memory for agents, and memory for autologie conversations"""
from datetime import datetime
from typing import Optional
from sqlmodel import Field, SQLModel, ARRAY, JSON, Relationship

class User(SQLModel, table=True):
    id: Optional[int] = Field(default = None, primary_key = True)
    name: str
    created_at: datetime

class Agent(SQLModel, table = True):
    id: Optional[int] = Field(default = None, primary_key = True)
    name: str
    type: str
    system_prompt: str
    response_schema:str
    tools: ARRAY[str]
    created_at: datetime

class Messages(SQLModel, table=True):
    id: Optional[int] = Field(default = None, primary_key = True)
    conversation_id: int = Field(default = None, foreign_key="conversations.id")
    role: str
    content: str
    created_at: datetime

class Conversations(SQLModel, table=True):
    id: Optional[int] = Field(default = None, primary_key = True)
    agent_id: int = Field(default = None, foreign_key="agents.id")
    user_id: int = Field(default = None, foreign_key="users.id")
    created_at: datetime
    
    messages : list["Messages"] = Relationship(back_populates = 'conversations')