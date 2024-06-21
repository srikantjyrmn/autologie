"""Tables t store memory for agents, and memory for autologie conversations"""
from datetime import datetime
from typing import Optional
from sqlmodel import Field, SQLModel, ARRAY, Relationship, create_engine, JSON, Session
from typing import Any, List
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column

class Users(SQLModel, table=True):
    id: Optional[int] = Field(default = None, primary_key = True)
    name: str
    created_at: datetime

class Agents(SQLModel, table = True):
    id: Optional[int] = Field(default = None, primary_key = True)
    name: str
    stype: str
    system_prompt: str
    response_schema:str
    #tools: ARRAY[float]
    created_at: datetime

class Conversations(SQLModel, table=True):
    id: Optional[int] = Field(default = None, primary_key = True)
    agent_id: int = Field(default = None, foreign_key="agents.id")
    user_id: int = Field(default = None, foreign_key="users.id")
    created_at: datetime
    messages : list["Messages"] = Relationship(back_populates = 'conversation')

class Messages(SQLModel, table=True):
    id: Optional[int] = Field(default = None, primary_key = True)
    conversation_id: int = Field(default = None, foreign_key="conversations.id")
    role: str
    content: str
    embedding: Any = Field(sa_column = Column(Vector(3)))
    created_at: datetime
    conversation: None|Conversations= Relationship(back_populates="messages")

class Files(SQLModel, table=True):
    id: Optional[int] = Field(default = None, primary_key = True)
    name:str
    size: int
    #metadata: JSON
    chunks : List["Chunks"] = Relationship(back_populates = 'file')

class Chunks(SQLModel, table=True):
    id: Optional[int] = Field(default = None, primary_key = True)
    file_id: int = Field(default=None, foreign_key="files.id")
    size: int
    #metadata: JSON
    contents : str
    embedding : Any = Field(sa_column = Column(Vector(3)))
    file: None|Files = Relationship(back_populates="chunks")
class Search_Results(SQLModel, table=True):
    id: Optional[int] = Field(default = None, primary_key = True)
    name:str
    size: int
    #metadata: JSON
    chunks : List["Search_Result_Chunks"] = Relationship(back_populates = 'search_result')

class Search_Result_Chunks(SQLModel, table=True):
    d: Optional[int] = Field(default = None, primary_key = True)
    search_result_id: int = Field(default=None, foreign_key="search_results.id")
    size: int
    #metadata: JSON
    contents : str
    embedding : Any = Field(sa_column = Column(Vector(3)))
    search_result: None|Search_Results = Relationship(back_populates="chunks")
sqlite_file_name = "db1.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

engine = create_engine(sqlite_url, echo=True)

psql_url = ""
psql_engine = create_engine("postgresql+psycopg2://reorganism:admin@localhost/postgres", echo=True)
def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def create_db_and_tables_psql():
    SQLModel.metadata.create_all(psql_engine)

def basic_things():
    with Session(psql_engine) as session:
        simpe_agent = Agents(
        name = 'simple_1',
        stype= 'unstrcutred',
        system_prompt="you are simple",
        response_schema="k",
        created_at=datetime.now()
        )
        user = Users(
            name = 'srikant', created_at=datetime.now()
        )
        
        session.add(simpe_agent)
        session.add(user)
        
        session.commit()

        session.refresh(simpe_agent)
        session.refresh(user)
        

        #hero_spider_boy.team = team_preventers
        #session.add(hero_spider_boy)
        session.commit()
        
        conversation = Conversations(
            agent_id = simpe_agent.id,
            user_id = user.id,
            created_at=datetime.now(),
            messages=[]
        )
        session.add(conversation)
        session.commit()
        session.refresh(conversation)
        message = Messages(
            content="Hi",
            role="user",
            embedding=[0,0,0],
            created_at=datetime.now(),
            conversation_id=conversation.id
        )
        session.add(message)
        session.commit()
        session.refresh(message)
        print("Created agent:", simpe_agent)
        print("Created user:", user)
        print("Created convo:", conversation)
        
        print("Created convo:", message)
        

def main():
    #create_db_and_tables()
    create_db_and_tables_psql()

if __name__ == "__main__":
    main()