import mlflow
from typing import Annotated, TypedDict
from typing import Literal, Any
from langgraph.types import Command
from mlflow.langchain.chat_agent_langgraph import ChatAgentState
from langgraph.graph import START, END, StateGraph
from agents.genie.graph import app as genie_agent
from agents.rag.graph import app as rag_agent
from agents.supervisor.resources.model import model


member_agents = ["rag", "genie"]
options = member_agents + ["FINISH"]


class Router(TypedDict):
    next: Literal[*options]


system_prompt = f"""You are a supervisor tasked with managing a conversation between the following workers: {member_agents}. The worker's capabilities are as follows.
                 
    - genie: Performs text to SQL against supply chain data tables. 
    - rag: Retrieves documentation related to Databricks and Apache Spark.

Given the following user request respond with the worker to act next. Each worker will perform a task and respond with their results and status. Call only the worker(s) that are important to the user's qeustion. Do not provide insights or summaries of the data returned by the workes. Your only job is to return the work names only. When finished respond with FINISH."""


def supervisor_node(state: ChatAgentState) -> Command[Literal[*member_agents, "__end__"]]:
    messages = [{"role": "system", "content": system_prompt}] + state['messages']
    supervisor_model = model.with_structured_output(Router)
    response = supervisor_model.invoke(messages)
    goto = response["next"]
    #goto = response.content
    if goto == "FINISH":
        goto = END
    return Command(goto=goto, update={"next": goto})
  

def rag_node(state: ChatAgentState):
  response = rag_agent.invoke(state)
  return Command(
        update={"messages": response['messages']},
        goto="supervisor",
    )


def genie_node(state: ChatAgentState):
  response = genie_agent.invoke(state)
  return Command(
        update={"messages": response['messages']},
        goto="supervisor",
    )
  

graph_builder = StateGraph(ChatAgentState)
graph_builder.add_node("supervisor", supervisor_node)
graph_builder.add_node("genie", genie_node)
graph_builder.add_node("rag", rag_node)
graph_builder.add_edge(START, "supervisor")
app = graph_builder.compile()