
from typing import Optional, Any, Generator
import uuid
from langgraph_swarm import create_handoff_tool, create_swarm
from langchain_core.messages import convert_to_openai_messages
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
import mlflow
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext)

#from swarm.rag_agent import rag_agent
#from swarm.genie_agent import genie_agent

from agents.rag.nodes import (system_message as rag_system_message,
                              route_tools as rag_route_tools)                                       
from agents.rag.tools import documentation_search_tool
from agents.rag.resources.model import model as rag_model

from agents.genie.nodes import (system_message as genie_system_message, 
                                route_tools as genie_route_tools)
from agents.genie.tools import text_to_sql_tool
from agents.genie.resources.model import model as genie_model


# RAG AGENT AND HANDOFF TOOL
genie_handoff_tool = create_handoff_tool(agent_name="genie", 
                                       description="Transfer to genie to help with supply chain data analysis and questions related to tabular data that potentially require sql query generation")

rag_tools = [documentation_search_tool, genie_handoff_tool]
rag_model_with_tools = rag_model.bind_tools(rag_tools, tool_choice="auto")
rag_tool_node = ToolNode(tools=rag_tools)


def rag_chatbot(state: MessagesState):
  messages = [{"role": "system", "content": rag_system_message}] + state["messages"]
  response = rag_model_with_tools.invoke(messages)
  return {"messages": [response]}



graph_builder = StateGraph(MessagesState)
graph_builder.add_node("rag_chatbot", rag_chatbot)
graph_builder.add_node("rag_tools", rag_tool_node)
graph_builder.add_conditional_edges("rag_chatbot", rag_route_tools,
    {"tools": "rag_tools", 
      END: END}
  ) 
graph_builder.add_edge(START, "rag_chatbot")
graph_builder.add_edge("rag_tools", "rag_chatbot")
rag_agent = graph_builder.compile(name="rag")

# RAG AGENT AND HANDOFF TOOL
rag_handoff_tool = create_handoff_tool(agent_name="rag", 
                                       description="Transfer to rag to help with any questions related to Databricks, Apache Spark, Delta, or any other question related to Databrick's associated technologies")


genie_tools = [text_to_sql_tool, rag_handoff_tool]
genie_model_with_tools = genie_model.bind_tools(genie_tools, tool_choice="auto")
genie_tool_node = ToolNode(tools=genie_tools)


def genie_chatbot(state: MessagesState):
  messages = [{"role": "system", "content": genie_system_message}] + state["messages"]
  response = genie_model_with_tools.invoke(messages)
  return {"messages": [response]}

graph_builder = StateGraph(MessagesState)
graph_builder.add_node("genie_chatbot", genie_chatbot)
graph_builder.add_node("genie_tools", genie_tool_node)
graph_builder.add_conditional_edges("genie_chatbot", genie_route_tools,
    {"tools": "genie_tools", 
      END: END}
  ) 
graph_builder.add_edge(START, "genie_chatbot")
graph_builder.add_edge("genie_tools", "genie_chatbot")
genie_agent = graph_builder.compile(name="genie")

# SWARM AGENT
workflow = create_swarm(
    [rag_agent, genie_agent],
    default_active_agent="rag"
)

swarm = workflow.compile()

class SwarmAgent(ChatAgent):
  def __init__(self):
    self.agent = swarm
    mlflow.langchain.autolog()


  def predict(self, 
              messages: list[ChatAgentMessage],
              context: Optional[ChatContext] = None,
              custom_inputs: Optional[dict[str, Any]] = None
              ) -> ChatAgentResponse:
    
    request = {"messages": self._convert_messages_to_dict(messages)}

    messages = []
    for event in self.agent.stream(request, stream_mode="updates"):
        for agent_name, agent_messages in event.items():
            for message in agent_messages['messages']:
                converted_message = convert_to_openai_messages(message)
                openai_format_with_id = converted_message | {"id": str(uuid.uuid4())}
                messages.extend([ChatAgentMessage(**openai_format_with_id)])
    return ChatAgentResponse(messages=messages)


  def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:

        request = {"messages": self._convert_messages_to_dict(messages)}

        for event in self.agent.stream(request, stream_mode="updates"):
            for agent_name, agent_messages in event.items():
                for message in agent_messages['messages']:
                    converted_message = convert_to_openai_messages(message)
                    openai_format_with_id = converted_message | {"id": str(uuid.uuid4())}
                    yield ChatAgentChunk(**{"delta": openai_format_with_id})                           
            

AGENT = SwarmAgent()
mlflow.models.set_model(AGENT)
