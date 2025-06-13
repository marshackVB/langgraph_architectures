
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

from agents.genie.graph import compile_genie_agent
from agents.genie.resources.model import model as genie_model
from agents.genie.tools import text_to_sql_tool

from agents.rag.graph import compile_rag_agent
from agents.rag.resources.model import model as rag_model
from agents.rag.tools import documentation_search_tool


# CONFIGURE RAG AGENT
genie_handoff_tool = create_handoff_tool(agent_name="genie_agent", 
                                        description="Transfer to genie to help with supply chain data analysis and questions related to tabular data that potentially require sql query generation")

rag_tools = [documentation_search_tool, genie_handoff_tool]
rag_model_with_tools = rag_model.bind_tools(rag_tools, tool_choice="auto")
rag_tool_node = ToolNode(tools=rag_tools)

rag_agent = compile_rag_agent(rag_model_with_tools, 
                              rag_tool_node)


# CONFIGURE GENIE AGENT
rag_handoff_tool = create_handoff_tool(agent_name="rag_agent", 
                                       description="Transfer to rag to help with any questions related to Databricks, Apache Spark, Delta, or any other question related to Databrick's associated technologies")


genie_tools = [text_to_sql_tool, rag_handoff_tool]
genie_model_with_tools = genie_model.bind_tools(genie_tools, tool_choice="auto")
genie_tool_node = ToolNode(tools=genie_tools)

genie_agent = compile_genie_agent(genie_model_with_tools, 
                                  genie_tool_node)

# SWARM AGENT
workflow = create_swarm(
    [rag_agent, genie_agent],
    default_active_agent="rag_agent"
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
