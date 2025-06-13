from typing import Optional, Any, Generator
import uuid
import mlflow
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext)
from databricks_langchain import ChatDatabricks
from langchain_core.messages import convert_to_openai_messages
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import ToolNode

from agents.genie.graph import compile_genie_agent
from agents.genie.resources.model import model as genie_model
from agents.genie.tools import text_to_sql_tool

from agents.rag.graph import compile_rag_agent
from agents.rag.resources.model import model as rag_model
from agents.rag.tools import documentation_search_tool


# CONFIGURE SUB AGENTS
# Genie
genie_tools = [text_to_sql_tool]
genie_model_with_tools = genie_model.bind_tools(genie_tools, tool_choice="auto")
genie_tool_node = ToolNode(tools=genie_tools)

genie_agent = compile_genie_agent(genie_model_with_tools, 
                                  genie_tool_node)
# Rag
rag_tools = [documentation_search_tool]
rag_model_with_tools = rag_model.bind_tools(rag_tools, tool_choice="auto")
rag_tool_node = ToolNode(tools=rag_tools)

rag_agent = compile_rag_agent(rag_model_with_tools, 
                              rag_tool_node)


# CONFIGURE THE SUPERVISOR
config = mlflow.models.ModelConfig(development_config="config.yaml")
supervisor_config = config.get("agents").get("supervisor")[0]
model = ChatDatabricks(endpoint=supervisor_config.get('llm'), 
                       extra_params=supervisor_config.get('llm_parameters'))

system_prompt = f"""You are a supervisor tasked with calling the below agents to fullfill the user's request.
                 
    - genie: Performs text to SQL against supply chain data tables and analyses results. 
    - rag: Retrieves documentation related to Databricks and Apache Spark and answers questions related to these topics.

Assign work to one agent at a time, do not call agents in parallel. Do not do any work yourself.
"""


supervisor = create_supervisor(
    model=model,
    agents=[rag_agent, genie_agent],
    prompt=system_prompt,
    add_handoff_back_messages=True,
    output_mode="full_history",
).compile()


# DEFINE MLFLOW CHATAGENT
class SupervisorAgent(ChatAgent):
  def __init__(self):
    self.agent = supervisor
    mlflow.langchain.autolog()


  def predict(self, 
              messages: list[ChatAgentMessage],
              context: Optional[ChatContext] = None,
              custom_inputs: Optional[dict[str, Any]] = None
              ) -> ChatAgentResponse:
    
    request = {"messages": self._convert_messages_to_dict(messages)}

    messages = []
    for event in supervisor.stream(request, stream_mode="updates"):
        for agent_name, agent_messages in event.items():
            for message in agent_messages['messages']:
                converted_message = convert_to_openai_messages(message)
                # Some messages have have an empty 'content' field, which will
                # throw and error when passing to ChatAgentMessage.
                if len(converted_message['content']) >= 1 and 'tool_calls' not in converted_message:             
                    # convert_to_openai_messages does not return a message id; however,
                    # ChatAgentMessage requires an id. Following MLflow's recommendation
                    # for creating this id.
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

        for event in supervisor.stream(request, stream_mode="updates"):
            for agent_name, agent_messages in event.items():
                for message in agent_messages['messages']:
                    converted_message = convert_to_openai_messages(message)
                    if len(converted_message['content']) >= 1 and 'tool_calls' not in converted_message:
                        openai_format_with_id = converted_message | {"id": str(uuid.uuid4())}
                        yield ChatAgentChunk(**{"delta": openai_format_with_id})


AGENT = SupervisorAgent()
mlflow.models.set_model(AGENT)