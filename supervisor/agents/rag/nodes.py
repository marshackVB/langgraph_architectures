from langgraph.graph import END
from mlflow.langchain.chat_agent_langgraph import ChatAgentState, ChatAgentToolNode 

from agents.rag.resources.model import model
from agents.rag.tools import documentation_search_tool

tools = [documentation_search_tool]
model_with_tools = model.bind_tools(tools, tool_choice="auto")
tool_node = ChatAgentToolNode(tools=tools)

system_message = """You are a trusted assistant capable of searching Databricks and Apache Spark documentation. Based on the users most recent question and relevent conversation history, call the documentation search tools you have access to. Use the retrieved documentation to answer the user's questions. Make sure your final answer is grounded on the documents returned by the tool."""


def chatbot(state: ChatAgentState):
  """
  Retrieve relevent documents based on the user's question and conversation
  history. Answer the user's quesiton based on the documentation.
  """
  messages = [{"role": "system", "content": system_message}] + state["messages"]
  response = model_with_tools.invoke(messages)
  return {"messages": [response]}


def route_tools(state: ChatAgentState):
  """
  A router that determines if tools should be called or a final
  answer returned to the user
  """
  ai_message = state["messages"][-1]
  
  if "tool_calls" in ai_message and len(ai_message['tool_calls']) > 0:
    return "tools"
  
  return END





