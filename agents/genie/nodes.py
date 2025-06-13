from langgraph.graph import END, MessagesState
from langgraph.prebuilt import ToolNode
from agents.genie.resources.model import model
from agents.genie.tools import text_to_sql_tool

tools = [text_to_sql_tool]
model_with_tools = model.bind_tools(tools, tool_choice="auto")
tool_node = ToolNode(tools=tools)

system_message = """You are a trusted assistant capable of interacting with supply chain performance data using a text to sql tool. Based on the user's most recent question and relevent conversation history, create a question to pass to the text to sql tool. Analyze the data returned to answer the user's question. Make sure your final answer is grounded in the data. Do not answer questions related to other topics. If a user asks a question that is partially related to another topic; only answer the portion of the question related to the supply chain data"""


def chatbot(state: MessagesState):
  """
  Retrieve relevent documents based on the user's question and conversation
  history. Answer the user's quesiton based on the documentation.
  """
  messages = [{"role": "system", "content": system_message}] + state["messages"]
  response = model_with_tools.invoke(messages)
  return {"messages": [response]}


def route_tools(state: MessagesState):
  """
  A router that determines if tools should be called or a final
  answer returned to the user
  """
  ai_message = state["messages"][-1]
  
  if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
  
  return END