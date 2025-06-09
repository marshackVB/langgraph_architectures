from langgraph.graph import StateGraph, START, END
from mlflow.langchain.chat_agent_langgraph import ChatAgentState
from agents.genie.nodes import chatbot, route_tools, tool_node

graph_builder = StateGraph(ChatAgentState)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges("chatbot", route_tools,
    {"tools": "tools", 
      END: END}
  ) 
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("tools", "chatbot")
app = graph_builder.compile()