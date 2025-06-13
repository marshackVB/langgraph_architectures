from langgraph.graph import StateGraph, START, END
from langgraph.graph import END, MessagesState
from agents.rag.nodes import chatbot, route_tools, tool_node


graph_builder = StateGraph(MessagesState)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges("chatbot", route_tools,
    {"tools": "tools", 
      END: END}
  ) 
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("tools", "chatbot")
rag_agent = graph_builder.compile(name="rag_agent")