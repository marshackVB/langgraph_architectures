# Graph used as Genie as agent

from langgraph.graph import START, END, StateGraph
from mlflow.langchain.chat_agent_langgraph import ChatAgentState

from agents.genie.nodes import call_genie


graph_builder = StateGraph(ChatAgentState)
graph_builder.add_node("Genie", call_genie)
graph_builder.add_edge(START, "Genie")
graph_builder.add_edge("Genie", END)
app = graph_builder.compile()