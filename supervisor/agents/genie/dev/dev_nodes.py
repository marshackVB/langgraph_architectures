# This was based on the Agent approach where Genie was called as an 
# Agent rather than a tool.

import mlflow
from mlflow.langchain.chat_agent_langgraph import ChatAgentState
from agents.genie.resources.genie_room import genie_agent

config = mlflow.models.ModelConfig(development_config="config.yaml")
genie_config = config.get("agents").get("genie")[0]


def call_genie(state: ChatAgentState):
  result = genie_agent.invoke(state)
  return {
        "messages": [
            {
                "role": "assistant",
                "content": result["messages"][-1].content,
                "name": genie_config['name'],
            }
        ]
    }