from typing import Optional, Any, Generator
import mlflow
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext)


from agents.supervisor.graph import app as supervisor


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
    for event in self.agent.stream(request, stream_mode="updates"):
        for node_data in event.values():
            # Supervisor router messages return None
            if node_data is not None:
                messages.extend(
                    ChatAgentMessage(**msg) for msg in node_data.get("messages", [])
                )
    return ChatAgentResponse(messages=messages)
  

  def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:

        request = {"messages": self._convert_messages_to_dict(messages)}

        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                # Supervisor router messages return None
                if node_data is not None:
                    yield from (
                        ChatAgentChunk(**{"delta": msg}) for msg in node_data.get("messages", [])
                    )

AGENT = SupervisorAgent()
mlflow.models.set_model(AGENT)