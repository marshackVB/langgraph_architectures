import mlflow
from databricks_langchain import ChatDatabricks

config = mlflow.models.ModelConfig(development_config="config.yaml")
rag_config = config.get("agents").get("genie")[0]

model = ChatDatabricks(endpoint=rag_config.get('llm'), extra_params=rag_config.get('llm_parameters'))