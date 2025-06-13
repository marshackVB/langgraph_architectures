import os
import mlflow
from databricks_langchain.genie import GenieAgent
from databricks.sdk import WorkspaceClient

config = mlflow.models.ModelConfig(development_config="config.yaml")
genie_config = config.get("agents").get("genie")[0]


genie_agent = GenieAgent(
    genie_space_id = genie_config['space_id'],
    genie_agent_name = genie_config['name'],
    description= genie_config['description'],
    client=WorkspaceClient(
        host=os.getenv("DATABRICKS_HOST"),
        token=os.getenv("DATABRICKS_TOKEN"),
    )
  )
