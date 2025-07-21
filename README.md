### LangGraph reference architectures deployed on Databricks Model Serving

This library provides references implementations of multi-agent LangGraph systems that incorporate MLflow logging and can be directly deployed to Databriks Model Serving. Two architecture references are available: [multi-agent supervisor](https://langchain-ai.github.io/langgraph/agents/multi-agent/#supervisor) and [multi-agent swarm](https://langchain-ai.github.io/langgraph/agents/multi-agent/#swarm).  
<br>
<br>
![langgraph architectures](img/supervisor_swarm_architectures.png)

#### Prerequisites
 - A Databricks accounts with access to Model Serving
 - A Databricks Vector Index (for the RAG agent)
   - If you don't have an existing Vector Index, you can easily create one by executing one of the demos available [here](https://www.databricks.com/resources/demos/tutorials/data-science-and-ai/lakehouse-ai-deploy-your-llm-chatbot?itm_data=demo_center). 
 - A Databricks Genie Space (for the Genie agent)
   - If you don't have an existing Genie Space, you can easily create one by installing one of the AI/BI Genie demos available [here](https://www.databricks.com/resources/demos/tutorials?itm_data=demo_center#data-warehouse-and-bi).

#### Project layout
Clone this repository as a git folder in your Databricks Workspace.
 - The **config.yaml** contains information that govern our agent deployment, including the below. Update the configuration to reflect your own Databricks Workspace objects. 
     - The Foundation Model endpoints that power the agents
     - The MLflow Experiment where our agent will be logged
     - The Unity Catalog location where our agens will be registred before serving deployment
     - Location of the Vector Index
     - The Genie Space ID
 - The RAG agent and the Genie Agent are defined in the **agents** directory.
 - The **supervisor_agent.py** file builds a supervisor workflow using the RAG and Genie agents as subagents.
 - To log the superisor agent to an MLflow Experiment and deploy it to a Model Serving endpoint, run the **supervisor_deploy_endpoint** notebook.
 - The **swarm_agent.py** file builds a swarm workflow between the RAG and Genie agents.
 - To log the superisor agent to an MLflow Experiment and deploy it as a Model Serving endpoint, run the **swarm_deploy_endpoint** notebook.

#### Considerations
 - Agentic systems performs best with highly sophisticated LLM. This project was tested with Claude Sonnet [4.0](https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/supported-models#-anthropic-claude-sonnet-4) and [3.7](https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/supported-models#-anthropic-claude-37-sonnet) available in Databricks.
 - LangGraph has higher-level APIs to create different agentic architectures. For instance, [create_react_agent](https://langchain-ai.github.io/langgraph/reference/agents/) can be used to configuring an LLM and tool calling loop. This project did not use this high level API for creating subagents, instead implemented the tool calling loop functionality directily using a router function, which determines if tools need to be called or if an agent has provided it final response.
 - This project did use the [higher-level API](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/#math-agent) for creating the supervisor agent though this could have been implemented using the [base componentry](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/#2-create-supervisor-with-langgraph-supervisor). Similarly, the swarm implementation can be [further customized](https://github.com/langchain-ai/langgraph-swarm-py?tab=readme-ov-file#how-to-customize), though this project used the higher-level [create-swarm](https://langchain-ai.github.io/langgraph/reference/swarm/#langgraph_swarm.swarm.create_swarm) option.

#### Helpful documentation and tutorials:
 - Understanding multi-agent handoffs in LangGraph ([link](https://www.youtube.com/watch?v=WTr6mHTw5cM))
 - Multi-agent swarms in LangGraph ([link](https://www.youtube.com/watch?v=JeyDrn1dSUQ))
 - The [langgraph-supervisor](https://github.com/langchain-ai/langgraph-supervisor-py) library
 - The [langgraph-swarm](https://github.com/langchain-ai/langgraph-swarm-py) library



