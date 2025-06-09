import mlflow
from databricks_langchain import DatabricksVectorSearch
from databricks_langchain import DatabricksEmbeddings


config = mlflow.models.ModelConfig(development_config="config.yaml")
rag_config = config.get("agents").get("rag")[0]
mlflow_config = config.get("mlflow")


embedding_model = DatabricksEmbeddings(endpoint=rag_config.get("embedding_model"))
vector_search_schema = rag_config.get("retrieval_schema")

retriever = DatabricksVectorSearch(
    endpoint = rag_config.get("endpoint_name"),
    index_name = rag_config.get("index_location"),
    text_column = vector_search_schema.get("chunk_text"),
    embedding=embedding_model,
    columns =[
      vector_search_schema.get("primary_key"),
      vector_search_schema.get("chunk_text"),
      vector_search_schema.get("document_uri")
    ]).as_retriever(search_kwargs=rag_config.get("retrieval_parameters"))


def format_documents(docs):
    chunk_template = config.get("chunk_template")
    chunk_contents = [
        chunk_template.format(
            chunk_text=d.page_content,
            document_uri=d.metadata[vector_search_schema.get("document_uri")],
        )
        for d in docs
    ]
    return "".join(chunk_contents)