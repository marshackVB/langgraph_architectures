from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from agents.rag.resources.retriever import retriever

def search_documentation(search: str):

  search_results = retriever.invoke(search)

  return search_results


class SeachSchema(BaseModel):
    search: str = Field(description = "Text to search. A similarity search will retrieve documentation most similar to the search terms.")


documentation_search_tool = StructuredTool.from_function(search_documentation, 
                                               name = "search_documentation",
                                               description = "Search the Spark documentation to retrieve relevent information",
                                               args_schema = SeachSchema,
                                               response_format = "content",
                                               return_direct=True, 
                                               verbose = False)