### The Supervisor architecture


###Problems:
 - When the supervisor calls the Genie agent; it might do more than just answer the genie-related question. For instance, if a question is asked related to a Genie table as well as Apach Spark, the Genie agent was accessed first. It answered the user's question based on the SQL results, but it also answered the Spark question. The Supervisor saw the answer and did not call the rag agent.
