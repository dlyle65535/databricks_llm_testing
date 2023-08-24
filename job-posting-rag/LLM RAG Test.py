# Databricks notebook source
dbutils.widgets.dropdown("reset_vector_database", "true", ["false", "true"], "Recompute embeddings for chromadb")

# COMMAND ----------

# MAGIC %pip install -U chromadb==0.3.22 langchain==0.0.251 bitsandbytes==0.40.1

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from mlflow.gateway import set_gateway_uri, create_route, delete_route
set_gateway_uri("databricks")

# COMMAND ----------

# Create a Route for completions with OpenAI GPT-3
delete_route( name="gpt-3.5-completions")
create_route(
    name="gpt-3.5-completions",
   route_type="llm/v1/completions",
   model={
       "name": "gpt-3.5-turbo",
       "provider": "openai",
       "openai_config": {
           "openai_api_key": "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"
       }
   }
)

# COMMAND ----------

from langchain.embeddings import HuggingFaceEmbeddings
 
# Download model from Hugging face
hf_embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
# Prepare a directory to store the document database. Any path on `/dbfs` will do.
job_descriptions_vector_db_path = "FileStore/david.lyle@databricks.com/job_descriptions_rag/data/vector_db"
 
# Don't recompute the embeddings if the're already available
compute_embeddings = dbutils.widgets.get("reset_vector_database") == "true"
 
if compute_embeddings:
  print(f"creating folder {job_descriptions_vector_db_path} under our blob storage (dbfs)")
  dbutils.fs.rm(job_descriptions_vector_db_path, True)
  dbutils.fs.mkdirs(job_descriptions_vector_db_path)

print(hf_embed)

# COMMAND ----------

from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
file_path = "dbfs:/FileStore/david.lyle@databricks.com/jobs_data/data_job_posts.csv"

df=(spark.read.format("csv")
    .option("header",True).option("multiline",True)
    .option("quote", "\"")
    .option("escape", "\"")
    .load(file_path)
  )

train_df, test_df, validation_df = df.randomSplit([0.6, 0.2, 0.2])
all_texts = train_df
 
if compute_embeddings: 
  print(f"Saving document embeddings under /dbfs{job_descriptions_vector_db_path}")
  # Transform our rows as langchain Documents
  # If you want to index shorter term, use the text_short field instead
  documents = [Document(page_content=r["JobDescription"], 
                        metadata={"source": "https://www.kaggle.com/datasets/madhab/jobposts"},
                        company=r["Company"]) for r in all_texts.collect()]
 
  # If your texts are long, you may need to split them. However it's best to summarize them instead as show above.
  # text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100)
  # documents = text_splitter.split_documents(documents)
 
  # Init the chroma db with the sentence-transformers/all-mpnet-base-v2 model loaded from hugging face  (hf_embed)
  db = Chroma.from_documents(collection_name="job_description_docs", documents=documents, embedding=hf_embed, persist_directory="/dbfs"+job_descriptions_vector_db_path)
  db.similarity_search("dummy") # tickle it to persist metadata (?)
  db.persist()

# COMMAND ----------

from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI, MlflowAIGateway, Databricks

# Create a prompt structure for Llama2 Chat (note that if using MPT the prompt structure would differ)
template = """[INST] <<SYS>>
You are an AI assistant, helping people write Job Postings. 
Use only information provided in the following paragraphs to answer the question at the end. 
Explain your answer with reference to these paragraphs.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. 
If you don't know the answer to a question, please don't share false information.

<</SYS>>

{context}
 
{question} [/INST]
"""
# Retrieve the AI Gateway Route
openai_completion_route = MlflowAIGateway(
  gateway_uri="databricks",
  route="gpt-3.5-completions"
)

prompt = PromptTemplate(input_variables=['context', 'question'], template=template)

# Wrap the prompt and Gateway Route into a chain
retrieval_qa_chain = RetrievalQA.from_chain_type(llm= Databricks(cluster_driver_port="7777"),\
                                                      chain_type="stuff", retriever=db.as_retriever(),\
                                                      chain_type_kwargs={"prompt": prompt})


# COMMAND ----------

query = "Please write a Job Description for a Data Scientist using Databricks"
job_posting = retrieval_qa_chain.run(query)
print(job_posting)

# COMMAND ----------

from langchain.llms import Databricks
llm = Databricks(cluster_driver_port="7777")

llm("How are you?")

# COMMAND ----------


