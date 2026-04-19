from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

def create_db(chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma.from_documents(chunks, embeddings)
    return db

def retreive(db, query, k=3):
    return db.similarity_search(query, k=k)
 