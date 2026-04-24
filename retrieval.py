from langchain_chroma import Chroma
from langchain_vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever, EnsembleRetriever

def create_hybrid(chunks, query,k=3):
    #Создаем BM25 для поиска
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k=k
    #Создаем chroma
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
    vectorstore = Chroma.from_documents(chunks, embeddings)
    chroma_retriever = vectorstore.as_retriever(search_kwargs={"k" : k})

    #create a hybrid via EnsembleRetriever (BM25 + chroma)
    ensemble = EnsembleRetriever(retrievers=[bm25_retriever, chroma_retriever], weights=[0.4,0.6])
    # BM25=40%,chroma=60%
    result = ensemble.invoke(query)
    return result
