from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

PROMPT = ChatPromptTemplate.from_template(
    """
You are a helpful assistant for students.
Answer the question using ONLY the context below.
If context does not contain the answer, say: "I couldn't find that in the lecture notes."
Context:
{context}
Question:
{query}
"""
)

def generate(results, query: str) -> str:
    #divide context
    context = "\n\n".join(doc.page_content for doc in results)

    llm = ChatOpenAI(model="gpt-4o-mini")
    query_llm = PROMPT.format_messages(context=context, query=query)
    response = llm.invoke(query_llm)
    return response.content
