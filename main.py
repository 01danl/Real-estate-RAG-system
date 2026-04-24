from ingestion import loader
from retrieval import create_hybrid
from generation import generate

def main(): 
    chunks = loader("Daniyal Bissenov BDA2402 CV.pdf")
    query = "What is the main topic of the document?"
    results = create_hybrid(chunks, query, k=3) 
    answer_from_llm = generate(results, query)
    print(answer_from_llm)
if __name__ == "__main__":
    main()