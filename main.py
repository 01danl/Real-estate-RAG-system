from ingestion import loader
from retrieval import create_db, retreive
from generation import generate

def main(): 
    chunks = loader("Daniyal Bissenov BDA2402 CV.pdf")
    db = create_db(chunks)
    query = "What is the main topic of the document?"
    results = retreive(db,query)
    answer_from_llm = generate(results, query)
    print(answer_from_llm)
if __name__ == "__main__":
    main()