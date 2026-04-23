from helper import load_config
from rag import RAGPipeline
from schema import QueryRequest

def main():
    config = load_config()

    pipeline = RAGPipeline(config)
    pipeline.ingest()

    while True:
        q = input("Enter question: ")
        req = QueryRequest(question=q)

        result = pipeline.query(req.question)

        print("\nANSWER:\n", result["answer"])
        print("\nCONFIDENCE:", result["confidence"])
        print("\nSOURCES:")
        for s in result["sources"]:
            print(s)

if __name__ == "__main__":
    main()