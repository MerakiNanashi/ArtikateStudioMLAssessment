from helper import load_config
from rag import RAGPipeline


def precision_at_k(retrieved, ground_truth, k=3):
    retrieved_docs = [r["document"] for r in retrieved[:k]]
    return int(ground_truth in retrieved_docs)


def run_eval(pipeline):
    test_set = [
        {
            "q": "What is the financing term in Contract_11?",
            "doc": "Contract_11.pdf"
        },
        {
            "q": "What is the financing term duration in Contract_1?",
            "doc": "Contract_1.pdf"
        },
        {
            "q": "What interest rate is applied to monthly payments in Contract_10?",
            "doc": "Contract_10.pdf"
        },
        {
            "q": "What penalty is charged for late payments in Contract_1?",
            "doc": "Contract_1.pdf"
        },
        {
            "q": "What happens if the borrower defaults in Contract_11?",
            "doc": "Contract_11.pdf"
        },
        {
            "q": "Under what condition can the agreement be terminated in Contract_10?",
            "doc": "Contract_10.pdf"
        },
        {
            "q": "What type of guarantee is placed on the financed asset in Contract_1?",
            "doc": "Contract_1.pdf"
        },
        {
            "q": "Is insurance required in Contract_11 and what is specified?",
            "doc": "Contract_11.pdf"
        },
        {
            "q": "What is the total financed amount in Contract_10?",
            "doc": "Contract_10.pdf"
        },
        {
            "q": "Who is the borrower in Contract_1?",
            "doc": "Contract_1.pdf"
        },
    ]

    scores = []

    for t in test_set:
        retrieved = pipeline.retrieve(t["q"])
        score = precision_at_k(retrieved, t["doc"])
        scores.append(score)

        print(f"\nQ: {t['q']}")
        print(f"Expected Doc: {t['doc']}")
        print("Top-3 Retrieved:", [r["document"] for r in retrieved[:3]])
        print("Score:", score)

    print("\nFinal Precision@3:", sum(scores) / len(scores))


def main():
    config = load_config()

    pipeline = RAGPipeline(config)
    pipeline.ingest()

    run_eval(pipeline)


if __name__ == "__main__":
    main()