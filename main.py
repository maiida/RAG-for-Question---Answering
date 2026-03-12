import argparse
from engine import RAGEngine


def main():
    parser = argparse.ArgumentParser(description="RAG for QA")
    parser.add_argument("question", type=str, help="Question to ask")
    parser.add_argument(
        "--mode",
        choices=["naive", "judge"],
        default="naive",
        help="naive: basic RAG | judge: detects ambiguity and asks for clarification"
    )
    parser.add_argument(
        "--chunking",
        choices=["recursive", "markdown"],
        default="recursive",
        help="Chunking strategy for the vector store"
    )
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the vector store from scratch")
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B", help="HuggingFace model ID for generation")
    args = parser.parse_args()

    engine = RAGEngine(chunking_strategy=args.chunking, rebuild=args.rebuild, model=args.model)

    if args.mode == "naive":
        print(engine.generate_answer_naive(args.question))

    elif args.mode == "judge":
        print(engine.generate_answer_judge(args.question))



if __name__ == "__main__":
    main()
