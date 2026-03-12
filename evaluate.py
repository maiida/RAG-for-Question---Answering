import argparse
import json

import matplotlib.pyplot as plt
import seaborn as sns

from engine import RAGEngine


def evaluate_doc_level(engine, eval_json):
    with open(eval_json, "r") as f:
        data = json.load(f)

    recalls = []
    precisions = []
    question_ids = []

    for q in data["questions"]:
        question_ids.append(q["id"])
        expected = set(q["expected_context"])

        retrieved_chunks = engine.retrieve_chunks(q["question"])
        retrieved_docs = set(c["source"] for c in retrieved_chunks)

        correct = expected & retrieved_docs

        recall    = len(correct) / len(expected)
        precision = len(correct) / len(retrieved_docs) if retrieved_docs else 0

        recalls.append(recall)
        precisions.append(precision)

    return recalls, precisions, question_ids


def plot_metrics(recs, precs, keys):
    sns.set_style("whitegrid")
    sns.set_palette("Set2")

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    axes[0].bar(keys, recs, color=sns.color_palette("Set2")[0])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Recall")
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(recs):
        axes[0].text(i, v + 0.02, f"{v*100:.0f}%", ha='center', fontweight='bold', fontsize=10)
    axes[0].set_xlabel("")
    axes[0].text(0.5, -0.15, "Recall per Document", ha='center', va='center', transform=axes[0].transAxes,
                 fontsize=12, fontweight='bold')

    axes[1].bar(keys, precs, color=sns.color_palette("Set2")[1])
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Precision")
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(precs):
        axes[1].text(i, v + 0.02, f"{v*100:.0f}%", ha='center', fontweight='bold', fontsize=10)
    axes[1].set_xlabel("")
    axes[1].set_xticks(range(len(keys)))
    axes[1].set_xticklabels(keys, rotation=45, ha='right', fontsize=10)
    axes[1].text(0.5, -0.25, "Precision per Document", ha='center', va='center', transform=axes[1].transAxes,
                 fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval")
    parser.add_argument(
        "--chunking",
        choices=["recursive", "markdown"],
        default="recursive",
        help="Chunking strategy to evaluate"
    )
    args = parser.parse_args()

    engine = RAGEngine(chunking_strategy=args.chunking)
    recalls, precisions, question_ids = evaluate_doc_level(engine, "evaluation/evaluation.json")
    plot_metrics(recalls, precisions, question_ids)
