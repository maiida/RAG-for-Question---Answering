import os

import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer


def get_metric(md_folder):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B")
    md_files = [f for f in os.listdir(md_folder) if f.endswith(".md")]

    num_tokens = []

    for f in md_files:
        with open(os.path.join(md_folder, f), encoding="utf-8") as file:
            text = file.read()
        num_tokens.append(len(tokenizer.encode(text)))

    return num_tokens, md_files


def plot_token_counts(tokens, files):
    sns.set_style("whitegrid")
    sns.set_palette("Set2")

    plt.figure(figsize=(8, 5))
    plt.bar(files, tokens, color=sns.color_palette("Set2")[2])

    for i, v in enumerate(tokens):
        plt.text(i, v + max(tokens)*0.01, f"{int(v)}", ha='center', fontweight='bold', fontsize=10)

    plt.ylabel("Number of tokens per document", fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.ylim(0, max(tokens)*1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.title("Tokens per document", fontsize=14, fontweight='bold', pad=15)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    tokens, files = get_metric("./docs")
    plot_token_counts(tokens, files)
