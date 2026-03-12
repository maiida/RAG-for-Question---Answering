from transformers import pipeline, AutoTokenizer

from config import Config
from ingest import build_vectorstore


class RAGEngine:

    def __init__(self, chunking_strategy: str = "recursive", rebuild: bool = False, model: str = "Qwen/Qwen3.5-2B"):
        self.config = Config()
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.pipe = pipeline("image-text-to-text", model=model)
        self.col, self.embed_model = build_vectorstore(
            self.config, chunking_strategy=chunking_strategy, rebuild=rebuild
        )
    def retrieve_chunks(self, query: str) -> list[dict]:
        query_embedding = self.embed_model.encode(query, normalize_embeddings=True).tolist()

        results = self.col.query(
            query_embeddings=[query_embedding],
            n_results=min(self.config.fetch_pool, self.col.count()),
        )

        docs      = results["documents"][0]
        metas     = results["metadatas"][0]
        distances = results["distances"][0]

        selected, used = [], 0

        for doc, meta, dist in zip(docs, metas, distances):
            if len(selected) >= self.config.fetch_k:
                break
            n = len(self.tokenizer.encode(doc, add_special_tokens=False))
            if used + n <= self.config.token_budget:
                selected.append({
                    "text":   doc,
                    "source": meta.get("source", "unknown"),
                    "score":  1.0 - dist,
                    "tokens": n,
                })
                used += n

        print(f"  [retriever] {len(selected)}/{len(docs)} chunks  ({used}/{self.config.token_budget} tokens)")
        return selected

    def format_chunks(self, chunks: list[dict]) -> str:
        return "\n".join(
            f"- {c['text']} \n (source: {c['source']}) \n "
            for c in chunks
        )

    def generate_answer_naive(self, question: str) -> str:
        context = self.format_chunks(self.retrieve_chunks(question))
        messages = [{"role": "user", "content": [{"type": "text", "text": f"""
                Use ONLY the context below to answer the question.
                Be concise. Do not invent information.
                CONTEXT: {context} QUESTION: {question} ANSWER:"""}]}]
        return self.pipe(text=messages)[0]['generated_text'][-1]["content"]


    def detect_ambiguity(self, question: str, context: str) -> dict:
        messages = [{"role": "user", "content": [{"type": "text", "text": f"""Analyze the context BELOW and see if it needs more information
    to answer the question confidently.

    CONTEXT:
    {context}

    QUESTION: {question}

    Answer in this exact format (no extra text):
    AMBIGUOUS: yes/no
    REASON: <one sentence explaining why or why not>
    CLARIFYING_QUESTION: <ask the user to specify which of the conflicting options applies to their situation, listing the options explicitly.
    Else write None>"""}]}]

        raw = self.pipe(text=messages)[0]['generated_text'][-1]["content"].strip()

        result = {"ambiguous": False, "reason": "", "clarifying_question": ""}
        for line in raw.splitlines():
            if line.startswith("AMBIGUOUS:"):
                result["ambiguous"] = "yes" in line.lower()
            elif line.startswith("REASON:"):
                result["reason"] = line.split(":", 1)[1].strip()
            elif line.startswith("CLARIFYING_QUESTION:"):
                val = line.split(":", 1)[1].strip()
                result["clarifying_question"] = "" if val.upper() == "NONE" else val
        return result

    def generate_answer_judge(self, question: str, verbose: bool = True) -> str:
        retrieved_chunks = self.retrieve_chunks(question)
        context = self.format_chunks(retrieved_chunks)

        judgment = self.detect_ambiguity(question, context)
        if verbose:
            print(f"  [judge] ambiguous={judgment['ambiguous']}  reason={judgment['reason']}")

        if judgment["ambiguous"] and judgment["clarifying_question"]:
            print(f"\nI found conflicting or incomplete information in the docs.\n")
            print(f"Specifically: {judgment['reason']}\n")
            print(f"Before I answer: {judgment['clarifying_question']}\n")
            clarification = input("Your clarification: ").strip()
            messages = [{"role": "user", "content": [{"type": "text", "text": f"""
    Use the user clarification {clarification} to extract relevant information from context and answer the question.
    Do not add context not related to the clarification.

    CONTEXT: {context}
    QUESTION: {question}
    ANSWER:"""}]}]
        else:
            messages = [{"role": "user", "content": [{"type": "text", "text": f"""Use ONLY the context below to answer the question.
    Be concise. Do not invent information.

    CONTEXT: {context}
    QUESTION: {question}
    ANSWER:"""}]}]

        return self.pipe(text=messages)[0]['generated_text'][-1]["content"]

