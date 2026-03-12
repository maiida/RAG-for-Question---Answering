import shutil

import chromadb
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from config import Config


def build_vectorstore(
    config: Config,
    chunking_strategy: str = "recursive",
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    rebuild: bool = False
) -> tuple[chromadb.Collection, SentenceTransformer]:

    model_id = config.EMBEDDING_MODELS[config.embedding_model]
    print(f"  Loading embedding model: {model_id} (device={config.embedding_device})")
    embed_model = SentenceTransformer(model_id, device=config.embedding_device)

    collection_name = chunking_strategy

    if not rebuild and config.chroma_dir.exists():
        client = chromadb.PersistentClient(path=str(config.chroma_dir))
        col = client.get_or_create_collection(
            collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        if col.count() > 0:
            return col, embed_model
        print(f"  Collection '{collection_name}' is empty — rebuilding...")

    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3")
        ]
    )

    ids, texts, metadatas = [], [], []

    for path in sorted(config.docs_dir.glob("*.md")):

        text = path.read_text(encoding="utf-8")

        if chunking_strategy == "recursive":
            chunks = recursive_splitter.create_documents([text])

        elif chunking_strategy == "markdown":
            chunks = markdown_splitter.split_text(text)

        else:
            raise ValueError(f"Unknown chunking strategy: {chunking_strategy}")

        for i, chunk in enumerate(chunks):
            content = chunk.page_content
            ids.append(f"{path.name}::{i}")
            texts.append(content)
            metadatas.append({
                "source": path.name,
                "chunk_id": f"{path.name}::{chunking_strategy}::{i}"
            })

        print(f"  Loaded: {path.name}  ({len(chunks)} chunks)")

    embeddings = embed_model.encode(texts, normalize_embeddings=True).tolist()

    if rebuild:
        shutil.rmtree(config.chroma_dir, ignore_errors=True)

    client = chromadb.PersistentClient(path=str(config.chroma_dir))
    collection = client.get_or_create_collection(
        collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)

    print(f"\nVector store ready ({len(ids)} chunks)")

    return collection, embed_model
