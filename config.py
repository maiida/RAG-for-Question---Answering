from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:

    docs_dir: Path = Path("./docs")
    chroma_dir: Path = Path("./chroma_db")

    EMBEDDING_MODELS = {
        "bge-small": "BAAI/bge-small-en-v1.5",
        "e5-small":  "intfloat/e5-small-v2",
    }
    embedding_model: str = "bge-small"
    embedding_device: str = "cpu"

    fetch_k: int = 5
    fetch_pool: int = 15
    token_budget: int = 700
