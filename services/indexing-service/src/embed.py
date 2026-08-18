import os
from fastembed import TextEmbedding

MODEL_NAME = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
_model = None


def get_model() -> TextEmbedding:
    global _model
    if _model is None:
        _model = TextEmbedding(model_name=MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_model()
    embeddings = list(model.embed(texts))
    return [e.tolist() for e in embeddings]


def embed_query(query: str) -> list[float]:
    return embed_texts([query])[0]
