from __future__ import annotations

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from torch.nn import Identity


def create_cross_encoder(model_name: str) -> CrossEncoder:
    # Keep the model's logits as the score so calibration and serving use the same scale.
    return CrossEncoder(model_name, device="cpu", activation_fn=Identity())


class CrossEncoderReranker:
    def __init__(self, model_name: str, batch_size: int) -> None:
        self.model = create_cross_encoder(model_name)
        self.batch_size = batch_size

    def score(self, query: str, documents: list[Document]) -> list[tuple[Document, float]]:
        if not documents:
            return []
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return sorted(
            [(doc, float(score)) for doc, score in zip(documents, scores, strict=True)],
            key=lambda pair: pair[1], reverse=True,
        )
