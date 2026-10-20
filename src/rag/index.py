from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.ingest.manifest import write_manifest_row

LIBRARY_DIR = Path(__file__).resolve().parent / "library"

DOCUMENT_METADATA = {
    "who_2021_aqg.md": {
        "title": "WHO Global Air Quality Guidelines 2021",
        "publisher": "World Health Organization",
        "url_or_doi": "https://www.who.int/publications/i/item/9789240034228",
        "licence": "CC BY-NC-SA 3.0 IGO",
    },
    "punjab_epd_standards.md": {
        "title": "Punjab Environmental Quality Standards (PEQS)",
        "publisher": "Punjab Environmental Protection Department",
        "url_or_doi": "https://epd.punjab.gov.pk/peqs",
        "licence": "Official Public Document / Government of Punjab",
    },
    "us_epa_aqi_guide.md": {
        "title": "US EPA Air Quality Index Technical Assistance Document",
        "publisher": "US Environmental Protection Agency",
        "url_or_doi": "https://www.airnow.gov/aqi/aqi-basics",
        "licence": "US Public Domain",
    },
    "pmd_ndma_health_advisory.md": {
        "title": "NDMA & PMD Smog Emergency Advisories",
        "publisher": "National Disaster Management Authority (Pakistan)",
        "url_or_doi": "https://cms.ndma.gov.pk",
        "licence": "Official Public Document / Government of Pakistan",
    },
}

class HybridRAGIndex:
    def __init__(self, library_dir: Path = LIBRARY_DIR):
        self.library_dir = library_dir
        self.chunks: List[Dict[str, Any]] = []
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.faiss_index: Optional[faiss.IndexFlatIP] = None
        self.tfidf_matrix = None
        self._load_and_index()

    def _load_and_index(self) -> None:
        """Parses library markdown documents into docid#section chunks and registers them."""
        now_utc = datetime.now(timezone.utc).isoformat()
        manifest_path = Path("data/manifest.csv")

        self.chunks = []
        doc_files = list(self.library_dir.glob("*.md"))

        for doc_file in doc_files:
            meta = DOCUMENT_METADATA.get(doc_file.name, {
                "title": doc_file.stem,
                "publisher": "Official Agency",
                "url_or_doi": f"local://{doc_file.name}",
                "licence": "Open Government / Public Information",
            })

            # Register document in manifest.csv
            try:
                write_manifest_row(
                    file_path=doc_file,
                    source="RAG Document Library",
                    publisher=meta["publisher"],
                    url_or_doi=meta["url_or_doi"],
                    query=f"document={doc_file.name}",
                    fetch_time_utc=now_utc,
                    licence=meta["licence"],
                    manifest_path=manifest_path,
                )
            except Exception as e:
                print(f"[RAG Index Warning] Manifest registration note: {e}")

            content = doc_file.read_text(encoding="utf-8")
            # Parse sections: ## docid#section_name
            sections = re.split(r"\n##\s+", content)
            
            for sec in sections:
                sec = sec.strip()
                if not sec or sec.startswith("#"):
                    continue
                lines = sec.split("\n", 1)
                sec_id = lines[0].strip()
                body = lines[1].strip() if len(lines) > 1 else ""
                
                self.chunks.append({
                    "chunk_id": sec_id,
                    "document": doc_file.name,
                    "title": meta["title"],
                    "text": body,
                    "full_chunk": f"{sec_id}: {body}",
                })

        if not self.chunks:
            return

        texts = [c["full_chunk"] for c in self.chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts).toarray().astype("float32")

        # Normalize L2 for Cosine Similarity in FAISS
        norms = np.linalg.norm(self.tfidf_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized_embeds = self.tfidf_matrix / norms

        dim = normalized_embeds.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(normalized_embeds)

    def search(self, query: str, top_k: int = 3, score_threshold: float = 0.15) -> List[Dict[str, Any]]:
        """
        Executes hybrid search across indexed chunks.
        Returns matched chunks with similarity scores.
        """
        if self.faiss_index is None or not self.chunks:
            return []

        q_vec = self.vectorizer.transform([query]).toarray().astype("float32")
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        scores, indices = self.faiss_index.search(q_vec, min(top_k, len(self.chunks)))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or score < score_threshold:
                continue
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(score)
            results.append(chunk)

        return results

if __name__ == "__main__":
    idx = HybridRAGIndex()
    res = idx.search("What is the WHO 24 hour PM2.5 guideline?")
    for r in res:
        print(f"[{r['chunk_id']}] (Score: {r['score']:.3f})\n{r['text']}\n")
