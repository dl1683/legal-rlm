"""Document clustering - group related documents by content similarity.

Uses TF-IDF and simple clustering without heavy ML dependencies.
"""

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from collections import Counter
import math
import re


@dataclass
class DocumentCluster:
    """A cluster of related documents."""
    id: int
    name: str  # Auto-generated or user-provided name
    documents: list[str] = field(default_factory=list)  # File paths
    keywords: list[str] = field(default_factory=list)  # Top terms
    centroid: Optional[dict] = None  # TF-IDF centroid vector

    def add_document(self, doc_path: str):
        """Add a document to the cluster."""
        if doc_path not in self.documents:
            self.documents.append(doc_path)

    @property
    def size(self) -> int:
        return len(self.documents)


class DocumentClusterer:
    """Simple document clustering based on TF-IDF similarity."""

    def __init__(self):
        self._document_vectors: dict[str, dict[str, float]] = {}
        self._idf_scores: dict[str, float] = {}
        self._vocab: set[str] = set()

    def fit(self, documents: dict[str, str]):
        """
        Fit the clustering model on documents.

        Args:
            documents: Dict mapping file path to document text
        """
        # Build vocabulary and compute TF-IDF
        self._build_vocabulary(documents)
        self._compute_idf(documents)
        self._compute_tfidf(documents)

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization."""
        # Lowercase and extract words
        text = text.lower()
        words = re.findall(r'\b[a-z]{3,}\b', text)

        # Remove common stop words
        stop_words = {
            "the", "and", "for", "that", "this", "with", "are", "was", "were",
            "been", "have", "has", "had", "will", "would", "could", "should",
            "from", "they", "their", "there", "these", "those", "which", "what",
            "when", "where", "who", "whom", "whose", "why", "how", "all", "any",
            "both", "each", "few", "more", "most", "other", "some", "such", "not",
            "only", "own", "same", "than", "too", "very", "can", "just", "shall",
        }

        return [w for w in words if w not in stop_words]

    def _build_vocabulary(self, documents: dict[str, str]):
        """Build vocabulary from all documents."""
        self._vocab = set()
        for text in documents.values():
            tokens = self._tokenize(text)
            self._vocab.update(tokens)

    def _compute_idf(self, documents: dict[str, str]):
        """Compute inverse document frequency for each term."""
        num_docs = len(documents)
        doc_counts: Counter = Counter()

        for text in documents.values():
            tokens = set(self._tokenize(text))
            doc_counts.update(tokens)

        self._idf_scores = {}
        for term, count in doc_counts.items():
            self._idf_scores[term] = math.log(num_docs / (1 + count))

    def _compute_tfidf(self, documents: dict[str, str]):
        """Compute TF-IDF vectors for all documents."""
        self._document_vectors = {}

        for doc_path, text in documents.items():
            tokens = self._tokenize(text)
            tf = Counter(tokens)

            # Normalize TF
            max_tf = max(tf.values()) if tf else 1

            vector = {}
            for term, count in tf.items():
                tf_norm = count / max_tf
                idf = self._idf_scores.get(term, 0)
                vector[term] = tf_norm * idf

            self._document_vectors[doc_path] = vector

    def _cosine_similarity(self, vec1: dict[str, float], vec2: dict[str, float]) -> float:
        """Compute cosine similarity between two vectors."""
        common_terms = set(vec1.keys()) & set(vec2.keys())
        if not common_terms:
            return 0.0

        dot_product = sum(vec1[t] * vec2[t] for t in common_terms)
        norm1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
        norm2 = math.sqrt(sum(v ** 2 for v in vec2.values()))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def cluster(self, num_clusters: int = 5, min_similarity: float = 0.2) -> list[DocumentCluster]:
        """
        Cluster documents using simple agglomerative approach.

        Args:
            num_clusters: Target number of clusters
            min_similarity: Minimum similarity to merge clusters

        Returns:
            List of DocumentCluster objects
        """
        if not self._document_vectors:
            return []

        # Initialize each document as its own cluster
        clusters: list[DocumentCluster] = []
        for i, doc_path in enumerate(self._document_vectors.keys()):
            cluster = DocumentCluster(
                id=i,
                name=f"Cluster {i}",
                documents=[doc_path],
            )
            cluster.centroid = self._document_vectors[doc_path].copy()
            clusters.append(cluster)

        # Agglomerative clustering
        while len(clusters) > num_clusters:
            # Find most similar pair
            best_sim = -1
            best_pair = (0, 1)

            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    sim = self._cosine_similarity(
                        clusters[i].centroid or {},
                        clusters[j].centroid or {},
                    )
                    if sim > best_sim:
                        best_sim = sim
                        best_pair = (i, j)

            # Stop if similarity too low
            if best_sim < min_similarity:
                break

            # Merge clusters
            i, j = best_pair
            merged = self._merge_clusters(clusters[i], clusters[j])
            clusters = [c for k, c in enumerate(clusters) if k not in (i, j)]
            clusters.append(merged)

        # Generate cluster names based on top keywords
        for cluster in clusters:
            cluster.keywords = self._get_top_terms(cluster, n=5)
            cluster.name = self._generate_cluster_name(cluster)

        return sorted(clusters, key=lambda c: c.size, reverse=True)

    def _merge_clusters(self, c1: DocumentCluster, c2: DocumentCluster) -> DocumentCluster:
        """Merge two clusters into one."""
        merged = DocumentCluster(
            id=min(c1.id, c2.id),
            name=f"Merged {c1.id}-{c2.id}",
            documents=c1.documents + c2.documents,
        )

        # Compute new centroid as average
        if c1.centroid and c2.centroid:
            all_terms = set(c1.centroid.keys()) | set(c2.centroid.keys())
            merged.centroid = {}
            for term in all_terms:
                v1 = c1.centroid.get(term, 0)
                v2 = c2.centroid.get(term, 0)
                merged.centroid[term] = (v1 + v2) / 2

        return merged

    def _get_top_terms(self, cluster: DocumentCluster, n: int = 5) -> list[str]:
        """Get top terms for a cluster."""
        if not cluster.centroid:
            return []

        sorted_terms = sorted(
            cluster.centroid.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return [term for term, score in sorted_terms[:n]]

    def _generate_cluster_name(self, cluster: DocumentCluster) -> str:
        """Generate a descriptive name for cluster."""
        if not cluster.keywords:
            return f"Cluster {cluster.id}"

        # Use top 2-3 keywords
        name_parts = cluster.keywords[:3]
        return " / ".join(name_parts).title()

    def get_similar_documents(self, doc_path: str, n: int = 5) -> list[tuple[str, float]]:
        """Find documents most similar to a given document."""
        if doc_path not in self._document_vectors:
            return []

        target_vec = self._document_vectors[doc_path]
        similarities = []

        for other_path, other_vec in self._document_vectors.items():
            if other_path == doc_path:
                continue
            sim = self._cosine_similarity(target_vec, other_vec)
            similarities.append((other_path, sim))

        return sorted(similarities, key=lambda x: x[1], reverse=True)[:n]

    def get_document_cluster_assignment(
        self,
        clusters: list[DocumentCluster],
    ) -> dict[str, int]:
        """Get cluster assignment for each document."""
        assignment = {}
        for cluster in clusters:
            for doc_path in cluster.documents:
                assignment[doc_path] = cluster.id
        return assignment


def cluster_by_document_type(file_paths: list[str]) -> dict[str, list[str]]:
    """
    Simple clustering by document type based on filename patterns.

    Args:
        file_paths: List of file paths

    Returns:
        Dict mapping category to list of files
    """
    categories = {
        "contracts": ["contract", "agreement", "amendment", "addendum"],
        "pleadings": ["complaint", "answer", "motion", "brief", "memorandum"],
        "discovery": ["interrogator", "deposition", "request", "response"],
        "orders": ["order", "judgment", "ruling", "decision"],
        "correspondence": ["letter", "email", "memo", "correspondence"],
        "exhibits": ["exhibit", "attachment", "appendix"],
        "reports": ["report", "analysis", "summary", "review"],
    }

    result: dict[str, list[str]] = {cat: [] for cat in categories}
    result["other"] = []

    for file_path in file_paths:
        filename_lower = Path(file_path).stem.lower()
        categorized = False

        for category, keywords in categories.items():
            if any(kw in filename_lower for kw in keywords):
                result[category].append(file_path)
                categorized = True
                break

        if not categorized:
            result["other"].append(file_path)

    # Remove empty categories
    return {k: v for k, v in result.items() if v}
