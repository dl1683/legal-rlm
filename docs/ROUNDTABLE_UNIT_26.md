# Roundtable Unit 26: Document Clustering

## Unit Goal
Group related documents by content similarity without heavy ML dependencies.

## Success Criteria
1. [x] DocumentCluster dataclass
2. [x] TF-IDF vectorization
3. [x] Agglomerative clustering
4. [x] Similarity search
5. [x] Document type categorization

## Changes Made

### clustering.py (NEW)
| Component | Description |
|-----------|-------------|
| DocumentCluster | Cluster of related documents |
| DocumentClusterer | TF-IDF based clustering |
| cluster_by_document_type() | Simple filename-based categorization |

### DocumentCluster Class
| Field | Description |
|-------|-------------|
| id | Cluster identifier |
| name | Auto-generated descriptive name |
| documents | List of document paths |
| keywords | Top terms characterizing cluster |
| centroid | TF-IDF centroid vector |

### DocumentClusterer Methods
| Method | Description |
|--------|-------------|
| fit() | Build TF-IDF vectors from documents |
| cluster() | Agglomerative clustering |
| get_similar_documents() | Find similar docs |
| get_document_cluster_assignment() | Get cluster for each doc |

### Algorithm
1. **Tokenization**: Extract words, remove stop words
2. **TF-IDF**: Compute term frequency-inverse document frequency
3. **Cosine Similarity**: Measure document similarity
4. **Agglomerative Clustering**: Merge most similar clusters iteratively
5. **Naming**: Auto-generate names from top keywords

### Document Type Categories
| Category | Keywords |
|----------|----------|
| contracts | contract, agreement, amendment |
| pleadings | complaint, motion, brief |
| discovery | interrogatory, deposition, request |
| orders | order, judgment, ruling |
| correspondence | letter, email, memo |
| exhibits | exhibit, attachment, appendix |
| reports | report, analysis, summary |

### Key Code
```python
@dataclass
class DocumentCluster:
    id: int
    name: str
    documents: list[str]
    keywords: list[str]
    centroid: Optional[dict]

class DocumentClusterer:
    def fit(self, documents: dict[str, str]):
        self._build_vocabulary(documents)
        self._compute_idf(documents)
        self._compute_tfidf(documents)

    def cluster(self, num_clusters: int = 5) -> list[DocumentCluster]:
        # Initialize each doc as cluster
        # Iteratively merge most similar
        while len(clusters) > num_clusters:
            best_pair = self._find_most_similar_pair(clusters)
            if best_sim < min_similarity:
                break
            clusters = self._merge_clusters(best_pair, clusters)
        return clusters

    def _cosine_similarity(self, vec1, vec2) -> float:
        dot = sum(vec1[t] * vec2[t] for t in common_terms)
        return dot / (norm1 * norm2)
```

## Usage Example
```python
from irys.core.clustering import DocumentClusterer, cluster_by_document_type

# Content-based clustering
clusterer = DocumentClusterer()
clusterer.fit({"doc1.pdf": "...", "doc2.pdf": "..."})
clusters = clusterer.cluster(num_clusters=5)

for cluster in clusters:
    print(f"{cluster.name}: {cluster.size} docs")
    print(f"  Keywords: {cluster.keywords}")

# Find similar documents
similar = clusterer.get_similar_documents("doc1.pdf", n=5)

# Simple type-based clustering
categories = cluster_by_document_type(file_paths)
```

## Review Notes
- No external ML dependencies (pure Python)
- TF-IDF provides good baseline similarity
- Agglomerative clustering works well for small-medium repositories
- Document type clustering useful for quick organization
- Can be integrated with investigation to prioritize document groups

## Next Unit
Unit 27: Relevance Feedback
