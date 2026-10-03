from collections import Counter, defaultdict
from typing import Any, Dict

from .config import CONFIG
from .normalization import BLOCK_STOPWORDS


class CountryCandidateIndex:
    def __init__(self, country: str, max_name_ratio: float = None, max_postal_ratio: float = None):
        self.country = country
        self.max_name_ratio = max_name_ratio or CONFIG.max_name_ratio
        self.max_postal_ratio = max_postal_ratio or CONFIG.max_postal_ratio
        self.name_token_index = defaultdict(list)
        self.postal_index = defaultdict(list)
        self.records = {}
        self.token_freqs = Counter()

    def build(self, records: Dict[str, Dict[str, Any]]):
        self.records = records
        pool_size = len(records)
        name_cutoff = max(250, int(pool_size * self.max_name_ratio))
        postal_cutoff = max(500, int(pool_size * self.max_postal_ratio))

        for eid, rec in records.items():
            st = rec.get("search_tokens") or rec["name_tokens"]
            for tok in st:
                self.name_token_index[tok].append(eid)
                self.token_freqs[tok] += 1
            if rec.get("postal_code"):
                self.postal_index[rec["postal_code"]].append(eid)

        pruned_name = 0
        for tok in list(self.name_token_index.keys()):
            if len(self.name_token_index[tok]) > name_cutoff:
                del self.name_token_index[tok]
                pruned_name += 1

        pruned_postal = 0
        for postal in list(self.postal_index.keys()):
            if len(self.postal_index[postal]) > postal_cutoff:
                del self.postal_index[postal]
                pruned_postal += 1

        return {
            "country": self.country,
            "pool_size": pool_size,
            "name_token_count": len(self.name_token_index),
            "postal_count": len(self.postal_index),
            "pruned_name": pruned_name,
            "pruned_postal": pruned_postal,
        }

    def query(self, query_rec: Dict[str, Any], max_candidates: int = None) -> Dict[str, Dict[str, Any]]:
        import heapq

        if max_candidates is None:
            max_candidates = CONFIG.max_candidates_per_entity

        scores = defaultdict(float)
        st = query_rec.get("search_tokens") or query_rec["name_tokens"]
        for tok in st:
            ids = self.name_token_index.get(tok)
            if ids:
                w = self._token_weight(tok, len(ids))
                for mid in ids:
                    scores[mid] += w

        postal = query_rec.get("postal_code")
        if postal and postal in self.postal_index:
            for mid in self.postal_index[postal]:
                scores[mid] += 2.0

        if not scores:
            sorted_tokens = sorted(query_rec["name_tokens"], key=lambda t: self.token_freqs.get(t, 0))
            for tok in sorted_tokens:
                ids = self.name_token_index.get(tok)
                if ids:
                    w = self._token_weight(tok, len(ids))
                    for mid in ids[:max_candidates]:
                        scores[mid] += w
                    break

        if not scores:
            return {}
        if len(scores) <= max_candidates:
            top_ids = list(scores.keys())
        else:
            top_ids = heapq.nlargest(max_candidates, scores.keys(), key=scores.get)
        return {mid: {"retrieval_score": scores[mid]} for mid in top_ids}

    def _token_weight(self, tok: str, id_count: int) -> float:
        if tok in BLOCK_STOPWORDS:
            return 0.0
        return 1.0 / (id_count ** 0.5)
