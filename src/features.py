import math
from typing import List, Optional

from rapidfuzz import fuzz

from .normalization import BLOCK_STOPWORDS


def get_token_idf(tok: str, token_doc_freq: dict, total_sample_docs: int) -> float:
    df = token_doc_freq.get(tok, 1)
    return math.log(1.0 + (total_sample_docs - df + 0.5) / (df + 0.5))


def compute_pair_features(r1: dict, r2: dict, retrieval_meta: Optional[dict] = None, token_doc_freq: Optional[dict] = None, total_sample_docs: int = 1) -> List[float]:
    n1, n2 = r1["norm_name"], r2["norm_name"]
    a1, a2 = r1["norm_address"], r2["norm_address"]

    f_n_ratio = fuzz.ratio(n1, n2) / 100.0
    f_n_partial = fuzz.partial_ratio(n1, n2) / 100.0
    f_n_tsort = fuzz.token_sort_ratio(n1, n2) / 100.0
    f_n_tset = fuzz.token_set_ratio(n1, n2) / 100.0

    f_a_ratio = fuzz.ratio(a1, a2) / 100.0
    f_a_partial = fuzz.partial_ratio(a1, a2) / 100.0
    f_a_tsort = fuzz.token_sort_ratio(a1, a2) / 100.0
    f_a_tset = fuzz.token_set_ratio(a1, a2) / 100.0

    n_tok1, n_tok2 = r1["name_tokens"], r2["name_tokens"]
    n_inter = n_tok1.intersection(n_tok2)
    n_union = n_tok1.union(n_tok2)
    n_jaccard = len(n_inter) / max(len(n_union), 1)

    a_tok1, a_tok2 = r1["addr_tokens"], r2["addr_tokens"]
    a_inter = a_tok1.intersection(a_tok2)
    a_union = a_tok1.union(a_tok2)
    a_jaccard = len(a_inter) / max(len(a_union), 1)

    p1, p2 = r1.get("postal_code"), r2.get("postal_code")
    postal_match = 1.0 if (p1 and p2 and p1 == p2) else (0.0 if (p1 and p2) else 0.5)

    len_n_ratio = min(len(n1), len(n2)) / max(len(n1), len(n2), 1)
    len_a_ratio = min(len(a1), len(a2)) / max(len(a1), len(a2), 1)
    ret_score = (retrieval_meta.get("retrieval_score", 0.0) if retrieval_meta else 0.0) / 10.0

    if token_doc_freq is None:
        token_doc_freq = {}

    n_w_inter = sum(get_token_idf(t, token_doc_freq, total_sample_docs) for t in n_inter)
    n_w_union = sum(get_token_idf(t, token_doc_freq, total_sample_docs) for t in n_union)
    idf_name_jaccard = n_w_inter / max(n_w_union, 1e-6)

    a_w_inter = sum(get_token_idf(t, token_doc_freq, total_sample_docs) for t in a_inter)
    a_w_union = sum(get_token_idf(t, token_doc_freq, total_sample_docs) for t in a_union)
    idf_addr_jaccard = a_w_inter / max(a_w_union, 1e-6)

    n_diff = (n_tok1 ^ n_tok2) - BLOCK_STOPWORDS
    idf_name_diff_penalty = min(sum(get_token_idf(t, token_doc_freq, total_sample_docs) for t in n_diff) / 20.0, 1.0)
    idf_shared_name_weight = min(n_w_inter / 20.0, 1.0)
    idf_shared_addr_weight = min(a_w_inter / 30.0, 1.0)

    composite = [
        f_n_ratio * f_a_ratio,
        f_n_tset * f_a_tset,
        f_n_ratio * postal_match,
        n_jaccard * a_jaccard,
        (f_n_ratio + f_a_ratio) / 2.0,
        f_n_partial * f_a_partial,
        f_n_tsort * f_a_tsort,
        idf_name_jaccard * idf_addr_jaccard,
        idf_name_jaccard * (1.0 - idf_name_diff_penalty),
    ]

    base_feats = [
        f_n_ratio, f_n_partial, f_n_tsort, f_n_tset,
        f_a_ratio, f_a_partial, f_a_tsort, f_a_tset,
        n_jaccard, a_jaccard, postal_match,
        len_n_ratio, len_a_ratio, ret_score,
        idf_name_jaccard, idf_addr_jaccard, idf_name_diff_penalty,
        idf_shared_name_weight, idf_shared_addr_weight,
    ] + composite

    padding = [0.0] * (72 - len(base_feats))
    return base_feats + padding
