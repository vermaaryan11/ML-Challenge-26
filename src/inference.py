from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .config import CONFIG
from .features import compute_pair_features


def evaluate_macro_f05(gt_mapping: Dict[str, set], pred_mapping: Dict[str, set], query_ids: Iterable[str]) -> float:
    f05_scores = []
    for q in query_ids:
        true_set = gt_mapping.get(q, set())
        pred_set = pred_mapping.get(q, set())
        if len(true_set) == 0 and len(pred_set) == 0:
            f05_scores.append(1.0)
        elif len(true_set) == 0 or len(pred_set) == 0:
            f05_scores.append(0.0)
        else:
            tp = len(true_set.intersection(pred_set))
            p = tp / max(len(pred_set), 1)
            r = tp / max(len(true_set), 1)
            f05 = 0.0 if (1.25 * p + r) == 0 else (1.25 * p * r) / (0.25 * p + r)
            f05_scores.append(f05)
    return float(np.mean(f05_scores))


def score_country_records(
    country: str,
    s1_records: Dict[str, dict],
    pool_records: Dict[str, dict],
    models: Dict[str, object],
    candidate_index,
    token_doc_freq: dict,
    total_sample_docs: int,
) -> Tuple[Dict[str, List[str]], Dict[str, List[Tuple[str, float]]]]:
    final_candidates = {}
    final_matches = defaultdict(list)

    for s1_id, rec1 in s1_records.items():
        cands = candidate_index.query(rec1, max_candidates=CONFIG.max_candidates_per_entity)
        final_candidates[s1_id] = sorted(cands.keys())
        batch_pairs, batch_meta = [], []

        for mid in final_candidates[s1_id]:
            rec2 = pool_records.get(mid)
            if rec2 is None:
                continue
            if rec1["norm_name"] == rec2["norm_name"] and rec1["norm_address"] == rec2["norm_address"]:
                final_matches[s1_id].append((mid, 1.0))
            else:
                feats = compute_pair_features(rec1, rec2, retrieval_meta=cands.get(mid), token_doc_freq=token_doc_freq, total_sample_docs=total_sample_docs)
                batch_pairs.append(feats)
                batch_meta.append((s1_id, mid))

        if batch_pairs:
            x_b = np.array(batch_pairs, dtype=np.float32)
            p1 = models["xgboost"].predict_proba(x_b)[:, 1]
            p2 = models["lightgbm"].predict_proba(x_b)[:, 1]
            p3 = models["catboost"].predict_proba(x_b)[:, 1]
            blend = 0.40 * p1 + 0.35 * p2 + 0.25 * p3
            for (s1_ref, mid_ref), prob in zip(batch_meta, blend):
                if float(prob) >= 0.20:
                    final_matches[s1_ref].append((mid_ref, float(prob)))

    resolved = {}
    threshold = CONFIG.segment_thresholds.get(country, CONFIG.segment_thresholds["DEFAULT"])
    for s1_id, claims in final_matches.items():
        claims.sort(key=lambda x: x[1], reverse=True)
        best = claims[0] if claims else None
        if best and best[1] >= threshold:
            resolved[s1_id] = [mid for mid, _ in claims[: CONFIG.max_matches_per_entity]]
        else:
            resolved[s1_id] = []

    return final_candidates, resolved


def write_submission_file(path: str, rows: List[Tuple[str, List[str]]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id, cands in rows:
            f.write(f"{s1_id}\t{','.join(cands)}\n")
