from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

from .candidate_index import CountryCandidateIndex
from .config import CONFIG
from .features import compute_pair_features


def build_training_matrix(
    s1_ids: List[str],
    s1_records: Dict[str, dict],
    pool_records: Dict[str, dict],
    true_matches_all: Dict[str, set],
    token_doc_freq: dict,
    total_sample_docs: int,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[Tuple[str, str]]]:
    train_country_indices = {}
    for country in ["US", "India"]:
        country_pool = {eid: rec for eid, rec in pool_records.items() if rec["country"] == country}
        country_index = CountryCandidateIndex(country)
        country_index.build(country_pool)
        train_country_indices[country] = country_index

    x_train_list, y_train_list, groups_train, pair_meta_train = [], [], [], []
    for s1_id in s1_ids:
        r1 = s1_records[s1_id]
        country_index = train_country_indices.get(r1["country"])
        if not country_index:
            continue
        candidate_meta = country_index.query(r1, max_candidates=50)
        true_matches = true_matches_all.get(s1_id, set())

        for mid in true_matches:
            r2 = pool_records.get(mid)
            if r2:
                feats = compute_pair_features(r1, r2, retrieval_meta=candidate_meta.get(mid), token_doc_freq=token_doc_freq, total_sample_docs=total_sample_docs)
                x_train_list.append(feats)
                y_train_list.append(1)
                groups_train.append(s1_id)
                pair_meta_train.append((s1_id, mid))

        hard_negs = [mid for mid in candidate_meta if mid not in true_matches]
        if len(hard_negs) <= CONFIG.graded_negs_per_entity:
            sampled_negs = hard_negs
        else:
            top_p = hard_negs[:4]
            mid_p = hard_negs[4:min(15, len(hard_negs))]
            low_p = hard_negs[min(15, len(hard_negs)) :]
            sampled_negs = list(top_p) + mid_p[:: max(1, len(mid_p) // 3)][:3] + low_p[:: max(1, len(low_p) // 3)][:3]
            sampled_negs = sampled_negs[: CONFIG.graded_negs_per_entity]

        for mid in sampled_negs:
            r2 = pool_records.get(mid)
            if r2:
                feats = compute_pair_features(r1, r2, retrieval_meta=candidate_meta.get(mid), token_doc_freq=token_doc_freq, total_sample_docs=total_sample_docs)
                x_train_list.append(feats)
                y_train_list.append(0)
                groups_train.append(s1_id)
                pair_meta_train.append((s1_id, mid))

    x_train = np.array(x_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.int32)
    return x_train, y_train, groups_train, pair_meta_train


def fit_ensemble_models(x_train: np.ndarray, y_train: np.ndarray, groups: List[str]) -> Tuple[dict, np.ndarray]:
    scale_w = float(np.sqrt((len(y_train) - np.sum(y_train)) / max(np.sum(y_train), 1)))
    oof_xgb = np.zeros(len(y_train), dtype=np.float32)
    oof_lgb = np.zeros(len(y_train), dtype=np.float32)
    oof_cb = np.zeros(len(y_train), dtype=np.float32)

    gkf = GroupKFold(n_splits=5)
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(x_train, y_train, groups=groups), 1):
        x_tr, y_tr = x_train[tr_idx], y_train[tr_idx]
        x_val, y_val = x_train[val_idx], y_train[val_idx]

        clf_x = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.08,
            scale_pos_weight=scale_w,
            random_state=42 + fold,
            n_jobs=-1,
            eval_metric="logloss",
        )
        clf_x.fit(x_tr, y_tr)
        oof_xgb[val_idx] = clf_x.predict_proba(x_val)[:, 1]

        clf_l = LGBMClassifier(
            n_estimators=100,
            num_leaves=31,
            learning_rate=0.08,
            scale_pos_weight=scale_w,
            random_state=42 + fold,
            n_jobs=-1,
            verbose=-1,
        )
        clf_l.fit(x_tr, y_tr)
        oof_lgb[val_idx] = clf_l.predict_proba(x_val)[:, 1]

        clf_c = CatBoostClassifier(
            iterations=120,
            depth=6,
            learning_rate=0.08,
            scale_pos_weight=scale_w,
            random_seed=42 + fold,
            thread_count=-1,
            verbose=0,
        )
        clf_c.fit(x_tr, y_tr)
        oof_cb[val_idx] = clf_c.predict_proba(x_val)[:, 1]

    oof_blend = 0.40 * oof_xgb + 0.35 * oof_lgb + 0.25 * oof_cb
    meta_clf = LogisticRegression(C=1.0, random_state=42)
    x_meta = np.column_stack([oof_xgb, oof_lgb, oof_cb])
    meta_clf.fit(x_meta, y_train)

    models = {
        "xgboost": XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.08,
            scale_pos_weight=scale_w,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=100,
            num_leaves=31,
            learning_rate=0.08,
            scale_pos_weight=scale_w,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
        "catboost": CatBoostClassifier(
            iterations=120,
            depth=6,
            learning_rate=0.08,
            scale_pos_weight=scale_w,
            random_seed=42,
            thread_count=-1,
            verbose=0,
        ),
    }
    models["xgboost"].fit(x_train, y_train)
    models["lightgbm"].fit(x_train, y_train)
    models["catboost"].fit(x_train, y_train)

    return models, oof_blend
