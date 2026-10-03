from __future__ import annotations

import gc
import glob
import os
import re
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np
import polars as pl

from .candidate_index import CountryCandidateIndex
from .config import CONFIG
from .features import compute_pair_features, get_token_idf
from .inference import evaluate_macro_f05
from .normalization import normalize_record
from .training import build_training_matrix, fit_ensemble_models


def build_document_frequency(paths):
    token_doc_freq = defaultdict(int)
    total_sample_docs = 0
    for p in paths:
        if not p or not os.path.isfile(p):
            continue
        df_part = pl.read_csv(p, separator="\t", columns=["business_name", "business_address"], n_rows=300000)
        total_sample_docs += len(df_part)
        for row in df_part.iter_rows():
            clean = re.sub(r"[^\w\s]", " ", f"{row[0] or ''} {row[1] or ''}".lower())
            for token in set(clean.split()):
                if len(token) >= 3:
                    token_doc_freq[token] += 1
    return dict(token_doc_freq), total_sample_docs


def run_pipeline():
    print("Polars version     :", pl.__version__)
    print("Python environment :", sys.version.split()[0])

    print("Building corpus-wide token document-frequency table across sources...")
    token_doc_freq, total_sample_docs = build_document_frequency([
        f"{CONFIG.train_dir}/train_source1.tsv",
        f"{CONFIG.train_dir}/train_source2.tsv",
        f"{CONFIG.train_dir}/train_source3.tsv",
        f"{CONFIG.test_dir}/test_source1.tsv",
        f"{CONFIG.test_dir}/test_source2.tsv",
        f"{CONFIG.test_dir}/test_source3.tsv",
    ])
    print(f"Corpus DF table compiled ({total_sample_docs:,} docs sampled, {len(token_doc_freq):,} vocabulary).")

    print("Loading ground truth labels...")
    gt_df = pl.read_csv(f"{CONFIG.train_dir}/train_ground_truth.tsv", separator="\t")
    true_matches_all = defaultdict(set)
    for row in gt_df.iter_rows():
        s1 = str(row[0]).strip()
        matched = str(row[1] or "").strip()
        if matched:
            for m in matched.split(","):
                m = m.strip()
                if m:
                    true_matches_all[s1].add(m)

    df_s1_full = pl.read_csv(f"{CONFIG.train_dir}/train_source1.tsv", separator="\t")
    print(f"Total entities available in Source 1: {len(df_s1_full):,}")

    df_train_pool = df_s1_full.slice(0, 500000)
    train_us = df_train_pool.filter(pl.col("country") == "US").sample(n=9000, seed=42)
    train_in = df_train_pool.filter(pl.col("country") == "India").sample(n=6000, seed=42)
    df_train_s1 = pl.concat([train_us, train_in]).sample(fraction=1.0, shuffle=True, seed=42)
    train_s1_ids = df_train_s1["entity_id"].to_list()
    train_s1_set = set(train_s1_ids)

    df_cold_pool = df_s1_full.slice(1000000, None)
    cold_us = df_cold_pool.filter(pl.col("country") == "US").sample(n=6000, seed=2026)
    cold_in = df_cold_pool.filter(pl.col("country") == "India").sample(n=4000, seed=2026)
    df_cold_s1 = pl.concat([cold_us, cold_in]).sample(fraction=1.0, shuffle=True, seed=2026)
    cold_s1_ids = df_cold_s1["entity_id"].to_list()
    cold_s1_set = set(cold_s1_ids)

    assert len(train_s1_set.intersection(cold_s1_set)) == 0
    print("PASS: Zero entity overlap confirmed.")

    s1_train_records = {row[0]: normalize_record(row) for row in df_train_s1.iter_rows()}
    s1_cold_records = {row[0]: normalize_record(row) for row in df_cold_s1.iter_rows()}

    pool_train_records = {}
    pool_cold_records = {}
    for fn in ["train_source2.tsv", "train_source3.tsv"]:
        df_src = pl.read_csv(f"{CONFIG.train_dir}/{fn}", separator="\t")
        df_tr_tgt = df_src.filter(pl.col("entity_id").is_in(list(set(m for s in train_s1_ids for m in true_matches_all.get(s, set())))))
        bg_us_tr = df_src.filter(pl.col("country") == "US").slice(0, 200000)
        bg_in_tr = df_src.filter(pl.col("country") == "India").slice(0, 200000)
        for row in pl.concat([df_tr_tgt, bg_us_tr, bg_in_tr]).unique(subset=["entity_id"]).iter_rows():
            pool_train_records[row[0]] = normalize_record(row)

        df_cd_tgt = df_src.filter(pl.col("entity_id").is_in(list(set(m for s in cold_s1_ids for m in true_matches_all.get(s, set())))))
        bg_us_cd = df_src.filter(pl.col("country") == "US").slice(500000, 200000)
        bg_in_cd = df_src.filter(pl.col("country") == "India").slice(500000, 200000)
        for row in pl.concat([df_cd_tgt, bg_us_cd, bg_in_cd]).unique(subset=["entity_id"]).iter_rows():
            pool_cold_records[row[0]] = normalize_record(row)

    print(f"Train Pool Size: {len(pool_train_records):,} | Cold Realistic Pool Size: {len(pool_cold_records):,}")

    x_train, y_train, groups_train, pair_meta_train = build_training_matrix(
        train_s1_ids,
        s1_train_records,
        pool_train_records,
        true_matches_all,
        token_doc_freq,
        total_sample_docs,
    )
    models, oof_blend = fit_ensemble_models(x_train, y_train, groups_train)

    tr_oof_pairs = defaultdict(list)
    for i, (s1_id, mid) in enumerate(pair_meta_train):
        tr_oof_pairs[s1_id].append((mid, float(oof_blend[i])))

    best_t_global, best_f = 0.83, 0.0
    for th in np.linspace(0.70, 0.90, 21):
        preds = {s: set(m for m, p in tr_oof_pairs[s] if p >= th) for s in train_s1_ids}
        sc = evaluate_macro_f05(true_matches_all, preds, train_s1_ids)
        if sc > best_f:
            best_f, best_t_global = sc, float(th)

    print(f"Frozen OOF Global Threshold : {best_t_global:.3f} (OOF Macro F0.5: {best_f:.5f})")

    print("Scoring 10,000 COLD VIRGIN entities with realistic background pool...")
    cold_country_indices = {}
    for country in ["US", "India"]:
        country_pool = {eid: rec for eid, rec in pool_cold_records.items() if rec["country"] == country}
        idx = CountryCandidateIndex(country)
        idx.build(country_pool)
        cold_country_indices[country] = idx

    cold_claims = defaultdict(list)
    batch_pairs, batch_meta = [], []
    for s1_id in cold_s1_ids:
        rec1 = s1_cold_records[s1_id]
        idx = cold_country_indices.get(rec1["country"])
        if not idx:
            continue
        cands_meta = idx.query(rec1, max_candidates=CONFIG.max_candidates_per_entity)
        for mid in cands_meta:
            rec2 = pool_cold_records.get(mid)
            if not rec2:
                continue
            if rec1["norm_name"] == rec2["norm_name"] and rec1["norm_address"] == rec2["norm_address"]:
                cold_claims[mid].append((s1_id, 1.0, "fast"))
            elif (
                rec1.get("postal_code")
                and rec2.get("postal_code")
                and rec1["postal_code"] == rec2["postal_code"]
                and len(rec1["postal_code"]) >= 5
                and rec1["norm_name"] and rec2["norm_name"]
                and rec1["norm_name"] != rec2["norm_name"]
            ):
                cold_claims[mid].append((s1_id, 0.999, "rule"))
            else:
                batch_pairs.append(compute_pair_features(rec1, rec2, retrieval_meta=cands_meta.get(mid), token_doc_freq=token_doc_freq, total_sample_docs=total_sample_docs))
                batch_meta.append((s1_id, mid))

    if batch_pairs:
        X_b = np.array(batch_pairs, dtype=np.float32)
        p1 = models["xgboost"].predict_proba(X_b)[:, 1]
        p2 = models["lightgbm"].predict_proba(X_b)[:, 1]
        p3 = models["catboost"].predict_proba(X_b)[:, 1]
        blend_p = 0.40 * p1 + 0.35 * p2 + 0.25 * p3
        for (s1_ref, mid_ref), prob in zip(batch_meta, blend_p):
            if float(prob) >= 0.20:
                cold_claims[mid_ref].append((s1_ref, float(prob), "model"))

    resolved_cold_matches = defaultdict(list)
    for mid, claimants in cold_claims.items():
        claimants.sort(key=lambda x: x[1], reverse=True)
        best_s1, best_pr, best_src = claimants[0]
        c_th = CONFIG.segment_thresholds.get(s1_cold_records[best_s1]["country"], CONFIG.global_decision_threshold)
        if best_pr >= c_th:
            resolved_cold_matches[best_s1].append((mid, best_pr, best_src))

    final_cold_matches = {}
    cold_singletons = 0
    for s1 in cold_s1_ids:
        m_list = sorted(resolved_cold_matches.get(s1, []), key=lambda x: x[1], reverse=True)[: CONFIG.max_matches_per_entity]
        final_cold_matches[s1] = set(x[0] for x in m_list)
        if not final_cold_matches[s1]:
            cold_singletons += 1

    cold_f05 = evaluate_macro_f05(true_matches_all, final_cold_matches, cold_s1_ids)
    print(f"Cold Macro F0.5 Score   : {cold_f05:.5f} ({cold_f05*100:.2f}%)")
    print(f"Singletons (0 matches)  : {cold_singletons:,} / {len(cold_s1_ids):,}")

    print("Generating official submission output files on test set...")
    s1_test_file = f"{CONFIG.test_dir}/test_source1.tsv"
    s2_test_file = f"{CONFIG.test_dir}/test_source2.tsv"
    s3_test_file = f"{CONFIG.test_dir}/test_source3.tsv"

    all_test_s1 = []
    test_countries = set()
    with open(s1_test_file, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        id_col, c_col = header.index("entity_id"), header.index("country")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                all_test_s1.append(parts[id_col])
                test_countries.add(parts[c_col])

    final_candidates = {s1: [] for s1 in all_test_s1}
    final_matches = {s1: [] for s1 in all_test_s1}

    def load_tsv_by_country(path: str, target_country: str):
        df = pl.read_csv(path, separator="\t").filter(pl.col("country") == target_country)
        return {row[0]: normalize_record(row) for row in df.iter_rows()}

    for country in sorted(test_countries):
        c_th = CONFIG.segment_thresholds.get(country, CONFIG.segment_thresholds["DEFAULT"])
        s1_c_recs = load_tsv_by_country(s1_test_file, country)
        pool_c_recs = load_tsv_by_country(s2_test_file, country)
        pool_c_recs.update(load_tsv_by_country(s3_test_file, country))

        country_index = CountryCandidateIndex(country)
        country_index.build(pool_c_recs)
        country_claims = defaultdict(list)
        batch_pairs = []
        batch_meta = []

        for s1_id, rec1 in s1_c_recs.items():
            cands = country_index.query(rec1, max_candidates=CONFIG.max_candidates_per_entity)
            final_candidates[s1_id] = sorted(cands.keys())
            for mid in cands:
                rec2 = pool_c_recs.get(mid)
                if rec2 is None:
                    continue
                if rec1["norm_name"] == rec2["norm_name"] and rec1["norm_address"] == rec2["norm_address"]:
                    country_claims[mid].append((s1_id, 1.0))
                elif (
                    rec1.get("postal_code")
                    and rec2.get("postal_code")
                    and rec1["postal_code"] == rec2["postal_code"]
                    and len(rec1["postal_code"]) >= 5
                    and rec1["norm_name"] and rec2["norm_name"]
                    and rec1["norm_name"] != rec2["norm_name"]
                ):
                    country_claims[mid].append((s1_id, 0.999))
                else:
                    feats = compute_pair_features(rec1, rec2, retrieval_meta=cands.get(mid), token_doc_freq=token_doc_freq, total_sample_docs=total_sample_docs)
                    batch_pairs.append(feats)
                    batch_meta.append((s1_id, mid))

            if len(batch_pairs) >= CONFIG.batch_size:
                X_b = np.array(batch_pairs, dtype=np.float32)
                p1 = models["xgboost"].predict_proba(X_b)[:, 1]
                p2 = models["lightgbm"].predict_proba(X_b)[:, 1]
                p3 = models["catboost"].predict_proba(X_b)[:, 1]
                blend_p = 0.40 * p1 + 0.35 * p2 + 0.25 * p3
                for (s1_ref, mid_ref), prob in zip(batch_meta, blend_p):
                    if float(prob) >= 0.20:
                        country_claims[mid_ref].append((s1_ref, float(prob)))
                batch_pairs, batch_meta = [], []

        country_resolved = defaultdict(list)
        for mid, claimants in country_claims.items():
            claimants.sort(key=lambda x: x[1], reverse=True)
            best_s1, best_pr = claimants[0]
            if best_pr >= c_th:
                country_resolved[best_s1].append((mid, best_pr))

        for s1_id in s1_c_recs:
            m_list = sorted(country_resolved.get(s1_id, []), key=lambda x: x[1], reverse=True)[: CONFIG.max_matches_per_entity]
            final_matches[s1_id] = [mid for mid, _ in m_list]
            cand_set = set(final_candidates[s1_id])
            for mid in final_matches[s1_id]:
                if mid not in cand_set:
                    final_candidates[s1_id].append(mid)

    cand_out_path = f"{CONFIG.output_dir}/candidate_pairs.tsv"
    match_out_path = f"{CONFIG.output_dir}/matching_results.tsv"
    with open(cand_out_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1 in all_test_s1:
            f.write(f"{s1}\t{','.join(final_candidates[s1])}\n")

    with open(match_out_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1 in all_test_s1:
            f.write(f"{s1}\t{','.join(final_matches[s1])}\n")

    val_script_candidates = [
        "student_resource/utils/validate_submission.py",
        "utils/validate_submission.py",
        f"{CONFIG.test_dir}/student_resource/utils/validate_submission.py",
        f"{CONFIG.test_dir}/utils/validate_submission.py",
    ]
    for p in glob.glob("/kaggle/input/**/validate_submission.py", recursive=True):
        val_script_candidates.append(p)

    val_script = next((p for p in val_script_candidates if os.path.isfile(p)), None)
    if val_script and os.path.exists(cand_out_path) and os.path.exists(match_out_path):
        cmd = [sys.executable, val_script, "--matching", match_out_path, "--candidate", cand_out_path, "--test-dir", CONFIG.test_dir]
        print(f"Running validator: {' '.join(cmd)}")
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout)
        if res.stderr:
            print("Validator stderr:\n", res.stderr)
        if res.returncode == 0:
            print("SUCCESS: Validator confirmed files are safe to submit!")
        else:
            print(f"Validator exited with code {res.returncode}")
    else:
        print(f"Validator script not found or output files missing. Status: script={val_script}, candidate={os.path.exists(cand_out_path)}, matching={os.path.exists(match_out_path)}")


if __name__ == "__main__":
    run_pipeline()
