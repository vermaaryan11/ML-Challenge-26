import glob
import os


# Primary Kaggle dataset path specified by user
KAGGLE_DATASET_PATH = "/kaggle/input/datasets/aamod06/corpus"


def find_dataset_dir(target_name: str) -> str:
    candidates = [
        os.path.join(KAGGLE_DATASET_PATH, target_name),
        os.path.join(KAGGLE_DATASET_PATH, "dataset", target_name),
        os.path.join(KAGGLE_DATASET_PATH, "student_resource", "dataset", target_name),
        KAGGLE_DATASET_PATH,
        f"/kaggle/input/**/{target_name}",
        f"/kaggle/input/**/dataset/{target_name}",
        f"/kaggle/input/*/{target_name}",
        f"student_resource/dataset/{target_name}",
        f"dataset/{target_name}",
        f"../dataset/{target_name}",
        target_name,
    ]
    expected_file = f"{target_name}_source1.tsv"
    for cand in candidates:
        if "*" in cand:
            for matched in glob.glob(cand, recursive=True):
                if os.path.isfile(os.path.join(matched, expected_file)):
                    return matched
                elif os.path.isdir(matched) and os.path.basename(matched) == target_name:
                    return matched
        else:
            if os.path.isfile(os.path.join(cand, expected_file)):
                return cand
            elif os.path.isdir(cand) and cand != KAGGLE_DATASET_PATH and os.path.basename(cand) == target_name:
                return cand
    if os.path.exists(KAGGLE_DATASET_PATH):
        sub = os.path.join(KAGGLE_DATASET_PATH, target_name)
        return sub if os.path.exists(sub) else KAGGLE_DATASET_PATH
    return f"student_resource/dataset/{target_name}"


TRAIN_DIR = find_dataset_dir("train")
TEST_DIR = find_dataset_dir("test")
OUTPUT_DIR = "/kaggle/working/output" if os.path.exists("/kaggle/working") else "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("output", exist_ok=True)


class PipelineConfig:
    train_dir: str = TRAIN_DIR
    test_dir: str = TEST_DIR
    output_dir: str = OUTPUT_DIR

    ensemble_weights = {
        "xgboost": 0.40,
        "lightgbm": 0.35,
        "catboost": 0.25,
    }

    max_name_ratio: float = 0.010
    max_postal_ratio: float = 0.020
    max_candidates_per_entity: int = 75
    graded_negs_per_entity: int = 10
    max_matches_per_entity: int = 10

    global_decision_threshold: float = 0.800
    segment_thresholds = {
        "US": 0.800,
        "India": 0.760,
        "France": 0.820,
        "DEFAULT": 0.800,
    }

    random_state: int = 42
    batch_size: int = 25000


CONFIG = PipelineConfig()
