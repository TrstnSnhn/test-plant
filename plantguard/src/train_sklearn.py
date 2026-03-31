from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from config import TRAIN_DIR, VAL_DIR, TEST_DIR, RESULTS_DIR
from utils.seed import set_seed
from PIL import Image

def load_images_flat(split_dir: Path, size: int = 64):
    """Load images as flattened arrays for sklearn."""
    X, y = [], []
    class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
    class_to_idx = {d.name: i for i, d in enumerate(class_dirs)}
    for d in class_dirs:
        for img_path in d.glob("*"):
            if not img_path.is_file():
                continue
            try:
                img = Image.open(img_path).resize((size, size)).convert("RGB")
                X.append(np.array(img).flatten())
                y.append(class_to_idx[d.name])
            except Exception:
                continue
    return np.array(X), np.array(y), list(class_to_idx.keys())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["training"].get("seed", 42))

    print("Loading train images...")
    X_train, y_train, class_names = load_images_flat(TRAIN_DIR)
    print(f"  Train: {X_train.shape}")
    print("Loading val images...")
    X_val, y_val, _ = load_images_flat(VAL_DIR)
    print(f"  Val: {X_val.shape}")

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    start = time.time()
    clf.fit(X_train, y_train)
    elapsed = time.time() - start

    val_preds = clf.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    val_f1 = f1_score(y_val, val_preds, average="macro")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {"model": "sklearn_rf", "val_acc": val_acc, "val_macro_f1": val_f1, "train_time_sec": elapsed}
    (RESULTS_DIR / "sklearn_rf_results.json").write_text(json.dumps(results, indent=2))
    print(f"sklearn RF — val_acc={val_acc:.4f}, val_f1={val_f1:.4f}, time={elapsed:.1f}s")

if __name__ == "__main__":
    main()
