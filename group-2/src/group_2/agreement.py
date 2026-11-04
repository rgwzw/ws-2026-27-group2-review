"""Compute Laplacian variance and analyze agreement with the first metric."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.stats import spearmanr
from group_2.second_metric import compute_laplacian_variance

def main():
    root = Path("datasets")
    scores_file = Path("group-2/results/fallback-v1/scores.jsonl")
    manifest_file = Path("synthetic-data-seminar/manifest.jsonl")
    
    id_to_path = {}
    with open(manifest_file, "r") as f:
        for line in f:
            row = json.loads(line)
            if row.get("eligibility", {}).get("analysis"):
                id_to_path[row["synthetic_id"]] = row["source_relpath"]
                
    scores1 = {}
    with open(scores_file, "r") as f:
        for line in f:
            row = json.loads(line)
            scores1[row["synthetic_id"]] = row["score"]
            
    print("Computing second metric...")
    scores2 = {}
    valid_ids = []
    
    for sid in list(scores1.keys())[:2000]:
        rel_path = id_to_path[sid]
        parts = Path(rel_path).parts
        suffix = parts[parts.index("synthetic-data"):]
        img_path = root.joinpath(*suffix)
        
        try:
            with Image.open(img_path) as img:
                arr = np.asarray(img.convert("RGB"), dtype=np.float64)
                val = compute_laplacian_variance(arr)
                scores2[sid] = val
                valid_ids.append(sid)
        except Exception:
            pass
            
    s1 = [scores1[sid] for sid in valid_ids]
    s2 = [scores2[sid] for sid in valid_ids]
    
    corr, pval = spearmanr(s1, s2)
    print(f"Spearman correlation: {corr:.4f} (p-value: {pval:.4e})")
    
    out_csv = Path("group-2/results/fallback-v1/metric_agreement.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w") as f:
        f.write("synthetic_id,metric1_score,metric2_laplacian\n")
        for sid in valid_ids:
            f.write(f"{sid},{scores1[sid]},{scores2[sid]}\n")
            
    print(f"Agreement results saved to {out_csv}")

if __name__ == "__main__":
    main()
