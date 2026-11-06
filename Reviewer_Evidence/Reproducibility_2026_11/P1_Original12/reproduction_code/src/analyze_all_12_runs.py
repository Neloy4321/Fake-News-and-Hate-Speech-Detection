#!/usr/bin/env python3
"""Recompute 12 original TALLIP predictions, paired tests and figures.

Usage: python3 analyze_all_12_runs.py --input-dir ../tallip_stat_inputs --output-dir results
Requires: numpy, scipy, scikit-learn, matplotlib.
No training, data relabeling, or synthetic predictions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.stats import binomtest
from sklearn import __version__ as sklearn_version
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)


SEEDS = (42, 52, 62)
DATASETS = ("fakenews", "hatespeech")
MODELS = ("banglabert", "mbert")
METRICS = ("accuracy", "macro_precision", "macro_recall", "macro_f1", "roc_auc")
ARCHIVES = {
    ("fakenews", "banglabert", 42): "TALLIP_Seed42_Results_ONLY.zip",
    ("fakenews", "banglabert", 52): "TALLIP_fakenews_banglabert_seed_52_evidence.zip",
    ("fakenews", "banglabert", 62): "TALLIP_fakenews_banglabert_seed_62_evidence.zip",
    ("fakenews", "mbert", 42): "TALLIP_FakeNews_mBERT_Seed42_Results.zip",
    ("fakenews", "mbert", 52): "TALLIP_FakeNews_mBERT_Seed52_Results.zip",
    ("fakenews", "mbert", 62): "TALLIP_FakeNews_mBERT_Seed62_Results.zip",
    ("hatespeech", "banglabert", 42): "TALLIP_HateSpeech_BanglaBERT_Seed42_Results.zip",
    ("hatespeech", "banglabert", 52): "TALLIP_HateSpeech_BanglaBERT_Seed52_Results.zip",
    ("hatespeech", "banglabert", 62): "TALLIP_HateSpeech_BanglaBERT_Seed62_Results.zip",
    **{("hatespeech", "mbert", seed): "TALLIP_HateSpeech_mBERT_3Seeds_Results.zip"
       for seed in SEEDS},
}
EXPECTED = {
    "fakenews": {"counts_before": {"train": 75905, "val": 9489, "test": 9489},
                 "counts_after": {"train": 75902, "val": 9489, "test": 9489}},
    "hatespeech": {"counts_before": {"train": 40224, "val": 5028, "test": 5029},
                   "counts_after": {"train": 40216, "val": 5028, "test": 5029}},
}


def calc_metrics(y, pred, prob):
    if not (len(y) == len(pred) == len(prob) and len(y)):
        raise ValueError("Test arrays are empty or differ in length")
    if not (np.isin(y, [0, 1]).all() and np.isin(pred, [0, 1]).all()):
        raise ValueError("Non-binary true/predicted labels")
    if not (np.isfinite(prob).all() and np.all((0 <= prob) & (prob <= 1))):
        raise ValueError("Invalid probability values")
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    return {"accuracy": float(accuracy_score(y, pred)), "macro_precision": float(p.mean()),
            "macro_recall": float(r.mean()), "macro_f1": float(f.mean()),
            "roc_auc": float(roc_auc_score(y, prob))}


def load_run(input_dir, dataset, model, seed):
    file = input_dir / ARCHIVES[(dataset, model, seed)]
    prefix = f"real_runs/hatespeech/mbert/seed_{seed}/" if dataset == "hatespeech" and model == "mbert" else ""
    with zipfile.ZipFile(file) as z:
        assert z.testzip() is None, f"Corrupt ZIP: {file}"
        needed = ("COMPLETE.json", "metrics.json", "data_audit.json",
                  "training_history.json", "test_predictions.npz")
        assert all(prefix + n in z.namelist() for n in needed), f"Missing evidence: {file}"
        complete = json.loads(z.read(prefix+"COMPLETE.json"))
        meta = json.loads(z.read(prefix+"metrics.json"))
        audit = json.loads(z.read(prefix+"data_audit.json"))
        history = json.loads(z.read(prefix+"training_history.json"))
        with np.load(io.BytesIO(z.read(prefix+"test_predictions.npz")), allow_pickle=False) as arr:
            y, pred, prob, ids = (arr[k].copy() for k in
                                  ("y_true", "y_pred", "y_prob", "test_row_id"))
    assert complete.get("finished") is True and not meta.get("experimental_only"), file
    assert (meta.get("task"), meta.get("model_key"), meta.get("seed")) == (dataset, model, seed), file
    assert all(audit.get(k) == v for k, v in EXPECTED[dataset].items()), file
    assert audit.get("normalizer") == "bangla" and meta.get("normalizer") == "bangla"
    assert audit.get("test_preserved") and audit["rows_removed_to_prevent_overlap"]["test"] == 0
    assert len(y) == EXPECTED[dataset]["counts_before"]["test"] and len(ids) == len(y)
    assert len(np.unique(ids)) == len(ids), "Non-unique test row IDs"
    assert len(history) == meta["epochs"] == 3
    best = max(history, key=lambda x: x["macro_f1"])
    assert best["epoch"] == meta["best_epoch"]
    assert abs(best["macro_f1"]-meta["best_validation_macro_f1"]) < 1e-10
    scores = calc_metrics(y, pred, prob)
    for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
        assert abs(scores[key]-meta["test"][key]) < 1e-10, (file, key)
    assert abs(scores["macro_f1"]-complete["test_macro_f1"]) < 1e-10, file
    return {"y": y, "pred": pred, "prob": prob, "id": ids, "scores": scores,
            "cm": confusion_matrix(y, pred, labels=[0, 1]),
            "meta": meta, "audit": audit, "filename": file.name}


def macro_f1_from_cm(tn, fp, fn, tp):
    """Vectorized version of binary macro F1 with zero_division=0."""
    d0, d1 = 2*tn+fp+fn, 2*tp+fp+fn
    f0 = np.divide(2*tn, d0, out=np.zeros_like(d0, dtype=float), where=d0 != 0)
    f1 = np.divide(2*tp, d1, out=np.zeros_like(d1, dtype=float), where=d1 != 0)
    return (f0+f1)/2


def paired_bootstrap(y, a, b, replicates=10000, seed=20261101):
    """Resample matched row triples using their eight-cell multinomial distribution."""
    assert len(y) == len(a) == len(b)
    n = len(y)
    groups = np.bincount(y*4+a*2+b, minlength=8)
    draws = np.random.default_rng(seed).multinomial(n, groups/n, size=replicates)
    a_tn, a_fp = draws[:,0]+draws[:,1], draws[:,2]+draws[:,3]
    a_fn, a_tp = draws[:,4]+draws[:,5], draws[:,6]+draws[:,7]
    b_tn, b_fp = draws[:,0]+draws[:,2], draws[:,1]+draws[:,3]
    b_fn, b_tp = draws[:,4]+draws[:,6], draws[:,5]+draws[:,7]
    diff_acc = (a_tn+a_tp-b_tn-b_tp)/n
    diff_f1 = macro_f1_from_cm(a_tn,a_fp,a_fn,a_tp)-macro_f1_from_cm(b_tn,b_fp,b_fn,b_tp)
    return {"delta_accuracy_ci95": np.quantile(diff_acc, [0.025,0.975]).tolist(),
            "delta_macro_f1_ci95": np.quantile(diff_f1, [0.025,0.975]).tolist()}


def exact_mcnemar(y, a, b):
    ca, cb = a == y, b == y
    a_only = int(np.count_nonzero(ca & ~cb))
    b_only = int(np.count_nonzero(~ca & cb))
    n = a_only+b_only
    p = float(binomtest(min(a_only,b_only), n, p=.5).pvalue) if n else 1.0
    return {"banglabert_correct_mbert_wrong": a_only,
            "mbert_correct_banglabert_wrong": b_only,
            "discordant_pairs": n, "mcnemar_exact_two_sided_p": p}


def holm(pvalues):
    """Holm correction for a prespecified family of tests."""
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    output = [0.0]*len(pvalues)
    running = 0.
    for rank, ix in enumerate(order):
        running = max(running, min(1., pvalues[ix]*(len(pvalues)-rank)))
        output[ix] = running
    return output


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v,list) else v
                        for k,v in row.items()})


def plot_figures(runs, aggregates, out):
    # Keep datasets in separate panels; their class populations are incomparable.
    fig, axes = plt.subplots(1,2,figsize=(12,4.7))
    for ax, ds in zip(axes, DATASETS):
        group = [next(x for x in aggregates if x["dataset"]==ds and x["model"]==m)
                 for m in MODELS]
        xx = np.arange(2)
        for j,metric in enumerate(("accuracy","macro_f1")):
            ax.bar(xx+(j-.5)*.34,[x[metric+"_mean"]*100 for x in group],width=.34,
                   yerr=[x[metric+"_sd_sample"]*100 for x in group],
                   color=("#2563a6","#da8251")[j], capsize=4, label=metric)
        ax.set_xticks(xx,["BanglaBERT","mBERT"])
        ax.set_ylim(0,100)
        ax.set_title("Fake News (n=9489)" if ds=="fakenews" else "Hate Speech (n=5029)")
        ax.set_ylabel("Test metric %, mean ± sample SD")
        ax.grid(axis="y",alpha=.2)
        ax.legend(loc="upper right")
    fig.suptitle("Three training seeds — original locked test labels")
    fig.tight_layout()
    fig.savefig(out/"model_comparison_mean_sd.png",dpi=220,bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2,2,figsize=(8.5,8))
    for i,ds in enumerate(DATASETS):
        for j,model in enumerate(MODELS):
            ax=axes[i,j]
            cm=runs[(ds,model,42)]["cm"]
            ax.imshow(cm,cmap="Blues")
            for y in range(2):
                for x in range(2):
                    ax.text(x,y,str(cm[y,x]),ha="center",va="center",
                            color="white" if cm[y,x]>.55*cm.max() else "black")
            ax.set_xticks([0,1]); ax.set_yticks([0,1])
            ax.set_xlabel("Prediction"); ax.set_ylabel("Source label")
            ax.set_title(ds+" / "+model+" / seed 42")
    fig.tight_layout()
    fig.savefig(out/"confusion_matrices_seed42.png",dpi=220,bbox_inches="tight")
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4.7))
    for ax,ds in zip(axes,DATASETS):
        for model in MODELS:
            r=runs[(ds,model,42)]
            fpr,tpr,_=roc_curve(r["y"],r["prob"])
            ax.plot(fpr,tpr,lw=2,label=f"{model} (AUC {r['scores']['roc_auc']:.3f})")
        ax.plot([0,1],[0,1],"--",color="gray")
        ax.set_title(ds+" / seed 42")
        ax.set_xlabel("False-positive rate"); ax.set_ylabel("True-positive rate")
        ax.legend(loc="lower right"); ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out/"roc_curves_seed42.png",dpi=220,bbox_inches="tight")
    plt.close(fig)


def markdown_report(aggregates, paired, manifests, count):
    lines=[
        "# ACM TALLIP-26-0521 — Phase 1 statistical analysis",
        "",
        "All 12 model-seed experiments were **recomputed from genuine archived test predictions**.",
        "No model retraining, data relabeling or invented measurements were performed.",
        "",
        "## Verified split populations",
        "",
        "| Dataset | Train before/after overlap filtering | Validation | Locked test |",
        "|---|---:|---:|---:|",
        "| Fake News | 75,905 / 75,902 | 9,489 | 9,489 |",
        "| Hate Speech | 40,224 / 40,216 | 5,028 | 5,029 |",
        "",
        "Across seeds and models within each dataset, test labels, row IDs and processed-text",
        "audit hashes match. All original test rows were retained.",
        "",
        "## Per-model test score (mean ± sample SD in percentage points)",
        "",
        "| Dataset | Model | Accuracy % | Macro Precision % | Macro Recall % | Macro F1 % | ROC-AUC % |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for x in aggregates:
        vals=[f"{x[m+'_mean']*100:.3f} ± {x[m+'_sd_sample']*100:.3f}" for m in METRICS]
        lines.append("| "+x["dataset"]+" | "+x["model"]+" | "+" | ".join(vals)+" |")
    lines += [
        "", "## Paired BanglaBERT versus mBERT on the same test instances",
        "",
        "Positive deltas favor BanglaBERT. McNemar p values are exact two-sided",
        "binomial p values based on discordant paired correctness (an accuracy question).",
        "Holm correction spans all six dataset×seed comparisons.",
        f"Macro-F1 and accuracy delta CIs are 95% paired bootstrap intervals ({count:,} resamples),",
        "**not** across-seed confidence intervals.",
        "",
        "| Dataset | Seed | ΔAccuracy pp [95% CI] | ΔMacro-F1 pp [95% CI] | McNemar p (Holm) |",
        "|---|---:|---|---|---:|",
    ]
    for x in paired:
        ca=x["delta_accuracy_ci95"]; cf=x["delta_macro_f1_ci95"]
        lines.append(
            f"| {x['dataset']} | {x['seed']} | {x['delta_accuracy']*100:+.3f} "
            f"[{ca[0]*100:+.3f}, {ca[1]*100:+.3f}] | "
            f"{x['delta_macro_f1']*100:+.3f} "
            f"[{cf[0]*100:+.3f}, {cf[1]*100:+.3f}] | "
            f"{x['mcnemar_exact_two_sided_p']:.3g} ({x['holm_adjusted_p']:.3g}) |")
    lines += [
        "", "## Methodological cautions",
        "",
        "1. The seeds share the same fixed original test data; they do not establish independent-domain generalization.",
        "2. Statistical tests treat test rows as independent. Article or comment clusters can make",
        "   simple row bootstrap and McNemar significance optimistic; cluster robustness was not run.",
        "3. McNemar tests error-rate differences; it does not directly test the macro-F1 gap.",
        "4. Sample SD across 3 training seeds is distinct from a paired bootstrap test-row CI.",
        "5. Source dataset labels are unchanged. Human-adjudicated labels and robustness to",
        "   corrected labels have NOT been analyzed in this phase.",
        "6. Previous training logs warned of potentially nondeterministic attention; bitwise",
        "   reproducibility of fresh GPU training cannot be claimed.",
        "7. These tests do not constitute XAI faithfulness, text-length ablation, or leakage-free",
        "   generalization beyond the checked cross-split exact-text overlaps.",
        "",
        "## Provenance",
        "",
        f"All {manifests['run_count']} COMPLETE markers, training histories, validation-checkpoint",
        "selection rules and saved predictions matched the recomputed metrics.",
        "All archives' SHA-256 file digests are recorded in the JSON manifest.",
        "",
    ]
    return "\n".join(lines)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir",type=Path,default=Path("../tallip_stat_inputs"))
    ap.add_argument("--output-dir",type=Path,default=Path("results"))
    ap.add_argument("--bootstrap-replicates",type=int,default=10000)
    ap.add_argument("--bootstrap-seed",type=int,default=20261101)
    a=ap.parse_args()
    if a.bootstrap_replicates<100:
        ap.error("At least 100 bootstrap draws are required")
    a.output_dir.mkdir(parents=True,exist_ok=True)
    runs={key:load_run(a.input_dir,*key) for key in ARCHIVES}
    alignment={}
    for ds in DATASETS:
        reference=runs[(ds,"banglabert",42)]
        for model in MODELS:
            for seed in SEEDS:
                r=runs[(ds,model,seed)]
                assert np.array_equal(r["y"],reference["y"]),f"Test truth mismatch: {ds}/{model}/{seed}"
                assert np.array_equal(r["id"],reference["id"]),f"Test row ID mismatch: {ds}/{model}/{seed}"
                assert r["audit"]["sha256_first_1000_texts"]==reference["audit"]["sha256_first_1000_texts"]
        alignment[ds]={"test_count":len(reference["y"]),
                       "true_class_counts":{str(k):int(v) for k,v in Counter(reference["y"]).items()},
                       "all_seed_model_test_alignment":"PASS",
                       "text_fingerprints":reference["audit"]["sha256_first_1000_texts"]}

    singles=[]; aggregates=[]
    for ds in DATASETS:
        for model in MODELS:
            group=[runs[(ds,model,seed)] for seed in SEEDS]
            for seed,r in zip(SEEDS,group):
                cm=r["cm"]
                singles.append({"dataset":ds,"model":model,"seed":seed,"test_n":len(r["y"]),
                                **r["scores"],"tn":int(cm[0,0]),"fp":int(cm[0,1]),
                                "fn":int(cm[1,0]),"tp":int(cm[1,1]),
                                "best_validation_macro_f1":r["meta"]["best_validation_macro_f1"],
                                "best_epoch":r["meta"]["best_epoch"],"source_archive":r["filename"]})
            row={"dataset":ds,"model":model,"seeds":"42,52,62","test_n":len(group[0]["y"])}
            for metric in METRICS:
                val=np.array([g["scores"][metric] for g in group])
                row[metric+"_mean"]=float(val.mean())
                row[metric+"_sd_sample"]=float(val.std(ddof=1))
            aggregates.append(row)

    paired=[]
    for ds in DATASETS:
        for seed in SEEDS:
            r1,r2=runs[(ds,"banglabert",seed)],runs[(ds,"mbert",seed)]
            row={"dataset":ds,"seed":seed,"test_n":len(r1["y"]),
                 "delta_accuracy":r1["scores"]["accuracy"]-r2["scores"]["accuracy"],
                 "delta_macro_f1":r1["scores"]["macro_f1"]-r2["scores"]["macro_f1"]}
            row.update(exact_mcnemar(r1["y"],r1["pred"],r2["pred"]))
            row.update(paired_bootstrap(r1["y"],r1["pred"],r2["pred"],
                                       replicates=a.bootstrap_replicates,
                                       seed=a.bootstrap_seed+seed+(1000 if ds=="hatespeech" else 0)))
            paired.append(row)
    adjusted=holm([x["mcnemar_exact_two_sided_p"] for x in paired])
    for x,pval in zip(paired,adjusted):
        x["holm_adjusted_p"]=pval

    write_csv(a.output_dir/"per_seed_metrics.csv",singles)
    write_csv(a.output_dir/"model_mean_sd.csv",aggregates)
    write_csv(a.output_dir/"paired_model_tests.csv",paired)
    plot_figures(runs,aggregates,a.output_dir)
    checks={"run_count":12,"checkpoint_history":"PASS","prediction_recomputation":"PASS",
            "test_alignment":"PASS"}
    (a.output_dir/"TALLIP_Phase1_Statistical_Results.md").write_text(
        markdown_report(aggregates,paired,checks,a.bootstrap_replicates),encoding="utf-8")
    hashes={}
    for filename in sorted(set(ARCHIVES.values())):
        hashes[filename]=hashlib.sha256((a.input_dir/filename).read_bytes()).hexdigest()
    manifest={"generated_at_utc":datetime.now(timezone.utc).isoformat(),
              "project":"TALLIP-26-0521", "no_new_training":True,
              "input_zip_sha256":hashes,"alignment":alignment,"checks":checks,
              "bootstrap_draws":a.bootstrap_replicates,"bootstrap_seed":a.bootstrap_seed,
              "holm_test_family_size":len(paired),
              "python":platform.python_version(),"numpy":np.__version__,
              "scipy":scipy.__version__,"sklearn":sklearn_version,
              "model_aggregates":aggregates,"paired_tests":paired}
    (a.output_dir/"analysis_manifest.json").write_text(
        json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8")
    print("PASS: 12 original runs, original test labels, data and metric alignment.")
    for x in aggregates:
        print(f"{x['dataset']:10s} {x['model']:10s}: "
              f"accuracy={100*x['accuracy_mean']:.4f}±{100*x['accuracy_sd_sample']:.4f}% "
              f"macro-F1={100*x['macro_f1_mean']:.4f}±{100*x['macro_f1_sd_sample']:.4f}%")
    for x in paired:
        print(f"Paired {x['dataset']:10s} seed{x['seed']}: "
              f"Δmacro-F1={100*x['delta_macro_f1']:+.3f}pp, "
              f"CI={[round(100*v,3) for v in x['delta_macro_f1_ci95']]}, "
              f"McNemar={x['mcnemar_exact_two_sided_p']:.4g}, "
              f"Holm={x['holm_adjusted_p']:.4g}")
    print("Wrote:",a.output_dir.resolve())


if __name__=="__main__":
    main()
