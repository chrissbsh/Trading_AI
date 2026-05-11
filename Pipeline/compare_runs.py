"""
compare_runs.py — Tableau comparatif de tous les runs xgb_baseline_v1.

Usage :
    python -m pipeline.compare_runs
    python -m pipeline.compare_runs --last 10          # seulement les 10 derniers runs
    python -m pipeline.compare_runs --model lightgbm   # lightgbm | xgboost | both (défaut)
    python -m pipeline.compare_runs --sort sharpe       # sharpe | f1 | bal_acc | return (défaut: sharpe)
    python -m pipeline.compare_runs --no-wf            # masque les colonnes walk-forward
"""

import os
import json
import argparse
import datetime
import numpy as np
import pandas as pd

RESULTS_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def load_run(run_dir: str, model: str) -> dict | None:
    """Charge toutes les métriques d'un run pour un modèle donné (lightgbm ou xgboost)."""
    row = {"run_id": os.path.basename(run_dir)}

    # ── Config pipeline ──────────────────────────────────────────────
    config_path = os.path.join(run_dir, "run_config.json")
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
        p = cfg.get("pipeline", {})
        row["timestamp"]        = cfg.get("timestamp", "")
        row["horizon"]          = p.get("pred_horizon", "")
        row["threshold"]        = str(p.get("fixed_thresholds", ""))
        row["thresh_strategy"]  = p.get("threshold_strategy", "")
        cw = p.get("class_weight_boost", {})
        row["cw_boost"]         = f"0:{cw.get('0', '?')} 1:{cw.get('1', '?')} 2:{cw.get('2', '?')}"
        row["ecart_min"]        = p.get("ecart_min", "")
        row["top_n_feat"]       = p.get("top_n_features", "")
        flags = cfg.get("run_flags", {})
        row["tune"]             = flags.get("tune", "")
        row["walk_forward"]     = flags.get("walk_forward", "")
        row["lag_features"]     = flags.get("lag_features", "")
        row["time_weighting"]   = flags.get("time_weighting", "")
        lgb_params = cfg.get("lightgbm_best_params") or {}
        row["lgb_n_est"]        = lgb_params.get("n_estimators", "")
        row["lgb_lr"]           = round(lgb_params.get("learning_rate", float("nan")), 4) if lgb_params.get("learning_rate") else ""
        row["lgb_depth"]        = lgb_params.get("max_depth", "")
        row["lgb_leaves"]       = lgb_params.get("num_leaves", "")
    else:
        row["timestamp"] = ""

    # ── Métriques holdout ────────────────────────────────────────────
    sub = os.path.join(run_dir, model)
    holdout_path = os.path.join(sub, "holdout_metrics.csv")
    if not os.path.exists(holdout_path):
        return None
    hm_df = pd.read_csv(holdout_path)
    # Préfère la ligne "calibrated" si elle existe, sinon "raw" ou première ligne
    if "variant" in hm_df.columns and "calibrated" in hm_df["variant"].values:
        hm = hm_df[hm_df["variant"] == "calibrated"].iloc[0]
        hm_raw = hm_df[hm_df["variant"] == "raw"].iloc[0]
        row["f1_macro_raw"] = round(hm_raw.get("f1_macro", float("nan")), 4)
        row["bal_acc_raw"]  = round(hm_raw.get("balanced_accuracy", float("nan")), 4)
    else:
        hm = hm_df.iloc[0]
    row["accuracy"]     = round(hm.get("accuracy", float("nan")), 4)
    row["bal_acc"]      = round(hm.get("balanced_accuracy", float("nan")), 4)
    row["f1_macro"]     = round(hm.get("f1_macro", float("nan")), 4)
    row["f1_weighted"]  = round(hm.get("f1_weighted", float("nan")), 4)
    row["n_samples"]    = int(hm.get("n_holdout_samples", 0))

    # ── Classification report : recall par classe ─────────────────────
    cr_path = os.path.join(sub, "classification_report.csv")
    if os.path.exists(cr_path):
        cr = pd.read_csv(cr_path, index_col=0)
        for cls in ["0", "2"]:
            if cls in cr.index:
                row[f"recall_c{cls}"] = round(cr.loc[cls, "recall"], 3)
            else:
                row[f"recall_c{cls}"] = float("nan")
        row["recall_c1"] = round(cr.loc["1", "recall"], 3) if "1" in cr.index else float("nan")

    # ── Métriques backtest ────────────────────────────────────────────
    bt_path = os.path.join(sub, "backtest_metrics.csv")
    if os.path.exists(bt_path):
        bt = pd.read_csv(bt_path).iloc[0]
        row["n_trades"]     = int(bt.get("n_trades", 0))
        row["return_pct"]   = round(bt.get("total_return", float("nan")) * 100, 1)
        row["win_rate"]     = round(bt.get("win_rate", float("nan")) * 100, 1)
        row["sharpe"]       = round(bt.get("sharpe_ratio", float("nan")), 3)
        row["max_dd"]       = round(bt.get("max_drawdown", float("nan")) * 100, 1)

    # ── Walk-forward : moyenne sur les folds ─────────────────────────
    wf_path = os.path.join(run_dir, "walk_forward_metrics.csv")
    if os.path.exists(wf_path):
        wf = pd.read_csv(wf_path)
        row["wf_bal_acc"]  = round(wf["balanced_accuracy"].mean(), 3)
        row["wf_f1"]       = round(wf["f1_macro"].mean(), 3)
        row["wf_f1_std"]   = round(wf["f1_macro"].std(), 3)
        row["wf_f1_c0"]    = round(wf["f1_class0"].mean(), 3)
        row["wf_f1_c2"]    = round(wf["f1_class2"].mean(), 3)
    else:
        row["wf_bal_acc"] = row["wf_f1"] = row["wf_f1_std"] = float("nan")
        row["wf_f1_c0"]  = row["wf_f1_c2"] = float("nan")

    return row


def load_all_runs(model: str, prefix: str = "xgb_baseline_v1") -> pd.DataFrame:
    runs = []
    for name in sorted(os.listdir(RESULTS_ROOT)):
        if not name.startswith(prefix):
            continue
        run_dir = os.path.join(RESULTS_ROOT, name)
        if not os.path.isdir(run_dir):
            continue
        row = load_run(run_dir, model)
        if row is not None:
            runs.append(row)
    return pd.DataFrame(runs)


def color_val(val, thresholds: list[tuple], colors: list[str], default: str = "") -> str:
    """Retourne un code ANSI pour colorier val selon les seuils (du plus bas au plus haut)."""
    ansi = {
        "green":  "\033[92m",
        "yellow": "\033[93m",
        "red":    "\033[91m",
        "reset":  "\033[0m",
    }
    if not isinstance(val, (int, float)) or np.isnan(val):
        return default
    for (lo, hi), color in zip(thresholds, colors):
        if lo <= val < hi:
            return ansi.get(color, "") + str(val) + ansi["reset"]
    return str(val)


def print_table(df: pd.DataFrame, show_wf: bool, show_config: bool) -> None:
    # Colonnes à afficher selon les options
    core_cols = [
        "run_id", "timestamp",
        "bal_acc", "f1_macro", "bal_acc_raw", "f1_macro_raw",
        "recall_c0", "recall_c1", "recall_c2",
        "n_trades", "return_pct", "win_rate", "sharpe", "max_dd",
    ]
    config_cols = [
        "horizon", "threshold", "thresh_strategy", "cw_boost", "ecart_min",
        "top_n_feat", "tune", "lag_features", "time_weighting",
        "lgb_n_est", "lgb_lr", "lgb_depth", "lgb_leaves",
    ]
    wf_cols = ["wf_bal_acc", "wf_f1", "wf_f1_std", "wf_f1_c0", "wf_f1_c2"]

    cols = core_cols[:]
    if show_config:
        cols += config_cols
    if show_wf:
        cols += wf_cols

    available = [c for c in cols if c in df.columns]
    display = df[available].copy()

    # Remplace NaN par "-"
    display = display.fillna("-")

    # Formatage console en colonnes alignées
    col_widths = {c: max(len(c), display[c].astype(str).str.len().max()) for c in available}

    header = "  ".join(c.ljust(col_widths[c]) for c in available)
    sep    = "  ".join("-" * col_widths[c] for c in available)
    print("\n" + header)
    print(sep)

    for _, row in display.iterrows():
        parts = []
        for c in available:
            val = row[c]
            w   = col_widths[c]
            parts.append(str(val).ljust(w))
        print("  ".join(parts))

    print(f"\n{len(display)} runs affichés.\n")


def main():
    parser = argparse.ArgumentParser(description="Comparaison des runs xgb_baseline_v1")
    parser.add_argument("--last",    type=int,   default=None,        help="Afficher seulement les N derniers runs")
    parser.add_argument("--model",   type=str,   default="lightgbm",  help="lightgbm | xgboost | both")
    parser.add_argument("--sort",    type=str,   default="sharpe",    help="sharpe | f1 | bal_acc | return")
    parser.add_argument("--no-wf",   action="store_true",             help="Masquer les colonnes walk-forward")
    parser.add_argument("--config",  action="store_true",             help="Afficher les colonnes de config pipeline")
    parser.add_argument("--min-f1",  type=float, default=None,        help="Filtrer : garder seulement f1_macro >= seuil")
    parser.add_argument("--prefix",  type=str,   default="xgb_baseline_v1", help="Préfixe des dossiers de runs")
    args = parser.parse_args()

    sort_map = {"sharpe": "sharpe", "f1": "f1_macro", "bal_acc": "bal_acc", "return": "return_pct"}
    sort_col = sort_map.get(args.sort, "sharpe")

    models = ["lightgbm", "xgboost"] if args.model == "both" else [args.model]

    for model in models:
        print(f"\n{'='*60}")
        print(f"  MODÈLE : {model.upper()}")
        print(f"{'='*60}")

        df = load_all_runs(model, prefix=args.prefix)
        if df.empty:
            print("  Aucun run trouvé.")
            continue

        if args.min_f1 is not None and "f1_macro" in df.columns:
            df = df[pd.to_numeric(df["f1_macro"], errors="coerce") >= args.min_f1]

        if sort_col in df.columns:
            df = df.sort_values(sort_col, ascending=False, na_position="last")

        if args.last:
            df = df.head(args.last)

        print_table(df, show_wf=not args.no_wf, show_config=args.config)

        # Sauvegarde au format CSV dans results_compare
        save_dir = os.path.join(os.path.dirname(RESULTS_ROOT), "results_compare")
        os.makedirs(save_dir, exist_ok=True)
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"comparison_{model}_{timestamp_str}.csv")
        
        # On sauvegarde toutes les colonnes pour avoir le maximum d'infos
        # sep=";" et decimal="," pour une lecture directe facilitée sous Excel fr
        df.to_csv(save_path, index=False, sep=";", decimal=",")
        print(f"  [+] Fichier CSV généré : {save_path}\n")

        # Résumé statistique des métriques clés
        num_cols = ["bal_acc", "f1_macro", "sharpe", "return_pct", "max_dd"]
        num_cols = [c for c in num_cols if c in df.columns]
        numeric = df[num_cols].apply(pd.to_numeric, errors="coerce")
        print("  Statistiques :")
        print(f"  {'':20s} {'mean':>8s} {'std':>8s} {'min':>8s} {'max':>8s}")
        for c in num_cols:
            s = numeric[c].dropna()
            if len(s):
                print(f"  {c:20s} {s.mean():8.3f} {s.std():8.3f} {s.min():8.3f} {s.max():8.3f}")


if __name__ == "__main__":
    main()
