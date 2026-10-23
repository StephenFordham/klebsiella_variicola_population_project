#!/usr/bin/env python3
"""Genome-level PlasmidFinder replicon and AMR co-occurrence analyses.

These analyses test genome-level associations between detected PlasmidFinder
replicons and acquired AMR genes/statuses. They do not prove that an AMR gene is
located on a given plasmid unless contig-level localisation is available.
"""

from __future__ import annotations

import argparse
import math
import re
import textwrap
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import fisher_exact
from statsmodels.tools.sm_exceptions import ConvergenceWarning, PerfectSeparationWarning

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT / "Klebsiella_variicola_curated_population.csv"
DEFAULT_OUTDIR = PROJECT / "plasmid_amr_associations"

GENOME_ID = "Genome ID"
REPLICON_COL = "PlasmidFinder Inc Match"
ESBL_COLS = ["Bla_ESBL_acquired", "Bla_ESBL_inhR_acquired"]
CARB_COL = "Bla_Carb_acquired"
STATUS_OUTCOMES = ["ESBL_positive", "carbapenemase_positive", "ESBL_or_carbapenemase_positive"]
MISSING_VALUES = {"", "-", "0", "none", "nan", "na", "n/a", "null"}

STABILITY_RULES = {
    "replicon_present_min30": "replicon present in fewer than 30 genomes",
    "replicon_absent_min30": "replicon absent in fewer than 30 genomes",
    "outcome_present_min20": "outcome/gene present in fewer than 20 genomes",
    "outcome_absent_min30": "outcome/gene absent in fewer than 30 genomes",
    "all_cells_min5": "one or more 2x2 table cells <5",
    "finite_or_ci": "odds ratio or 95% CI is non-finite",
    "ci_ratio_max25": "95% CI ratio >25",
    "q_lt_0.05": "FDR-adjusted q-value >=0.05",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Curated Klebsiella variicola CSV")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="Output directory")
    parser.add_argument("--min-replicon-n", type=int, default=1, help="Minimum replicon count to test; stability filters are applied separately")
    parser.add_argument("--group-min-n", type=int, default=10, help="Minimum ST/country/source count retained as its own level in adjusted models")
    parser.add_argument("--top-n-lineage", type=int, default=40, help="Maximum stable hits included in lineage/geography sensitivity table")
    return parser.parse_args()


def is_present_value(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return str(value).strip().lower() not in MISSING_VALUES


def split_semicolon(value: object) -> list[str]:
    if not is_present_value(value):
        return []
    return [x.strip() for x in str(value).split(";") if is_present_value(x)]


def clean_gene(gene: str) -> str:
    gene = gene.strip()
    gene = re.sub(r"[?*^].*$", "", gene)
    gene = re.sub(r"-\d+%$", "", gene)
    gene = gene.replace(".v1", "")
    if gene.startswith("bla"):
        gene = gene[3:]
    return gene.strip()


def parse_gene_list(value: object) -> set[str]:
    return {clean_gene(x) for x in split_semicolon(value) if clean_gene(x)}


def bh_adjust(pvalues: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(pvalues), dtype=float)
    if len(p) == 0:
        return np.array([], dtype=float)
    p = np.where(np.isfinite(p), p, 1.0)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty(len(p), dtype=float)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def fmt_p(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return "P < 0.001" if value < 0.001 else f"P = {value:.3f}"


def fmt_q(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return "q < 0.001" if value < 0.001 else f"q = {value:.3f}"


def fmt_pct(n: int, d: int) -> str:
    return "NA" if d == 0 else f"{100*n/d:.1f}%"


@dataclass
class AssociationResult:
    replicon: str
    outcome: str
    outcome_type: str
    a: int
    b: int
    c: int
    d: int
    odds_ratio: float
    ci_low: float
    ci_high: float
    p_value: float
    replicon_present_n: int
    replicon_absent_n: int
    outcome_present_n: int
    outcome_absent_n: int
    prevalence_replicon_positive: float
    prevalence_replicon_negative: float


def association_test(data: pd.DataFrame, replicon: str, outcome: str, outcome_type: str) -> AssociationResult:
    rep = data[f"replicon::{replicon}"].astype(bool)
    out = data[outcome].astype(bool)
    a = int((rep & out).sum())
    b = int((rep & ~out).sum())
    c = int((~rep & out).sum())
    d = int((~rep & ~out).sum())
    odds, p = fisher_exact([[a, b], [c, d]])
    if min(a, b, c, d) > 0:
        ci_low = math.exp(math.log(odds) - 1.96 * math.sqrt(1/a + 1/b + 1/c + 1/d))
        ci_high = math.exp(math.log(odds) + 1.96 * math.sqrt(1/a + 1/b + 1/c + 1/d))
    else:
        # Haldane-Anscombe correction gives an approximate interval for sparse full tables.
        ac, bc, cc, dc = a + 0.5, b + 0.5, c + 0.5, d + 0.5
        corrected_or = (ac * dc) / (bc * cc)
        ci_low = math.exp(math.log(corrected_or) - 1.96 * math.sqrt(1/ac + 1/bc + 1/cc + 1/dc))
        ci_high = math.exp(math.log(corrected_or) + 1.96 * math.sqrt(1/ac + 1/bc + 1/cc + 1/dc))
    rep_n = a + b
    rep_abs = c + d
    out_n = a + c
    out_abs = b + d
    return AssociationResult(
        replicon=replicon,
        outcome=outcome,
        outcome_type=outcome_type,
        a=a,
        b=b,
        c=c,
        d=d,
        odds_ratio=float(odds),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=float(p),
        replicon_present_n=rep_n,
        replicon_absent_n=rep_abs,
        outcome_present_n=out_n,
        outcome_absent_n=out_abs,
        prevalence_replicon_positive=a / rep_n if rep_n else np.nan,
        prevalence_replicon_negative=c / rep_abs if rep_abs else np.nan,
    )


def result_rows(results: list[AssociationResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "replicon": r.replicon,
            "outcome": r.outcome,
            "outcome_type": r.outcome_type,
            "replicon_present_outcome_present": r.a,
            "replicon_present_outcome_absent": r.b,
            "replicon_absent_outcome_present": r.c,
            "replicon_absent_outcome_absent": r.d,
            "replicon_present_n": r.replicon_present_n,
            "replicon_absent_n": r.replicon_absent_n,
            "outcome_present_n": r.outcome_present_n,
            "outcome_absent_n": r.outcome_absent_n,
            "outcome_prevalence_replicon_positive": r.prevalence_replicon_positive,
            "outcome_prevalence_replicon_negative": r.prevalence_replicon_negative,
            "odds_ratio": r.odds_ratio,
            "ci_low": r.ci_low,
            "ci_high": r.ci_high,
            "ci_ratio": r.ci_high / r.ci_low if r.ci_low and np.isfinite(r.ci_low) and np.isfinite(r.ci_high) else np.inf,
            "fisher_p": r.p_value,
        })
    return pd.DataFrame(rows)


def apply_q_and_stability(df: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    if df.empty:
        df["q_value"] = []
        df["stable_association"] = []
        df["stability_failures"] = []
        return df
    df = df.copy()
    if group_cols:
        df["q_value"] = np.nan
        for _, idx in df.groupby(group_cols).groups.items():
            df.loc[list(idx), "q_value"] = bh_adjust(df.loc[list(idx), "fisher_p"])
    else:
        df["q_value"] = bh_adjust(df["fisher_p"])

    failures = []
    stable = []
    for row in df.itertuples(index=False):
        failed = []
        if row.replicon_present_n < 30:
            failed.append("replicon_present_min30")
        if row.replicon_absent_n < 30:
            failed.append("replicon_absent_min30")
        if row.outcome_present_n < 20:
            failed.append("outcome_present_min20")
        if row.outcome_absent_n < 30:
            failed.append("outcome_absent_min30")
        if min(row.replicon_present_outcome_present, row.replicon_present_outcome_absent, row.replicon_absent_outcome_present, row.replicon_absent_outcome_absent) < 5:
            failed.append("all_cells_min5")
        if not all(np.isfinite(x) for x in [row.odds_ratio, row.ci_low, row.ci_high]):
            failed.append("finite_or_ci")
        if not np.isfinite(row.ci_ratio) or row.ci_ratio > 25:
            failed.append("ci_ratio_max25")
        if not row.q_value < 0.05:
            failed.append("q_lt_0.05")
        failures.append("; ".join(f"{rule}: {STABILITY_RULES[rule]}" for rule in failed) if failed else "none")
        stable.append(len(failed) == 0)
    df["stable_association"] = stable
    df["stability_failures"] = failures
    return df.sort_values(["stable_association", "q_value", "odds_ratio"], ascending=[False, True, False])


def collapse_rare(series: pd.Series, min_n: int) -> pd.Series:
    s = series.fillna("Unknown").astype(str).str.strip().replace({"": "Unknown", "-": "Unknown"})
    counts = s.value_counts()
    return s.where(s.map(counts).fillna(0) >= min_n, "Other")


def load_and_prepare(path: Path, outdir: Path) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    data = pd.read_csv(path, dtype=str, keep_default_na=False)
    duplicate_mask = data.duplicated(GENOME_ID, keep=False)
    duplicates = data.loc[duplicate_mask].sort_values(GENOME_ID)
    duplicates.to_csv(outdir / "duplicate_genome_ids.csv", index=False)
    if duplicate_mask.any():
        data = data.drop_duplicates(GENOME_ID, keep="first").copy()

    data["replicon_set"] = data[REPLICON_COL].apply(lambda x: sorted(set(split_semicolon(x))))
    data["replicon_count_unique"] = data["replicon_set"].apply(len)
    data["any_replicon"] = data["replicon_count_unique"] > 0

    for col in ESBL_COLS + [CARB_COL]:
        if col not in data.columns:
            data[col] = "-"
    data["ESBL_positive"] = data[ESBL_COLS].apply(lambda row: any(is_present_value(x) for x in row), axis=1)
    data["carbapenemase_positive"] = data[CARB_COL].apply(is_present_value)
    data["ESBL_or_carbapenemase_positive"] = data["ESBL_positive"] | data["carbapenemase_positive"]
    data["esbl_carb_gene_set"] = data[ESBL_COLS + [CARB_COL]].apply(lambda row: sorted(set().union(*(parse_gene_list(x) for x in row))), axis=1)

    acquired_cols = [c for c in data.columns if c.endswith("_acquired")]
    data["all_amr_gene_set"] = data[acquired_cols].apply(lambda row: sorted(set().union(*(parse_gene_list(x) for x in row))), axis=1)

    replicons = sorted({r for values in data["replicon_set"] for r in values})
    target_genes = sorted({g for values in data["esbl_carb_gene_set"] for g in values})
    all_amr_genes = sorted({g for values in data["all_amr_gene_set"] for g in values})
    for replicon in replicons:
        data[f"replicon::{replicon}"] = data["replicon_set"].apply(lambda values, r=replicon: r in values)
    for gene in target_genes:
        data[f"target_gene::{gene}"] = data["esbl_carb_gene_set"].apply(lambda values, g=gene: g in values)
    for gene in all_amr_genes:
        data[f"amr_gene::{gene}"] = data["all_amr_gene_set"].apply(lambda values, g=gene: g in values)

    return data, replicons, target_genes, all_amr_genes


def summarise_inputs(data: pd.DataFrame, replicons: list[str], outdir: Path) -> None:
    rep_rows = []
    for r in replicons:
        n = int(data[f"replicon::{r}"].sum())
        rep_rows.append({"replicon": r, "genome_count": n, "prevalence": n / len(data)})
    pd.DataFrame(rep_rows).sort_values(["genome_count", "replicon"], ascending=[False, True]).to_csv(outdir / "replicon_summary.csv", index=False)

    rows = []
    for outcome in STATUS_OUTCOMES:
        n = int(data[outcome].sum())
        rows.append({"status": outcome, "positive_n": n, "negative_n": len(data) - n, "prevalence": n / len(data)})
    pd.DataFrame(rows).to_csv(outdir / "esbl_carbapenemase_status_summary.csv", index=False)


def run_association_set(data: pd.DataFrame, replicons: list[str], outcomes: list[tuple[str, str]], q_group_cols: list[str] | None) -> pd.DataFrame:
    results = []
    for replicon in replicons:
        for outcome, outcome_type in outcomes:
            results.append(association_test(data, replicon, outcome, outcome_type))
    df = result_rows(results)
    return apply_q_and_stability(df, group_cols=q_group_cols)


def lineage_sensitivity(data: pd.DataFrame, hits: pd.DataFrame, outdir: Path, top_n: int) -> pd.DataFrame:
    if hits.empty:
        empty = pd.DataFrame(columns=["analysis_group", "replicon", "outcome", "double_positive_n"])
        empty.to_csv(outdir / "lineage_geography_sensitivity_for_top_hits.csv", index=False)
        return empty
    ranked = hits[hits["stable_association"]].copy()
    if ranked.empty:
        ranked = hits.copy()
    ranked = ranked.sort_values(["q_value", "odds_ratio"], ascending=[True, False]).head(top_n)
    rows = []
    for row in ranked.itertuples(index=False):
        rep = data[f"replicon::{row.replicon}"].astype(bool)
        outcome_col = getattr(row, "outcome_column", row.outcome)
        out = data[outcome_col].astype(bool)
        outcome_label = str(row.outcome).replace("target_gene::", "").replace("amr_gene::", "")
        double = data[rep & out].copy()
        st_counts = double["ST"].fillna("Unknown").replace("", "Unknown").value_counts()
        country_counts = double["Country"].fillna("Unknown").replace("", "Unknown").value_counts()
        st_max = int(st_counts.iloc[0]) if len(st_counts) else 0
        country_max = int(country_counts.iloc[0]) if len(country_counts) else 0
        n = len(double)
        rows.append({
            "analysis_group": row.outcome_type,
            "replicon": row.replicon,
            "outcome": outcome_label,
            "odds_ratio": row.odds_ratio,
            "q_value": row.q_value,
            "double_positive_n": n,
            "double_positive_ST_count": int(st_counts.size),
            "largest_ST": st_counts.index[0] if len(st_counts) else "none",
            "largest_ST_double_positive_n": st_max,
            "largest_ST_double_positive_fraction": st_max / n if n else np.nan,
            "lineage_dominated": bool(n and st_max / n > 0.70),
            "double_positive_country_count": int(country_counts.size),
            "largest_country": country_counts.index[0] if len(country_counts) else "none",
            "largest_country_double_positive_n": country_max,
            "largest_country_double_positive_fraction": country_max / n if n else np.nan,
            "geography_dominated": bool(n and country_max / n > 0.70),
        })
    out = pd.DataFrame(rows)
    out.to_csv(outdir / "lineage_geography_sensitivity_for_top_hits.csv", index=False)
    return out


def detect_unstable_ci(orv: float, lo: float, hi: float) -> str:
    if not all(np.isfinite(x) for x in [orv, lo, hi]):
        return "non-finite estimate or confidence interval"
    if lo <= 0 or hi / lo > 25:
        return "unstable confidence interval"
    return "ok"


def adjusted_models(data: pd.DataFrame, stable_status: pd.DataFrame, outdir: Path, min_group_n: int) -> pd.DataFrame:
    rows = []
    if stable_status.empty:
        pd.DataFrame(rows).to_csv(outdir / "adjusted_logistic_models_if_stable.csv", index=False)
        return pd.DataFrame(rows)
    model_data = data.copy()
    model_data["ST_grouped"] = collapse_rare(model_data["ST"], min_group_n)
    model_data["Country_grouped"] = collapse_rare(model_data["Country"], min_group_n)
    model_data["Source_category_grouped"] = collapse_rare(model_data["Source category"], min_group_n)
    model_data["Collection_year_numeric"] = pd.to_numeric(model_data["Collection year"], errors="coerce")
    model_data = model_data.dropna(subset=["Collection_year_numeric"]).copy()
    model_data["Collection_year_centered"] = model_data["Collection_year_numeric"] - model_data["Collection_year_numeric"].median()

    candidates = stable_status[stable_status["outcome"].isin(["ESBL_positive", "carbapenemase_positive"]) & stable_status["stable_association"]].copy()
    for hit in candidates.itertuples(index=False):
        outcome = hit.outcome
        rep_col = f"replicon::{hit.replicon}"
        d = model_data[[outcome, rep_col, "ST_grouped", "Country_grouped", "Collection_year_centered", "Source_category_grouped"]].copy()
        d[outcome] = d[outcome].astype(int)
        d[rep_col] = d[rep_col].astype(int)
        n_events = int(d[outcome].sum())
        n_params = 1 + 1 + (d["ST_grouped"].nunique() - 1) + (d["Country_grouped"].nunique() - 1) + 1 + (d["Source_category_grouped"].nunique() - 1)
        base = {
            "replicon": hit.replicon,
            "outcome": outcome,
            "n": len(d),
            "events": n_events,
            "parameters": n_params,
        }
        if n_params <= 0 or n_events < 10 * n_params:
            rows.append({**base, "model_fit": "not_fit", "reason": f"fewer than 10 outcome events per model parameter ({n_events}/{n_params})"})
            continue
        formula = f'Q("{outcome}") ~ Q("{rep_col}") + C(ST_grouped) + C(Country_grouped) + Collection_year_centered + C(Source_category_grouped)'
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = smf.glm(formula, data=d, family=sm.families.Binomial()).fit(maxiter=250)
            warning_names = {w.category.__name__ for w in caught}
            term = f'Q("{rep_col}")'
            if term not in model.params.index:
                rows.append({**base, "model_fit": "not_reported", "reason": "replicon term missing from fitted model"})
                continue
            beta = model.params[term]
            lo, hi = model.conf_int().loc[term]
            orv, ci_lo, ci_hi = float(np.exp(beta)), float(np.exp(lo)), float(np.exp(hi))
            ci_status = detect_unstable_ci(orv, ci_lo, ci_hi)
            if (not getattr(model, "converged", True)) or ci_status != "ok" or "PerfectSeparationWarning" in warning_names or "ConvergenceWarning" in warning_names:
                rows.append({
                    **base,
                    "model_fit": "fit_but_not_reported",
                    "reason": "; ".join(sorted(warning_names)) or ci_status,
                    "adjusted_odds_ratio": orv,
                    "ci_low": ci_lo,
                    "ci_high": ci_hi,
                    "p_value": float(model.pvalues[term]),
                })
                continue
            rows.append({
                **base,
                "model_fit": "reported",
                "reason": "ok",
                "adjusted_odds_ratio": orv,
                "ci_low": ci_lo,
                "ci_high": ci_hi,
                "p_value": float(model.pvalues[term]),
                "warnings": ";".join(sorted(warning_names)),
            })
        except Exception as exc:
            rows.append({**base, "model_fit": "not_fit", "reason": str(exc)})
    out = pd.DataFrame(rows)
    out.to_csv(outdir / "adjusted_logistic_models_if_stable.csv", index=False)
    return out


def heatmap(df: pd.DataFrame, outdir: Path, filename: str, title: str) -> None:
    stable = df[df["stable_association"]].copy()
    if stable.empty:
        return
    stable["log2_or"] = np.log2(stable["odds_ratio"].replace(0, np.nan))
    matrix = stable.pivot_table(index="replicon", columns="outcome", values="log2_or", aggfunc="max")
    matrix = matrix.loc[matrix.abs().max(axis=1).sort_values(ascending=False).index]
    fig_h = max(4, 0.28 * len(matrix) + 1.8)
    fig_w = max(6, 0.65 * len(matrix.columns) + 4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    arr = matrix.to_numpy(dtype=float)
    vmax = np.nanmax(np.abs(arr)) if np.isfinite(arr).any() else 1
    im = ax.imshow(arr, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(matrix.shape[1]), matrix.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(matrix.shape[0]), matrix.index)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(arr[i, j]):
                ax.text(j, i, f"{arr[i,j]:.1f}", ha="center", va="center", fontsize=8)
    ax.set_title(title, loc="left", fontweight="bold")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log2 odds ratio")
    fig.tight_layout()
    fig.savefig(outdir / f"{filename}.svg")
    fig.savefig(outdir / f"{filename}.png", dpi=300)
    plt.close(fig)


def prevalence_bar(data: pd.DataFrame, replicons: list[str], outcome: str, outdir: Path, filename: str, title: str) -> None:
    rows = []
    for rep in replicons:
        mask = data[f"replicon::{rep}"].astype(bool)
        n = int(mask.sum())
        if n < 30:
            continue
        rows.append({"replicon": rep, "n": n, "prevalence": float(data.loc[mask, outcome].mean())})
    plot = pd.DataFrame(rows).sort_values(["prevalence", "n"], ascending=[False, False]).head(25)
    if plot.empty:
        return
    fig, ax = plt.subplots(figsize=(10, max(4, 0.32 * len(plot) + 1.4)))
    y = np.arange(len(plot))
    ax.barh(y, plot["prevalence"] * 100, color="#5E81AC")
    ax.set_yticks(y, plot["replicon"])
    ax.invert_yaxis()
    ax.set_xlabel(f"{outcome.replace('_', ' ')} prevalence among replicon-positive genomes (%)")
    ax.set_title(title, loc="left", fontweight="bold")
    for yi, row in enumerate(plot.itertuples(index=False)):
        ax.text(row.prevalence * 100 + 0.8, yi, f"{row.prevalence*100:.1f}% (n={row.n})", va="center", fontsize=8)
    ax.set_xlim(0, min(105, max(10, plot["prevalence"].max() * 100 + 18)))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(outdir / f"{filename}.svg")
    fig.savefig(outdir / f"{filename}.png", dpi=300)
    plt.close(fig)


def write_summary(
    data: pd.DataFrame,
    status: pd.DataFrame,
    genes: pd.DataFrame,
    all_genes: pd.DataFrame,
    sensitivity: pd.DataFrame,
    adjusted: pd.DataFrame,
    outdir: Path,
    duplicate_n: int,
) -> None:
    def top_lines(df: pd.DataFrame, outcome: str, n: int = 5) -> list[str]:
        sub = df[(df["outcome"].eq(outcome)) & (df["stable_association"])].sort_values(["q_value", "odds_ratio"], ascending=[True, False]).head(n)
        lines = []
        for row in sub.itertuples(index=False):
            lines.append(f"- `{row.replicon}`: OR {row.odds_ratio:.2f} (95% CI {row.ci_low:.2f}-{row.ci_high:.2f}), {fmt_q(row.q_value)}; prevalence {row.outcome_prevalence_replicon_positive*100:.1f}% with replicon versus {row.outcome_prevalence_replicon_negative*100:.1f}% without.")
        return lines or ["- No stable associations passed all prespecified filters."]

    stable_genes = genes[genes["stable_association"]].sort_values(["q_value", "odds_ratio"], ascending=[True, False]).head(12)
    gene_lines = []
    for row in stable_genes.itertuples(index=False):
        gene_lines.append(f"- `{row.replicon}` with `{row.outcome}`: OR {row.odds_ratio:.2f} (95% CI {row.ci_low:.2f}-{row.ci_high:.2f}), {fmt_q(row.q_value)}.")
    if not gene_lines:
        gene_lines = ["- No individual ESBL/carbapenemase gene associations passed all prespecified filters."]

    lineage_lines = []
    if not sensitivity.empty:
        for row in sensitivity.sort_values(["lineage_dominated", "geography_dominated", "q_value"], ascending=[False, False, True]).head(10).itertuples(index=False):
            flags = []
            if row.lineage_dominated:
                flags.append(f"lineage-dominated by {row.largest_ST} ({row.largest_ST_double_positive_fraction*100:.1f}%)")
            if row.geography_dominated:
                flags.append(f"geography-dominated by {row.largest_country} ({row.largest_country_double_positive_fraction*100:.1f}%)")
            if flags:
                lineage_lines.append(f"- `{row.replicon}` / `{row.outcome}`: {', '.join(flags)} among {row.double_positive_n} double-positive genomes.")
    if not lineage_lines:
        lineage_lines = ["- No top stable association exceeded the 70% lineage- or geography-dominance threshold, or no stable associations were available for this sensitivity step."]

    if adjusted.empty:
        adj_text = "No adjusted logistic models were reportable."
    else:
        reported = adjusted[adjusted.get("model_fit", pd.Series(dtype=str)).eq("reported")]
        if reported.empty:
            reasons = adjusted["reason"].value_counts().head(5).to_dict() if "reason" in adjusted else {}
            adj_text = "Adjusted logistic models were attempted for stable ESBL/carbapenemase status hits, but no estimates met the prespecified reporting criteria. Main reasons: " + "; ".join(f"{k} ({v})" for k, v in reasons.items()) + "."
        else:
            adj_text = "Adjusted logistic models produced reportable estimates for: " + "; ".join(f"{r.replicon}/{r.outcome} adjusted OR {r.adjusted_odds_ratio:.2f} (95% CI {r.ci_low:.2f}-{r.ci_high:.2f})" for r in reported.itertuples(index=False)) + "."

    text = f"""# Genome-level PlasmidFinder replicon-AMR associations in curated *Klebsiella variicola*

## Scope and caveat

This analysis tested genome-level co-occurrence between PlasmidFinder replicons and AMR genes/statuses in `{len(data):,}` curated *K. variicola* genomes. Duplicate `Genome ID` records detected before de-duplication: `{duplicate_n}`. These are association analyses only: a replicon and AMR gene detected in the same genome does not prove that the AMR gene is located on that plasmid unless contig-level AMR/plasmid localisation data are available.

## Stable status-level associations

Stable associations required replicon and outcome prevalence thresholds, all four contingency-table cells >=5, finite odds ratio and 95% CI, CI ratio <=25, and Benjamini-Hochberg q <0.05.

### ESBL positivity
{chr(10).join(top_lines(status, 'ESBL_positive'))}

### Carbapenemase positivity
{chr(10).join(top_lines(status, 'carbapenemase_positive'))}

### ESBL or carbapenemase positivity
{chr(10).join(top_lines(status, 'ESBL_or_carbapenemase_positive'))}

## Individual ESBL/carbapenemase gene signals

{chr(10).join(gene_lines)}

These gene-level signals identify which beta-lactamase families drive genome-level co-occurrence. Interpret them as co-detection patterns, not plasmid localisation.

## Lineage and geography sensitivity

{chr(10).join(lineage_lines)}

## Adjusted models

{adj_text}

Because ST, country, collection year and source category are highly structured in this genome collection, adjusted models were treated as sensitivity analyses. The Fisher/stability-filtered genome-level co-occurrence results are the primary analysis.

## Output files

- `replicon_summary.csv`
- `esbl_carbapenemase_status_summary.csv`
- `replicon_vs_status_all_results.csv`
- `replicon_vs_status_stable_results.csv`
- `replicon_vs_esbl_carb_genes_all_results.csv`
- `replicon_vs_esbl_carb_genes_stable_results.csv`
- `lineage_geography_sensitivity_for_top_hits.csv`
- `adjusted_logistic_models_if_stable.csv`
- exploratory figures in SVG and PNG format
"""
    (outdir / "plasmid_amr_association_summary.md").write_text(text)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update({
        "font.family": "Ubuntu",
        "font.sans-serif": ["Ubuntu"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })

    raw = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    duplicate_n = int(raw.duplicated(GENOME_ID, keep=False).sum())
    data, replicons, target_genes, all_amr_genes = load_and_prepare(args.input, args.outdir)
    tested_replicons = [r for r in replicons if int(data[f"replicon::{r}"].sum()) >= args.min_replicon_n]
    summarise_inputs(data, tested_replicons, args.outdir)

    status_outcomes = [(outcome, "status") for outcome in STATUS_OUTCOMES]
    status_results = run_association_set(data, tested_replicons, status_outcomes, q_group_cols=None)
    status_results.to_csv(args.outdir / "replicon_vs_status_all_results.csv", index=False)
    status_stable = status_results[status_results["stable_association"]].copy()
    status_stable.to_csv(args.outdir / "replicon_vs_status_stable_results.csv", index=False)

    target_outcomes = [(f"target_gene::{gene}", "ESBL/carbapenemase_gene") for gene in target_genes]
    target_internal = run_association_set(data, tested_replicons, target_outcomes, q_group_cols=None)
    target_results = target_internal.copy()
    target_results["outcome"] = target_results["outcome"].str.replace("target_gene::", "", regex=False)
    target_results.to_csv(args.outdir / "replicon_vs_esbl_carb_genes_all_results.csv", index=False)
    target_stable = target_results[target_results["stable_association"]].copy()
    target_stable.to_csv(args.outdir / "replicon_vs_esbl_carb_genes_stable_results.csv", index=False)

    amr_outcomes = [(f"amr_gene::{gene}", "all_acquired_amr_gene") for gene in all_amr_genes]
    all_amr_results = run_association_set(data, tested_replicons, amr_outcomes, q_group_cols=None)
    all_amr_results["outcome"] = all_amr_results["outcome"].str.replace("amr_gene::", "", regex=False)
    all_amr_results.to_csv(args.outdir / "replicon_vs_all_amr_genes_all_results.csv", index=False)
    all_amr_results[all_amr_results["stable_association"]].to_csv(args.outdir / "replicon_vs_all_amr_genes_stable_results.csv", index=False)

    sensitivity_input = pd.concat([status_results, target_internal], ignore_index=True)
    sensitivity_input["outcome_column"] = sensitivity_input["outcome"]
    sensitivity = lineage_sensitivity(data, sensitivity_input, args.outdir, args.top_n_lineage)

    adjusted = adjusted_models(data, status_results, args.outdir, args.group_min_n)

    heatmap(status_results, args.outdir, "stable_replicon_status_log2or_heatmap", "Stable replicon-status associations")
    heatmap(target_results, args.outdir, "stable_replicon_esbl_carb_gene_log2or_heatmap", "Stable replicon-gene associations")
    prevalence_bar(data, tested_replicons, "ESBL_positive", args.outdir, "esbl_prevalence_by_common_replicon", "ESBL prevalence by common replicon")
    prevalence_bar(data, tested_replicons, "carbapenemase_positive", args.outdir, "carbapenemase_prevalence_by_common_replicon", "Carbapenemase prevalence by common replicon")

    write_summary(data, status_results, target_results, all_amr_results, sensitivity, adjusted, args.outdir, duplicate_n)
    print(f"Wrote plasmid-AMR association outputs to {args.outdir}")
    print(f"Run command: conda run -n st273_phylogeny_env python scripts/analyse_plasmid_amr_associations.py")


if __name__ == "__main__":
    main()
