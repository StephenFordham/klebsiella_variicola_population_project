"""Prepare revised inputs using a common Kleborate ESBL-genotype definition.

Run with st273_phylogeny_env. Source CSVs are never overwritten.
ESBL_positive is the primary outcome; ESBL_standard_field_positive is retained
as the narrower sensitivity outcome. Original gene calls remain unchanged.
"""

from pathlib import Path
import hashlib
import json

import pandas as pd


ESBL_COLUMNS = ("Bla_ESBL_acquired", "Bla_ESBL_inhR_acquired")
ABSENT = {"", "-", "0", "na", "n/a", "nan", "none", "null"}
METHOD = (
    "ESBL-genotype positivity was defined as at least one reported acquired "
    "resistance determinant in either the Kleborate Bla_ESBL_acquired or "
    "Bla_ESBL_inhR_acquired field. Genomes with hits in both fields were counted "
    "once. The same definition was applied to both species. This classification "
    "represented gene carriage rather than phenotypically confirmed ESBL production."
)


def parse_hits(value):
    if pd.isna(value):
        return set()
    return {
        gene.strip() for gene in str(value).split(";")
        if gene.strip().lower() not in ABSENT
    }


def annotate_esbl(data):
    """Return a copy with primary/sensitivity status and unnormalised gene union."""
    standard = data[ESBL_COLUMNS[0]].map(parse_hits)
    inhibitor = data[ESBL_COLUMNS[1]].map(parse_hits)
    out = data.copy()
    out["ESBL_standard_field_positive"] = standard.map(bool)
    out["ESBL_inhibitor_resistant_field_positive"] = inhibitor.map(bool)
    out["ESBL_positive"] = (
        out["ESBL_standard_field_positive"]
        | out["ESBL_inhibitor_resistant_field_positive"]
    )
    out["ESBL_genes_combined"] = [
        "; ".join(sorted(a | b)) if a | b else "-"
        for a, b in zip(standard, inhibitor)
    ]
    return out


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parent
    project = root.parent.parent
    inputs = [
        ("Klebsiella variicola", project / "Klebsiella_variicola_curated_population.csv",
         "Klebsiella_variicola_analysis_ready.csv"),
        ("Klebsiella pneumoniae", root / "klebsiella_pneumoniae_comparator_dataset.csv",
         "Klebsiella_pneumoniae_analysis_ready.csv"),
    ]
    outdir = root / "analysis_inputs"
    outdir.mkdir(exist_ok=True)
    summaries, changes, provenance = [], [], []
    for species, source, output_name in inputs:
        before_hash = file_hash(source)
        data = pd.read_csv(source, dtype=str, keep_default_na=False)
        if data["Genome ID"].duplicated().any() or data["Genome ID"].str.strip().eq("").any():
            raise ValueError(f"Genome identifiers must be present and unique: {source}")
        out = annotate_esbl(data)
        changed = out.ESBL_positive & ~out.ESBL_standard_field_positive
        delta = out.loc[changed, [
            "Genome ID", "Genome Name", "ST", "Country", "Source category",
            *ESBL_COLUMNS, "ESBL_standard_field_positive", "ESBL_positive",
        ]].copy()
        delta.insert(0, "species", species)
        changes.append(delta)
        summaries.append({
            "species": species,
            "n_genomes": len(out),
            "ESBL_standard_field_n": int(out.ESBL_standard_field_positive.sum()),
            "ESBL_inhibitor_resistant_field_n": int(out.ESBL_inhibitor_resistant_field_positive.sum()),
            "ESBL_positive_n": int(out.ESBL_positive.sum()),
            "ESBL_positive_percent": 100 * out.ESBL_positive.mean(),
            "additional_positive_n": int(changed.sum()),
        })
        target = outdir / output_name
        out.to_csv(target, index=False)
        check = pd.read_csv(target, dtype=str, keep_default_na=False)
        pd.testing.assert_frame_equal(check[data.columns], data)
        assert check.ESBL_positive.eq("True").equals(out.ESBL_positive)
        assert before_hash == file_hash(source)
        provenance.append({
            "input": str(source), "input_sha256": before_hash,
            "output": str(target), "output_sha256": file_hash(target),
        })

    pd.DataFrame(summaries).to_csv(root / "ESBL_definition_summary.csv", index=False)
    pd.concat(changes, ignore_index=True).to_csv(root / "ESBL_definition_changed_genomes.csv", index=False)
    (root / "ESBL_genotype_definition_methods.txt").write_text(METHOD + "\n", encoding="utf-8")
    manifest = {
        "primary_status": "ESBL_positive: at least one hit in either ESBL field",
        "sensitivity_status": "ESBL_standard_field_positive: first ESBL field only",
        "gene_union": "ESBL_genes_combined preserves allele labels and removes exact duplicates",
        "files": provenance,
        "run_command": f'conda run -n st273_phylogeny_env python "{Path(__file__).resolve()}"',
        "pending": [
            "Rerun comparisons, gene summaries and source-matched sensitivity analyses.",
            "Use the shared ESBL definition in revised within-species models and candidate scoring.",
            "Check convergence annotations and phylogenetic cluster summaries against revised status.",
            "Update affected figures, captions, tables and manuscript results after recalculation.",
        ],
        "scope": "Input preparation only; historical analysis scripts and submitted outputs are unchanged.",
    }
    (root / "ESBL_definition_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(summaries).to_string(index=False))
    print("Changed K. variicola genomes:")
    print(changes[0].to_string(index=False))


if __name__ == "__main__":
    main()
