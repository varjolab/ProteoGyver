#!/usr/bin/env python3
# Read "batch settings.toml" via tomlkit and run the headless pipeline.

import os, sys, json
from dataclasses import asdict

from tomlkit import loads as toml_loads
from tomlkit.items import Array  # for type checks

from pipeline_batch import BatchConfig, run_pipeline


def _unwrap(x):
    """Convert tomlkit items into plain Python types (recursively)."""
    # Most tomlkit items expose .value; Arrays need element-wise unwrap.
    if hasattr(x, "value"):
        return x.value
    if isinstance(x, Array):
        return [_unwrap(i) for i in x]
    if isinstance(x, list):  # just in case we already got plain list
        return [_unwrap(i) for i in x]
    # Tables behave like dicts already
    if isinstance(x, dict):
        return {k: _unwrap(v) for k, v in x.items()}
    return x


def _get(doc: dict, path: list[str], default=None):
    cur = doc
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return _unwrap(cur) if cur is not None else default


def _resolve(base: str, p):
    if p is None:
        return None
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))


def _load_config(toml_path: str) -> BatchConfig:
    with open(toml_path, "r", encoding="utf-8") as f:
        doc = toml_loads(f.read())

    base = os.path.dirname(os.path.abspath(toml_path))

    # Sections as dicts (still tomlkit items inside)
    gen  = _unwrap(doc.get("general", {}))
    prot = _unwrap(doc.get("proteomics", {}))
    inta = _unwrap(doc.get("interactomics", {}))

    # Mandatory
    workflow = gen["workflow"]  # "proteomics" | "interactomics"
    data     = _resolve(base, gen["data"])
    samples  = _resolve(base, gen["samples"])

    return BatchConfig(
        # --- general ---
        workflow=workflow,
        data_table_path=data,
        sample_table_path=samples,
        outdir=_resolve(base, gen.get("outdir", "batch_out")) or "batch_out",
        figure_template=gen.get("figure_template", "plotly_white"),
        remove_common_contaminants=bool(gen.get("remove_common_contaminants", True)),
        rename_replicates=bool(gen.get("rename_replicates", False)),
        unique_only=bool(gen.get("unique_only", False)),
        force_supervenn=bool(gen.get("force_supervenn", False)),

        # --- proteomics ---
        na_filter_percent=int(prot.get("na_filter_percent", 70)),
        na_filter_type=prot.get("na_filter_type", "sample-group"),
        normalization=prot.get("normalization", "no_normalization"),
        imputation=prot.get("imputation", "QRILC"),
        control_group=prot.get("control_group"),
        comparison_file=_resolve(base, prot.get("comparison_file")),
        fc_threshold=float(prot.get("fc_threshold", 1.5)),
        p_threshold=float(prot.get("p_threshold", 0.05)),
        test_type=prot.get("test_type", "independent"),

        # --- interactomics ---
        uploaded_controls=list(inta.get("uploaded_controls", [])),
        additional_controls=list(inta.get("additional_controls", [])),
        crapome_sets=list(inta.get("crapome_sets", [])),
        proximity_filtering=bool(inta.get("proximity_filtering", False)),
        n_controls=int(inta.get("n_controls", 3)),
        saint_bfdr_threshold=float(inta.get("saint_bfdr_threshold", 0.05)),
        crapome_percentage_threshold=int(inta.get("crapome_percentage_threshold", 20)),
        crapome_fc_threshold=int(inta.get("crapome_fc_threshold", 2)),
        rescue_enabled=bool(inta.get("rescue_enabled", False)),
        chosen_enrichments=list(inta.get("chosen_enrichments", [])),
    )


def main():
    # Default to your preferred filename; allow overriding via argv[1]
    settings_path = sys.argv[1] if len(sys.argv) > 1 else "batch settings.toml"
    if not os.path.exists(settings_path):
        sys.stderr.write(f"Settings file not found: {settings_path}\n")
        sys.exit(2)

    cfg = _load_config(settings_path)
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
