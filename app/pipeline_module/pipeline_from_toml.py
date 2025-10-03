#!/usr/bin/env python3
# Read "batch settings.toml" via tomlkit and run the headless pipeline.

import os, sys, json
from dataclasses import asdict
from components.tools import utils
from tomlkit import loads as toml_loads
from tomlkit.items import Array  # for type checks
from pipeline_module import pipeline_batch
from pathlib import Path
from typing import Any

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

def load_pipeline_parameters(
    user_toml: Path,
    defaults_dir: Path,
) -> dict[str, Any]:
    """
    Build final parameters with precedence:
      common_default  <- workflow_default  <- user_toml
    """

    common = utils.load_toml(defaults_dir / "common.toml")
    user = utils.load_toml(user_toml)

    # workflow can be "interactomics" or "proteomics" (case-insensitive)
    try:
        workflow = str(user["general"]["workflow"]).strip().lower()
    except KeyError:
        raise KeyError(
            "User TOML must define top-level key 'workflow' (e.g. 'interactomics' or 'proteomics')."
        )

    if workflow in ("interactomics", "interactome"):
        defname = 'interactomics'
    elif workflow in ("proteomics", "proteome"):
        defname = 'proteomics'
    else:
        raise ValueError(
            f"Unsupported workflow '{workflow}'. Expected 'interactomics' or 'proteomics'."
        )
    wf_defaults = utils.load_toml(defaults_dir / f"{defname}.toml")

    # Merge order: common <- wf_defaults <- user
    final_params = utils.deep_merge(common, wf_defaults)
    final_params = utils.deep_merge(final_params, user)

    return final_params

def load_config(toml_path: str, default_toml_dir: Path | None = None) -> pipeline_batch.BatchConfig:
    if default_toml_dir:
        new_toml_path = f'{toml_path}_full_parameters.toml'
        params = load_pipeline_parameters(Path(toml_path), default_toml_dir)
        utils.save_toml(params, Path(new_toml_path))
        toml_path = new_toml_path
    with open(toml_path, "r", encoding="utf-8") as f:
        doc = toml_loads(f.read())

    base = os.path.dirname(os.path.abspath(toml_path))

    # Sections as dicts (still tomlkit items inside)
    gen  = _unwrap(doc.get("general", {}))
    pipeline = _unwrap(doc.get("pipeline", {}))
    prot = _unwrap(doc.get("proteomics", {}))
    inta = _unwrap(doc.get("interactomics", {}))

    # Mandatory
    workflow = gen["workflow"]  # "proteomics" | "interactomics"
    data     = _resolve(base, gen["data"])
    samples  = _resolve(base, gen["sample table"])

    # Helper for empty-string-as-None
    def _none_if_empty(s):
        return None if (s is None or (isinstance(s, str) and s.strip() == "")) else s

    return pipeline_batch.BatchConfig(
        # --- pipeline ---
        plot_formats=list(pipeline.get("plot_formats", ["png", "html", "pdf"])),
        keep_batch_output=bool(pipeline.get("keep_batch_output", False)),

        # --- general ---
        workflow=workflow,
        data_table_path=data,
        sample_table_path=samples,
        outdir=os.path.join(base, "pipeline_temp_files"),
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
        control_group=_none_if_empty(prot.get("control_group")),
        comparison_file=_resolve(base, _none_if_empty(prot.get("comparison_file"))),
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