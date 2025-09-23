#!/usr/bin/env python3
# Run your Dash analysis headlessly, step-by-step.

import os, json, base64, time
from io import StringIO
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# --- import the same modules used by your callbacks ---
from components import parsing, qc_analysis, proteomics, interactomics, db_functions
from components.figures.color_tools import get_assigned_colors

# -------- Config --------
@dataclass
class BatchConfig:
    data_table_path: str                 # e.g. "data/your_maxquant_proteingroups.tsv"
    sample_table_path: str               # e.g. "data/experimental_design.tsv"
    outdir: str = "batch_out"            # where to write JSON artifacts
    figure_template: str = "plotly_white"
    remove_common_contaminants: bool = True
    rename_replicates: bool = False
    unique_only: bool = False
    workflow: str = "proteomics"
    # Proteomics knobs
    na_filter_percent: int = 70
    na_filter_type: str = "sample-group"        # "sample-group" | "sample-set"
    normalization: str = "no_normalization"        # "Median" | "Quantile" | "Vsn" | "no_normalization"
    imputation: str = "QRILC"              # "knn" | "mean" | ...
    control_group: Optional[str] = None  # If None, provide comparison_file instead
    comparison_file: Optional[str] = None
    fc_threshold: float = 2
    p_threshold: float = 0.05
    test_type: str = "independent"
    # Interactomics knobs
    uploaded_controls: List[str] = field(default_factory=list)
    additional_controls: List[str] = field(default_factory=list)
    crapome_sets: List[str] = field(default_factory=list)
    proximity_filtering: bool = False
    n_controls: int = 3
    saint_bfdr_threshold: float = 0.05
    crapome_percentage_threshold: int = 20
    crapome_fc_threshold: int = 2
    rescue_enabled: bool = False
    chosen_enrichments: List[str] = field(default_factory=list)

# -------- Helpers that mimic Dash's upload content --------
def _upload_contents_for_path(path: str) -> Tuple[str, str, int]:
    """Return (contents_str, filename, last_modified_int) like dash Upload component."""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    # Dash passes strings like "data:application/octet-stream;base64,AAAA..."
    contents = f"data:application/octet-stream;base64,{b64}"
    filename = os.path.basename(path)
    last_mod = int(os.path.getmtime(path) * 1000)
    return contents, filename, last_mod

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _dump_json(outdir: str, name: str, obj: Any):
    _ensure_dir(outdir)
    with open(os.path.join(outdir, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

# -------- Pipeline --------
def run_pipeline(cfg: BatchConfig) -> Dict[str, Any]:
    # 1) Load parameters & db/contaminants (mirrors QC_and_data_analysis.py)
    params = parsing.parse_parameters("parameters.toml")  # same as app
    db_file = os.path.join(*params["Data paths"]["Database file"])
    contaminant_list = db_functions.get_contaminants(db_file)

    # 2) “Upload” data & sample tables from disk (use same parsing functions the app uses)
    data_contents, data_name, data_mtime = _upload_contents_for_path(cfg.data_table_path)
    sample_contents, sample_name, sample_mtime = _upload_contents_for_path(cfg.sample_table_path)

    # App gives a style dict and file-loading config; we can pass a dummy style and the real file settings
    dummy_style = {}
    file_loading_cfg = params["file loading"]

    # Same functions the callbacks use:
    # parsing.parse_data_file -> returns (upload_style, data_info, data_tables)
    _, data_info, data_tables = parsing.parse_data_file(
        data_contents, data_name, data_mtime, dummy_style, file_loading_cfg
    )
    _, expdes_info, expdes_table = parsing.parse_sample_table(
        sample_contents, sample_name, sample_mtime, dummy_style
    )

    # 3) Format the data (mirrors `validate_data` callback)
    # also set your figure template
    import plotly.io as pio
    pio.templates.default = cfg.figure_template

    session_name = f'{time.strftime("%Y-%m-%d-%H-%M-%S")}--batch'
    data_dictionary = parsing.format_data(
        session_name,
        data_tables,
        data_info,
        expdes_table,
        expdes_info,
        contaminant_list if cfg.remove_common_contaminants else [],
        cfg.rename_replicates,
        cfg.unique_only,
        params["workflow parameters"]["interactomics"]["control indicators"],
        params["file loading"]["Bait ID column names"],
    )

    _dump_json(cfg.outdir, "01_data_dictionary", data_dictionary)

    # 4) Assign replicate colors (mirrors assign_replicate_colors)
    rep_colors, rep_colors_with_cont = get_assigned_colors(
        data_dictionary["sample groups"]["norm"]
    )
    _dump_json(cfg.outdir, "02_replicate_colors", rep_colors)
    _dump_json(cfg.outdir, "02_replicate_colors_with_cont", rep_colors_with_cont)

    # 5) QC chain (call the same functions used in callbacks)
    artifacts: Dict[str, Any] = {}

    # TIC
    _, tic_data = qc_analysis.parse_tic_data(
        data_dictionary["data tables"]["experimental design"],
        rep_colors,
        db_file,
        params["Figure defaults"]["full-height"],
    )
    artifacts["tic"] = tic_data

    # Counts
    table_to_use = data_dictionary["data tables"]["table to use"]
    _, count_data = qc_analysis.count_plot(
        data_dictionary["data tables"]["with-contaminants"][table_to_use],
        rep_colors_with_cont,
        contaminant_list,
        params["Figure defaults"]["full-height"],
    )
    artifacts["counts"] = count_data

    # Common proteins
    _, common_data = qc_analysis.common_proteins(
        data_dictionary["data tables"][table_to_use],
        db_file,
        params["Figure defaults"]["full-height"],
        additional_groups={"Other contaminants": contaminant_list},
        id_str="qc",
    )
    artifacts["common_proteins"] = common_data

    # Coverage
    _, coverage_data = qc_analysis.coverage_plot(
        data_dictionary["data tables"][table_to_use],
        params["Figure defaults"]["half-height"],
    )
    artifacts["coverage"] = coverage_data

    # Reproducibility
    _, repro_data = qc_analysis.reproducibility_plot(
        data_dictionary["data tables"][table_to_use],
        data_dictionary["sample groups"]["norm"],
        table_to_use,
        params["Figure defaults"]["full-height"],
    )
    artifacts["reproducibility"] = repro_data

    # Missing
    _, missing_data = qc_analysis.missing_plot(
        data_dictionary["data tables"][table_to_use],
        rep_colors,
        params["Figure defaults"]["half-height"],
    )
    artifacts["missing"] = missing_data

    # Sum
    _, sum_data = qc_analysis.sum_plot(
        data_dictionary["data tables"][table_to_use],
        rep_colors,
        params["Figure defaults"]["half-height"],
    )
    artifacts["sum"] = sum_data

    # Mean
    _, mean_data = qc_analysis.mean_plot(
        data_dictionary["data tables"][table_to_use],
        rep_colors,
        params["Figure defaults"]["half-height"],
    )
    artifacts["mean"] = mean_data

    # Distribution
    title = parsing.get_distribution_title(table_to_use)
    _, dist_data = qc_analysis.distribution_plot(
        data_dictionary["data tables"][table_to_use],
        rep_colors,
        data_dictionary["sample groups"]["rev"],
        params["Figure defaults"]["full-height"],
        title,
    )
    artifacts["distribution"] = dist_data

    _dump_json(cfg.outdir, "03_qc_artifacts", artifacts)

    # 6) Workflow-specific analysis
    if cfg.workflow.lower() == "proteomics":
        return _run_proteomics_workflow(cfg, data_dictionary, rep_colors, params, artifacts)
    elif cfg.workflow.lower() == "interactomics":
        return _run_interactomics_workflow(cfg, data_dictionary, rep_colors, params, artifacts)
    else:
        raise ValueError(f"Unknown workflow: {cfg.workflow}")


def _run_proteomics_workflow(cfg: BatchConfig, data_dictionary: Dict[str, Any], 
                            rep_colors: Dict[str, Any], params: Dict[str, Any], 
                            artifacts: Dict[str, Any]) -> Dict[str, Any]:
    """Run the proteomics analysis workflow."""
    # NA filter
    _, na_filtered = proteomics.na_filter(
        data_dictionary,
        cfg.na_filter_percent,
        params["Figure defaults"]["full-height"],
        filter_type=cfg.na_filter_type,
    )
    _dump_json(cfg.outdir, "10_na_filtered", na_filtered)

    # Normalization
    norm_res = proteomics.normalization(
        na_filtered, cfg.normalization,
        params["Figure defaults"]["full-height"],
        params["Config"]["R error file"],
    )
    
    _, normalized = norm_res
    _dump_json(cfg.outdir, "11_normalized", normalized)
    # Imputation
    if normalized is not None:
        _, imputed = proteomics.imputation(
            normalized, cfg.imputation,
            params["Figure defaults"]["full-height"],
            params["Config"]["R error file"],
        )
        _dump_json(cfg.outdir, "12_imputed", imputed)
    else:
        imputed = None

    # PCA (optional)
    if imputed is not None:
        _, pca_data = proteomics.pca(
            imputed,
            data_dictionary["sample groups"]["rev"],
            params["Figure defaults"]["full-height"],
            rep_colors,
        )
        _dump_json(cfg.outdir, "13_pca", pca_data)

    # Volcano (control vs comparisons)
    volcano = None
    if imputed is not None:
        sgroups = data_dictionary["sample groups"]["norm"]

        # If a comparisons file is provided, validate like the UI does
        comp_data = None
        comp_style = {"background-color": "green"}
        if cfg.comparison_file:
            comp_contents, comp_name, _ = _upload_contents_for_path(cfg.comparison_file)
            comp_style, comp_data = parsing.check_comparison_file(
                comp_contents, comp_name, sgroups, comp_style
            )

        # Build comparisons
        comparisons = parsing.parse_comparisons(
            cfg.control_group, comp_data, sgroups
        )

        # Run DA (same as callback)
        volcano_div, volcano_data = proteomics.differential_abundance(
            imputed,
            sgroups,
            comparisons,
            cfg.fc_threshold,
            cfg.p_threshold,
            params["Figure defaults"]["full-height"],
            cfg.test_type,
            os.path.join(*params["Data paths"]["Database file"]),
        )
        volcano = volcano_data
        _dump_json(cfg.outdir, "14_volcano", volcano)
    _dump_json(cfg.outdir, "00_summary", {
        "workflow": "proteomics",
        "session_name": data_dictionary["other"]["session name"],
        "artifacts": artifacts,
        "na_filtered": na_filtered,
        "normalized": (normalized is not None),
        "imputed": (imputed is not None),
        "volcano": volcano is not None,
        "outdir": cfg.outdir,
    })


def _run_interactomics_workflow(cfg: BatchConfig, data_dictionary: Dict[str, Any],
                               rep_colors: Dict[str, Any], params: Dict[str, Any],
                               artifacts: Dict[str, Any]) -> Dict[str, Any]:
    """Run the interactomics analysis workflow."""
    db_file = os.path.join(*params["Data paths"]["Database file"])
    
    # Check if we have spectral count data
    if '"No data"' in data_dictionary["data tables"]["spc"]:
        return {
            "workflow": "interactomics",
            "session_name": data_dictionary["other"]["session name"],
            "artifacts": artifacts,
            "error": "No spectral count data available for interactomics analysis",
            "outdir": cfg.outdir,
        }
    
    # 1) Generate SAINT container (prepare controls and data)
    _, saint_dict, crapome_data = interactomics.generate_saint_container(
        data_dictionary,
        cfg.uploaded_controls,
        cfg.additional_controls,
        cfg.crapome_sets,
        db_file,
        cfg.proximity_filtering,
        cfg.n_controls
    )
    _dump_json(cfg.outdir, "20_saint_dict", saint_dict)
    _dump_json(cfg.outdir, "20_crapome_data", crapome_data)
    
    # 2) Run SAINT analysis
    if not saint_dict:  # Empty dict means no data
        return {
            "workflow": "interactomics",
            "session_name": data_dictionary["other"]["session name"],
            "artifacts": artifacts,
            "error": "Insufficient data for SAINT analysis",
            "outdir": cfg.outdir,
        }
    
    session_name = data_dictionary["other"]["session name"]
    bait_uniprots = data_dictionary["other"].get("bait uniprots", {})
    
    saint_output, saint_failed = interactomics.run_saint(
        saint_dict,
        params["External tools"]["SAINT tempdir"],
        session_name,
        bait_uniprots,
        cleanup=True
    )
    
    if "SAINT failed" in saint_output:
        return {
            "workflow": "interactomics",
            "session_name": session_name,
            "artifacts": artifacts,
            "error": "SAINT analysis failed",
            "saint_failed": True,
            "outdir": cfg.outdir,
        }
    
    _dump_json(cfg.outdir, "21_saint_output_raw", saint_output)
    
    # 3) Add CRAPome data if available
    if crapome_data and crapome_data != '{"columns":[],"index":[],"data":[]}':
        saint_output = interactomics.add_crapome(saint_output, crapome_data)
        _dump_json(cfg.outdir, "22_saint_with_crapome", saint_output)
    
    # 4) Filter SAINT results
    filtered_saint = interactomics.saint_filtering(
        saint_output,
        cfg.saint_bfdr_threshold,
        cfg.crapome_percentage_threshold,
        cfg.crapome_fc_threshold,
        cfg.rescue_enabled
    )
    _dump_json(cfg.outdir, "23_saint_filtered", filtered_saint)
    
    # 5) Generate network plot
    _, network_elements, interactions = interactomics.do_network(
        filtered_saint,
        params["Figure defaults"]["full-height"]["height"]
    )
    _dump_json(cfg.outdir, "24_network_elements", network_elements)
    _dump_json(cfg.outdir, "24_interactions", interactions)
    
    # 6) PCA analysis
    _, pca_data = interactomics.pca(
        filtered_saint,
        params["Figure defaults"]["full-height"],
        rep_colors
    )
    _dump_json(cfg.outdir, "25_pca", pca_data)
    
    # 7) Enrichment analysis (if requested)
    enrichment_results = None
    enrichment_data = None
    enrichment_info = None
    if cfg.chosen_enrichments:
        _, enrichment_data, enrichment_info = interactomics.enrich(
            filtered_saint,
            cfg.chosen_enrichments,
            params["Figure defaults"]["full-height"],
            parameters_file="parameters.toml"
        )
        _dump_json(cfg.outdir, "26_enrichment_data", enrichment_data)
        _dump_json(cfg.outdir, "26_enrichment_info", enrichment_info)
    
    # 8) MS-microscopy analysis
    _, msmic_data = interactomics.do_ms_microscopy(
        filtered_saint,
        db_file,
        params["Figure defaults"]["full-height"]
    )
    _dump_json(cfg.outdir, "27_msmic_data", msmic_data)
    _dump_json(cfg.outdir, "00_summary", {
        "workflow": "interactomics",
        "session_name": session_name,
        "artifacts": artifacts,
        "saint_output": saint_output,
        "saint_failed": saint_failed,
        "filtered_saint": filtered_saint,
        "network_elements": len(network_elements) if network_elements else 0,
        "interactions": len(interactions) if interactions else 0,
        "enrichment_analysis": enrichment_data is not None,
        "msmic_analysis": msmic_data is not None,
        "outdir": cfg.outdir,
    })

# -------- CLI runner --------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Run ProteoGyver pipeline headlessly.")
    ap.add_argument("--data", required=True, help="Path to data table (e.g., proteinGroups.tsv)")
    ap.add_argument("--samples", required=True, help="Path to sample/experimental design table")
    ap.add_argument("--outdir", default="batch_out")
    ap.add_argument("--control", default=None, help="Control group (else provide --comparisons)")
    ap.add_argument("--comparisons", default=None, help="Path to comparisons file (optional)")
    ap.add_argument("--na", type=int, default=70, help="NA filter percent (default 70)")
    ap.add_argument("--na-type", default="sample-group", help="NA filter type (sample-group/sample-set)")
    ap.add_argument("--norm", default="no_normalization")
    ap.add_argument("--imp", default="QRILC")
    ap.add_argument("--fc", type=float, default=1.5)
    ap.add_argument("--p", type=float, default=0.05)
    ap.add_argument("--test", default="independent")
    ap.add_argument("--workflow", default="proteomics", choices=["proteomics", "interactomics"], 
                    help="Analysis workflow (default: proteomics)")
    # Interactomics arguments
    ap.add_argument("--uploaded-controls", nargs="*", default=[], 
                    help="List of uploaded control sample names")
    ap.add_argument("--additional-controls", nargs="*", default=[], 
                    help="List of additional control sets to use")
    ap.add_argument("--crapome-sets", nargs="*", default=[], 
                    help="List of CRAPome datasets to use")
    ap.add_argument("--proximity-filtering", action="store_true", 
                    help="Enable proximity filtering for controls")
    ap.add_argument("--n-controls", type=int, default=3, 
                    help="Number of controls to keep (default: 3)")
    ap.add_argument("--saint-bfdr", type=float, default=0.05, 
                    help="SAINT BFDR threshold (default: 0.05)")
    ap.add_argument("--crapome-pct", type=int, default=20, 
                    help="CRAPome percentage threshold (default: 20)")
    ap.add_argument("--crapome-fc", type=int, default=2, 
                    help="CRAPome fold change threshold (default: 2)")
    ap.add_argument("--rescue", action="store_true", 
                    help="Enable rescue filtering")
    ap.add_argument("--enrichments", nargs="*", default=[], 
                    help="List of enrichment analyses to perform")
    args = ap.parse_args()

    cfg = BatchConfig(
        data_table_path=args.data,
        sample_table_path=args.samples,
        outdir=args.outdir,
        workflow=args.workflow,
        control_group=args.control,
        comparison_file=args.comparisons,
        na_filter_percent=args.na,
        na_filter_type=args.na_type,
        normalization=args.norm,
        imputation=args.imp,
        fc_threshold=args.fc,
        p_threshold=args.p,
        test_type=args.test,
        uploaded_controls=args.uploaded_controls,
        additional_controls=args.additional_controls,
        crapome_sets=args.crapome_sets,
        proximity_filtering=args.proximity_filtering,
        n_controls=args.n_controls,
        saint_bfdr_threshold=args.saint_bfdr,
        crapome_percentage_threshold=args.crapome_pct,
        crapome_fc_threshold=args.crapome_fc,
        rescue_enabled=args.rescue,
        chosen_enrichments=args.enrichments,
    )
    run_pipeline(cfg)
