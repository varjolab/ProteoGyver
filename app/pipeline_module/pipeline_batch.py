#!/usr/bin/env python3
# Run Dash analysis headlessly, step-by-step.

from __future__ import annotations

import os, json, base64, time
import logging
import pandas as pd
from io import StringIO
from dataclasses import dataclass, field, asdict, is_dataclass
from typing import Any, Dict, List, Optional, Tuple
import pickle
from collections.abc import Mapping
from pathlib import Path

from components import parsing, qc_analysis, proteomics, interactomics, db_functions
from components.figures import color_tools
from _version import __version__ 

def dash_to_wire(obj):
    """Recursively convert Dash/Plotly components to JSON-serializable structures.

    - Leaves primitives (str, int, float, bool, None) untouched.
    - Converts any object exposing ``to_plotly_json()`` (Dash components, go.Figure).
    - Recurses through dicts and lists/tuples.
    - Dataclasses are converted via ``asdict()`` then recursed.

    :param obj: Any Python object (Dash component, go.Figure, dict/list, primitives).
    :returns: JSON-serializable structure with components replaced by dicts/lists.
    """
    # Fast path: primitives / “don’t touch”
    if obj is None or isinstance(obj, (str, int, float, bool, bytes, bytearray, memoryview)):
        return obj

    # Numpy scalars -> built-in types (optional but handy)
    try:
        import numpy as np  # type: ignore
        if isinstance(obj, np.generic):
            return obj.item()
    except Exception:
        pass

    # Dash/Plotly components & figures expose this
    to_json = getattr(obj, "to_plotly_json", None)
    if callable(to_json):
        return dash_to_wire(to_json())

    # Dataclasses → dict, then recurse
    if is_dataclass(obj):
        return dash_to_wire(asdict(obj))

    # Mappings → dict, then recurse on values
    if isinstance(obj, Mapping):
        return {k: dash_to_wire(v) for k, v in obj.items()}

    # Sequences (lists/tuples) → list, recurse per element
    if isinstance(obj, (list, tuple)):
        return [dash_to_wire(x) for x in obj]

    # Anything else: leave as-is (you can add more coercions if needed)
    return obj



# -------- Config --------
@dataclass
class BatchConfig:

    # --- data ---
    data_table_path: str                 # e.g. "data/your_maxquant_proteingroups.tsv"
    sample_table_path: str               # e.g. "data/experimental_design.tsv"
    outdir: str = "batch_out"            # where to write JSON artifacts
    figure_template: str = "plotly_white"
    remove_common_contaminants: bool = True
    rename_replicates: bool = False
    unique_only: bool = False
    workflow: str = "proteomics"

    # --- pipeline ---
    plot_formats: List[str] = field(default_factory=lambda: ["png", "html", "pdf"])
    keep_batch_output: bool = False
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
    force_supervenn: bool = False

# -------- Helpers that mimic Dash's upload content --------
def _upload_contents_for_path(path: str) -> Tuple[str, str, int]:
    """Return tuple compatible with Dash Upload: (contents, filename, mtime_ms).

    :param path: Path to a file on disk.
    :returns: Tuple of (base64 contents string, filename, last-modified ms).
    """
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

def _collect_version_info(db_file: str) -> Dict[str, Any]:
    """Collect version information for Proteogyver, database, and external data.
    
    Mirrors the save_version_info function from QC_and_data_analysis.py.
    
    :param db_file: Path to SQLite database file.
    :returns: Dictionary mapping entity names to versions.
    """
    version_dict = {
        'Proteogyver version': __version__,
    }
    # Get database versions
    for update_type, version in db_functions.get_database_versions(db_file).items():
        version_dict[f'Database: {update_type}'] = version
    # Get external data versions
    conn = db_functions.create_connection(db_file, mode='ro')
    try:
        for dataset, version, _ in db_functions.get_full_table_as_pd(conn, 'data_versions').values:
            version_dict[dataset] = version
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f'Error getting external versions: {e}')
    finally:
        conn.close()  # type: ignore
    return version_dict

# -------- Pipeline --------
def run_pipeline(cfg: BatchConfig, params: dict) -> Dict[str, Any]:
    """Execute the batch pipeline mirroring the app's QC and analysis steps.

    :param cfg: Batch configuration object.
    :param params: Parsed application parameters.
    :returns: Summary dict and JSON artifacts written to ``cfg.outdir``.
    """
    # 1) Load parameters & db/contaminants (mirrors QC_and_data_analysis.py)
    db_file = os.path.join(*params["Data paths"]["Database file"])
    contaminant_list = db_functions.get_contaminants(db_file)
    
    # 1.5) Collect and save version information (mirrors save_version_info callback)
    version_info = _collect_version_info(db_file)
    _dump_json(cfg.outdir, "00_version_info", version_info)

    # 2) “Upload” data & sample tables from disk (use same parsing functions the app uses)
    data_contents, data_name, data_mtime = _upload_contents_for_path(cfg.data_table_path)
    sample_contents, sample_name, sample_mtime = _upload_contents_for_path(cfg.sample_table_path)

    # App gives a style dict and file-loading config; we can pass a dummy style and the real file settings
    dummy_style = {}
    file_loading_cfg = params["file loading"]

    # Same functions the callbacks use:
    # parsing.parse_data_file -> returns (upload_style, data_info, data_tables)
    _, data_info, data_tables, warnings = parsing.parse_data_file(
        data_contents, data_name, data_mtime, dummy_style, file_loading_cfg
    )
    if len(warnings) > 0:
        warnings.insert(0, 'Data table warnings')
        warnings.append('- This might be due to file format. Supported formats are: csv (comma separated); tsv, txt, tab (tab separated); xlsx, xls (excel)')
    
    _, expdes_info, expdes_table = parsing.parse_sample_table(
        sample_contents, sample_name, sample_mtime, dummy_style
    )
    exp_cols_found: list[str] = expdes_info['required columns found']
    if len(exp_cols_found) < 2:
        req_cols: list[str] = ['sample name', 'sample group']
        fcols = ', '.join([expdes_info[col] for col in req_cols if col in expdes_info])
        warnings = [
            'Sample table warnings',
            f'- Experimental design table is missing required columns. Found columns: {fcols}, required columns: {", ".join(req_cols)}.',
            '- This might be due to file format. Supported formats are: csv (comma separated); tsv, txt, tab (tab separated); xlsx, xls (excel)'
        ]
    if len(warnings) > 0:
        # Return error information to be handled by pipeline_input_watcher
        return {
            "workflow": cfg.workflow,
            "session_name": f'{time.strftime("%Y-%m-%d-%H-%M-%S")}--batch',
            "error": "Pipeline terminated due to warnings in input files",
            "warnings": warnings,
            "outdir": cfg.outdir,
        }
        
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
    data_dictionary['info'] = data_info
    data_dictionary['input_data_tables'] = data_tables
    data_dictionary['input_sample_table'] = expdes_table
    _dump_json(cfg.outdir, "01_data_dictionary", data_dictionary)

    # 4) Assign replicate colors (mirrors assign_replicate_colors)
    rep_colors, rep_colors_with_cont = color_tools.get_assigned_colors(
        data_dictionary["sample groups"]["norm"]
    )
    _dump_json(cfg.outdir, "02_replicate_colors", rep_colors)
    _dump_json(cfg.outdir, "02_replicate_colors_with_cont", rep_colors_with_cont)

    # 5) QC chain (call the same functions used in callbacks)
    artifacts: Dict[str, Any] = {}
    divs = {}
    # TIC
    tic_div, tic_data = qc_analysis.parse_tic_data(
        data_dictionary["data tables"]["experimental design"],
        rep_colors,
        db_file,
        params["Figure defaults"]["full-height"],
    )
    artifacts["tic"] = tic_data
    divs["tic"] = tic_div
    # Counts
    table_to_use = data_dictionary["data tables"]["table to use"]
    count_div, count_data = qc_analysis.count_plot(
        data_dictionary["data tables"]["with-contaminants"][table_to_use],
        rep_colors_with_cont,
        contaminant_list,
        params["Figure defaults"]["full-height"],
    )
    artifacts["counts"] = count_data
    divs["counts"] = count_div
    # Common proteins
    common_div, common_data = qc_analysis.common_proteins(
        data_dictionary["data tables"][table_to_use],
        db_file,
        params["Figure defaults"]["full-height"],
        additional_groups={"Other contaminants": contaminant_list},
        id_str="qc",
    )
    artifacts["common_proteins"] = common_data
    divs["common_proteins"] = common_div
    # Coverage
    coverage_div, coverage_data = qc_analysis.coverage_plot(
        data_dictionary["data tables"][table_to_use],
        params["Figure defaults"]["half-height"],
    )
    artifacts["coverage"] = coverage_data
    divs["coverage"] = coverage_div
    # Reproducibility
    repro_div, repro_data = qc_analysis.reproducibility_plot(
        data_dictionary["data tables"][table_to_use],
        data_dictionary["sample groups"]["norm"],
        table_to_use,
        params["Figure defaults"]["full-height"],
    )
    artifacts["reproducibility"] = repro_data
    divs["reproducibility"] = repro_div
    # Missing
    missing_div, missing_data = qc_analysis.missing_plot(
        data_dictionary["data tables"][table_to_use],
        rep_colors,
        params["Figure defaults"]["half-height"],
    )
    artifacts["missing"] = missing_data
    divs["missing"] = missing_div
    # Sum
    sum_div, sum_data = qc_analysis.sum_plot(
        data_dictionary["data tables"][table_to_use],
        rep_colors,
        params["Figure defaults"]["half-height"],
    )
    artifacts["sum"] = sum_data
    divs["sum"] = sum_div
    # Mean
    mean_div, mean_data = qc_analysis.mean_plot(
        data_dictionary["data tables"][table_to_use],
        rep_colors,
        params["Figure defaults"]["half-height"],
    )
    artifacts["mean"] = mean_data
    divs["mean"] = mean_div
    # Distribution
    title = parsing.get_distribution_title(table_to_use)
    dist_div, dist_data = qc_analysis.distribution_plot(
        data_dictionary["data tables"][table_to_use],
        rep_colors,
        data_dictionary["sample groups"]["rev"],
        params["Figure defaults"]["full-height"],
        title,
    )
    artifacts["distribution"] = dist_data
    divs["distribution"] = dist_div

    # Commonality
    commonality_div, commonality_data, pdf_str = qc_analysis.commonality_plot(
        data_dictionary['data tables'][data_dictionary['data tables']['table to use']],
        data_dictionary['sample groups']['rev'],
        params['Figure defaults']['full-height'],
        cfg.force_supervenn, 
    )
    artifacts["commonality"] = commonality_data
    artifacts["commonality_pdf"] = pdf_str
    divs["commonality"] = commonality_div
    with open(os.path.join(cfg.outdir, "03_qc_divs.pickle"), "wb") as f:
        pickle.dump(divs, f)
    _dump_json(cfg.outdir, "03_qc_artifacts", artifacts)
    # 6) Workflow-specific analysis
    if cfg.workflow.lower() == "proteomics":
        return _run_proteomics_workflow(cfg, data_dictionary, rep_colors, params, artifacts)
    elif cfg.workflow.lower() == "interactomics":
        return _run_interactomics_workflow(cfg, data_dictionary, rep_colors, rep_colors_with_cont, params, artifacts)
    else:
        raise ValueError(f"Unknown workflow: {cfg.workflow}")


def _run_proteomics_workflow(cfg: BatchConfig, data_dictionary: Dict[str, Any], 
                            rep_colors: Dict[str, Any], params: Dict[str, Any], 
                            artifacts: Dict[str, Any]) -> Dict[str, Any]:
    """Run the proteomics analysis workflow.

    :param cfg: Batch configuration.
    :param data_dictionary: Parsed/validated inputs and groups.
    :param rep_colors: Replicate color assignments.
    :param params: Parsed application parameters.
    :param artifacts: QC artifacts dict.
    :returns: Proteomics summary dict.
    """

    divs = {}
    # NA filter
    na_filter_div, na_filtered = proteomics.na_filter(
        data_dictionary,
        cfg.na_filter_percent,
        params["Figure defaults"]["full-height"],
        filter_type=cfg.na_filter_type,
    )
    _dump_json(cfg.outdir, "10_na_filtered", na_filtered)
    divs["na_filter"] = na_filter_div
    # Normalization
    normalization_div, normalized = proteomics.normalization(
        na_filtered, cfg.normalization,
        params["Figure defaults"]["full-height"],
        params["Config"]["script error file"],
    )
    _dump_json(cfg.outdir, "11_normalized", normalized)
    divs["normalization"] = normalization_div
    # Imputation
    if normalized is not None:
        missing_values_in_other_samples_div = proteomics.missing_values_in_other_samples(
            normalized,
            params["Figure defaults"]["full-height"],
        )
        divs["missing_values_in_other_samples"] = missing_values_in_other_samples_div
        imputation_div, imputed = proteomics.imputation(
            normalized, cfg.imputation,
            params["Figure defaults"]["full-height"],
            params["Config"]["script error file"],
            sample_groups_rev=data_dictionary["sample groups"]["rev"]
        )
        _dump_json(cfg.outdir, "12_imputed", imputed)
        divs["imputation"] = imputation_div
    else:
        imputed = None

    # PCA (optional)
    if imputed is not None:
        pca_div, pca_data = proteomics.pca(
            imputed,
            data_dictionary["sample groups"]["rev"],
            params["Figure defaults"]["full-height"],
            rep_colors,
        )
        _dump_json(cfg.outdir, "13_pca", pca_data)
        divs["pca"] = pca_div

        # CV analysis
        if True:
            cv_div, cv_data = proteomics.perc_cvplot(
                data_dictionary['data tables'][data_dictionary['data tables']['table to use']],
                na_filtered,
                data_dictionary["sample groups"]["norm"],
                rep_colors,
                params["Figure defaults"]["full-height"],
            )
            _dump_json(cfg.outdir, "13_cv", cv_data)
            divs["cv"] = cv_div
        # Clustermap/correlation clustering
        clustermap_div, clustermap_data = proteomics.clustermap(
            imputed,
            params["Figure defaults"]["full-height"]
        )
        _dump_json(cfg.outdir, "13_clustermap", clustermap_data)
        divs["clustermap"] = clustermap_div
        # Perturbation analysis (if we have control groups)
        # Find control groups from comparisons
        control_groups = set()
        if cfg.comparison_file:
            import pandas as pd
            comp_df = pd.read_csv(cfg.comparison_file, sep='\t')
            if 'Control' in comp_df.columns:
                control_groups.update(comp_df['Control'].unique())
    # Volcano (control vs comparisons) — optional when controls/comparisons provided
    volcano = None
    if imputed is not None:
        sgroups = data_dictionary["sample groups"]["norm"]

        # If a comparisons file is provided and valid, validate like the UI does
        comp_data = None
        comp_style = {"background-color": "green"}
        comparisons_file_path = None
        try:
            if cfg.comparison_file and isinstance(cfg.comparison_file, str) and len(cfg.comparison_file.strip()) > 0 and os.path.isfile(cfg.comparison_file):
                comparisons_file_path = cfg.comparison_file
        except Exception:
            comparisons_file_path = None

        if comparisons_file_path:
            comp_contents, comp_name, _ = _upload_contents_for_path(comparisons_file_path)
            comp_style, comp_data = parsing.check_comparison_file(
                comp_contents, comp_name, sgroups, comp_style
            )

        # Normalize control group (treat empty string as None)
        control_group_clean = cfg.control_group if (cfg.control_group and str(cfg.control_group).strip() != "") else None

        # Only run DA if we have either a control group or valid comparisons
        has_controls_or_comparisons = bool(control_group_clean) or (comp_data is not None and len(comp_data) > 0)

        if has_controls_or_comparisons:
            comparisons = parsing.parse_comparisons(
                control_group_clean, comp_data, sgroups
            )

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
            divs["volcano"] = volcano_div
        else:
            # No control information; skip DA gracefully
            volcano = None
    with open(os.path.join(cfg.outdir, "04_proteomics_divs.pickle"), "wb") as f:
        pickle.dump(divs, f)
    summary = {
        "workflow": "proteomics",
        "session_name": data_dictionary["other"]["session name"],
        "artifacts": artifacts,
        "na_filtered": na_filtered,
        "normalized": (normalized is not None),
        "imputed": (imputed is not None),
        "volcano": volcano is not None,
        "outdir": cfg.outdir,
    }
    _dump_json(cfg.outdir, "00_summary", summary)
    return summary


def _run_interactomics_workflow(cfg: BatchConfig, data_dictionary: Dict[str, Any],
                               rep_colors: Dict[str, Any], rep_colors_with_cont: Dict[str, Any], params: Dict[str, Any],
                               artifacts: Dict[str, Any]) -> Dict[str, Any]:
    """Run the interactomics analysis workflow.

    :param cfg: Batch configuration.
    :param data_dictionary: Parsed/validated inputs and groups.
    :param rep_colors: Replicate color assignments.
    :param rep_colors_with_cont: Replicate colors incl. contaminants.
    :param params: Parsed application parameters.
    :param artifacts: QC artifacts dict.
    :returns: Interactomics summary dict.
    """
    db_file = os.path.join(*params["Data paths"]["Database file"])
    contaminant_list = db_functions.get_contaminants(db_file)
    divs = {}
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
    saint_div, saint_dict, crapome_data = interactomics.generate_saint_container(
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
    divs["saint"] = saint_div
    
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
    
    filtered_saint = interactomics.map_intensity(filtered_saint, data_dictionary['data tables']['intensity'], data_dictionary['sample groups']['norm'])
    _dump_json(cfg.outdir, "23_saint_filtered_and_intensity_mapped", filtered_saint)
    known_div, filtered_saint_with_knowns = interactomics.known_plot(filtered_saint, db_file, rep_colors_with_cont, params['Figure defaults']['half-height'])
    _dump_json(cfg.outdir, "23_saint_filtered_and_intensity_mapped_with_knowns", filtered_saint_with_knowns)
    divs["known"] = known_div
    # 4.5) Common proteins plot
    saint_matrix = interactomics.get_saint_matrix(filtered_saint)
    common_proteins_div, common_proteins_data = qc_analysis.common_proteins(
        saint_matrix.to_json(orient='split'),
        db_file,
        params["Figure defaults"]["full-height"],
        additional_groups={"Other contaminants": contaminant_list},
        id_str="interactomics",
    )
    _dump_json(cfg.outdir, "23_common_proteins", common_proteins_data)
    divs["common_proteins"] = common_proteins_div
    # 5) Generate network plot
    network_div, network_elements, interactions = interactomics.do_network(
        filtered_saint,
        params["Figure defaults"]["full-height"]["height"]
    )
    _dump_json(cfg.outdir, "24_network_elements", network_elements)
    _dump_json(cfg.outdir, "24_interactions", interactions)
    divs["network"] = network_div
    # 6) PCA analysis
    pca_div, pca_data = interactomics.pca(
        filtered_saint,
        params["Figure defaults"]["full-height"],
        rep_colors
    )
    _dump_json(cfg.outdir, "25_pca", pca_data)
    divs["pca"] = pca_div
    # 7) Enrichment analysis (if requested)
    enrichment_data = None
    enrichment_info = None
    
    root_dir = Path(__file__).resolve().parents[1]
    parameters_path = os.path.join(root_dir, 'config','parameters.toml')
    if cfg.chosen_enrichments:
        enrichment_div, enrichment_data, enrichment_info = interactomics.enrich(
            filtered_saint,
            cfg.chosen_enrichments,
            params["Figure defaults"]["full-height"],
            parameters_file=parameters_path
        )
        _dump_json(cfg.outdir, "26_enrichment_data", enrichment_data)
        _dump_json(cfg.outdir, "26_enrichment_info", enrichment_info)
        divs["enrichment"] = enrichment_div
    # 8) MS-microscopy analysis
    msmic_div, msmic_data = interactomics.do_ms_microscopy(
        filtered_saint,
        db_file,
        params["Figure defaults"]["full-height"]
    )
    _dump_json(cfg.outdir, "27_msmic_data", msmic_data)
    divs["msmic"] = msmic_div
    with open(os.path.join(cfg.outdir, "05_interactomics_divs.pickle"), "wb") as f:
        pickle.dump(divs, f)
    summary = {
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
    }
    _dump_json(cfg.outdir, "00_summary", summary)
    return summary
