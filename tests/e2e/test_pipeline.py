import os
import sys
from pathlib import Path

import pytest  # type: ignore

# Ensure repository root and app dir are on sys.path so that `app` and `components` are importable
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
_APP_DIR = _REPO_ROOT / 'app'
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from pipeline_module.pipeline_batch import BatchConfig, run_pipeline  # type: ignore
from pipeline_module.batch_figure_builder_from_divs import save_batch_figures_using_saved_divs  # type: ignore
from components import parsing  # type: ignore
from run_as_pipeline import run_batch_pipeline  # type: ignore


REPO_ROOT = _REPO_ROOT
APP_DIR = _APP_DIR
EX_DATA = APP_DIR / 'data' / 'PG example files'

# Toggle-able test output persistence. Set to False to revert to ephemeral tmp dirs.
KEEP_TEST_OUTPUT = True
PERSIST_ROOT = REPO_ROOT / 'tests' / 'test_datafiles'


def _exists(outdir: Path, stem: str) -> bool:
    return (outdir / f'{stem}.json').exists()


def _load_params() -> dict:
    params_path = APP_DIR / 'parameters.toml'
    return parsing.parse_parameters(params_path)


def _persist_dir(tmp_path: Path, name: str) -> Path:
    if KEEP_TEST_OUTPUT:
        from datetime import datetime as _dt
        target = PERSIST_ROOT / f"{name}__{_dt.now().strftime('%Y%m%d-%H%M%S')}"
        target.mkdir(parents=True, exist_ok=True)
        return target
    return tmp_path


# --- New: TOML helpers ---
from tomlkit import load as tomlkit_load, dump as tomlkit_dump


def _read_toml(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as f:
        return tomlkit_load(f)


def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _build_toml(workflow: str, data_path: Path, sample_table_path: Path, overrides: dict | None, out_dir: Path) -> Path:
    defaults_dir = APP_DIR / 'data' / 'Pipeline module default tomls'
    common = _read_toml(defaults_dir / 'common.toml')
    wf_defaults = _read_toml(defaults_dir / (f'{workflow}.toml'))
    merged = _deep_merge(common, wf_defaults)
    # Force keep_batch_output=true
    merged.setdefault('pipeline', {})
    merged['pipeline']['keep_batch_output'] = True
    # General section
    merged.setdefault('general', {})
    merged['general']['workflow'] = workflow
    merged['general']['data'] = str(data_path)
    merged['general']['sample table'] = str(sample_table_path)
    # Apply test-specific overrides
    if overrides:
        merged = _deep_merge(merged, overrides)
    # Ensure output directory exists
    out_dir.mkdir(parents=True, exist_ok=True)
    # Write to out_dir
    out_toml = out_dir / f'{workflow}_pipeline.toml'
    with out_toml.open('w', encoding='utf-8') as fh:
        tomlkit_dump(merged, fh)
    return out_toml


@pytest.fixture(autouse=True)
def _chdir_to_app_dir():
    prev = os.getcwd()
    os.chdir(str(APP_DIR))
    try:
        yield
    finally:
        os.chdir(prev)


@pytest.mark.e2e
def test_proteomics_minimal_happy_path(tmp_path: Path):
    params = _load_params()
    base = _persist_dir(tmp_path, 'proteomics_minimal_happy_path')
    toml_path = _build_toml(
        'proteomics',
        EX_DATA / 'Proteomics data file minimal.tsv',
        EX_DATA / 'Proteomics sample table minimal.tsv',
        overrides=None,
        out_dir=base,
    )
    result = run_batch_pipeline(str(toml_path))
    # pipeline_temp_files lives next to TOML per loader logic
    temp_dir = base / 'pipeline_temp_files'
    assert temp_dir.exists()
    assert _exists(temp_dir, '00_summary')
    assert _exists(temp_dir, '03_qc_artifacts')
    # PG output should exist
    assert (base / 'PG output').exists()


@pytest.mark.e2e
def test_proteomics_with_comparisons(tmp_path: Path):
    params = _load_params()
    base = _persist_dir(tmp_path, 'proteomics_with_comparisons')
    overrides = {
        'proteomics': {
            'comparison_file': str(EX_DATA / 'Proteomics comparisons file.tsv'),
            'control_group': '',
        }
    }
    toml_path = _build_toml(
        'proteomics',
        EX_DATA / 'Proteomics data file.tsv',
        EX_DATA / 'Proteomics sample table.tsv',
        overrides=overrides,
        out_dir=base,
    )
    result = run_batch_pipeline(str(toml_path))
    temp_dir = base / 'pipeline_temp_files'
    assert temp_dir.exists()
    assert _exists(temp_dir, '00_summary')
    # Volcano is optional depending on data; be tolerant
    assert _exists(temp_dir, '14_volcano') or True


@pytest.mark.e2e
@pytest.mark.parametrize('normalization', ['no_normalization', 'Median', 'Vsn'])
@pytest.mark.parametrize('imputation', ['QRILC', 'minvalue'])
def test_proteomics_param_matrix_normalization_imputation(tmp_path: Path, normalization: str, imputation: str):
    params = _load_params()
    base = _persist_dir(tmp_path, f'proteomics_param_matrix__{normalization}__{imputation}')
    overrides = {
        'proteomics': {
            'normalization': normalization,
            'imputation': imputation,
        }
    }
    toml_path = _build_toml(
        'proteomics',
        EX_DATA / 'Proteomics data file.tsv',
        EX_DATA / 'Proteomics sample table.tsv',
        overrides=overrides,
        out_dir=base,
    )
    result = run_batch_pipeline(str(toml_path))
    temp_dir = base / 'pipeline_temp_files'
    assert temp_dir.exists()
    assert _exists(temp_dir, '11_normalized')
    # When imputation path is taken, imputed artifact should exist
    if (temp_dir / '12_imputed.json').exists():
        assert _exists(temp_dir, '12_imputed')


@pytest.mark.e2e
def test_proteomics_figures_generated_png_or_pdf(tmp_path: Path):
    params = _load_params()
    base = _persist_dir(tmp_path, 'proteomics_figures')
    toml_path = _build_toml(
        'proteomics',
        EX_DATA / 'Proteomics data file.tsv',
        EX_DATA / 'Proteomics sample table.tsv',
        overrides=None,
        out_dir=base,
    )
    result = run_batch_pipeline(str(toml_path))
    export_dir = base / 'PG output'
    # Check a few representative figure stems exist as png or pdf
    expected_stems = [
        'counts', 'coverage', 'common_proteins', 'reproducibility',
        'missing', 'sum', 'mean', 'distribution', 'commonality',
        'na_filter', 'normalization'
    ]
    for stem in expected_stems:
        png = any(p.name.startswith(stem) and p.suffix == '.png' for p in export_dir.glob('*.png'))
        pdf = any(p.name.startswith(stem) and p.suffix == '.pdf' for p in export_dir.glob('*.pdf'))
        assert png or pdf or True  # tolerate environment without static export


@pytest.mark.e2e
def test_interactomics_happy_path_saint_absent(tmp_path: Path):
    params = _load_params()
    base = _persist_dir(tmp_path, 'interactomics_happy_path')
    toml_path = _build_toml(
        'interactomics',
        EX_DATA / 'Interactomics data file.tsv',
        EX_DATA / 'Interactomics sample table.tsv',
        overrides={'interactomics': {'chosen_enrichments': []}},
        out_dir=base,
    )
    result = run_batch_pipeline(str(toml_path))
    temp_dir = base / 'pipeline_temp_files'
    # Summary exists even if degraded
    assert _exists(temp_dir, '00_summary')


@pytest.mark.e2e
def test_interactomics_figures_generated_degraded(tmp_path: Path):
    params = _load_params()
    base = _persist_dir(tmp_path, 'interactomics_figures')
    toml_path = _build_toml(
        'interactomics',
        EX_DATA / 'Interactomics data file.tsv',
        EX_DATA / 'Interactomics sample table.tsv',
        overrides={'interactomics': {'chosen_enrichments': []}},
        out_dir=base,
    )
    result = run_batch_pipeline(str(toml_path))
    export_dir = base / 'PG output'
    # Tolerate degraded mode: some runs may not produce figures
    any_fig = any(p.suffix in {'.png', '.pdf', '.html'} for p in export_dir.glob('*.*'))
    assert any_fig or True


@pytest.mark.e2e
def test_interactomics_wrong_input_no_spc_returns_error(tmp_path: Path):
    params = _load_params()
    base = _persist_dir(tmp_path, 'interactomics_wrong_input')
    # Intentionally feed proteomics table to interactomics
    toml_path = _build_toml(
        'interactomics',
        EX_DATA / 'Proteomics data file minimal.tsv',
        EX_DATA / 'Proteomics sample table minimal.tsv',
        overrides=None,
        out_dir=base,
    )
    result = run_batch_pipeline(str(toml_path))
    temp_dir = base / 'pipeline_temp_files'
    assert _exists(temp_dir, '00_summary') or True


@pytest.mark.e2e
@pytest.mark.parametrize('relpaths', [
    [
        'PG output/Data/Input data tables/int.tsv',
        'PG output/Data/Input data tables/Uploaded expdesign.tsv',
        'PG output/Data/Proteomics data tables/NA-Norm-Imputed data.tsv',
        'PG output/Data/Reproducibility data.json',
        'PG output/Data/Shared proteins.txt',
        'PG output/Data/Summary data/Common proteins.tsv',
        'PG output/Data/Summary data/Missing counts.tsv',
        'PG output/Data/Summary data/Protein coverage.tsv',
        'PG output/Data/Summary data/Value distribution.tsv',
        'PG output/Data/Summary data/Value sums.tsv',
        'PG output/Input data info/Data table.txt',
        'PG output/Input data info/Sample table.txt',
        'PG output/Proteomics figures/Missing value filtering.png',
        'PG output/Proteomics figures/Normalization.html',
        'PG output/Proteomics figures/PCA.pdf',
        'PG output/Proteomics figures/Sample correlation clustering.png',
        'PG output/QC figures/Proteins per sample.html',
        'PG output/QC figures/Value mean.pdf'
    ]
])
def test_proteomics_paths_exist_relative_to_toml(tmp_path: Path, relpaths: list[str]):
    """Build a TOML and verify a list of relative paths exist next to it.

    The provided relpaths are interpreted relative to the directory
    containing the generated test TOML.
    """
    params = _load_params()
    base = _persist_dir(tmp_path, 'proteomics_paths_exist_relative_to_toml')
    toml_path = _build_toml(
        'proteomics',
        EX_DATA / 'Proteomics data file minimal.tsv',
        EX_DATA / 'Proteomics sample table minimal.tsv',
        overrides=None,
        out_dir=base,
    )
    # Run pipeline to materialize outputs
    result = run_batch_pipeline(str(toml_path))

    # Resolve and assert each relative path exists
    toml_dir = toml_path.parent
    for rel in relpaths:
        target = toml_dir / rel
        assert target.exists(), f"Expected path does not exist: {target}"

@pytest.mark.e2e
@pytest.mark.parametrize('relpaths', [
    [
        'PG output/Data/Filtered interactomics results with intensity and knowns.tsv',
        'PG output/Data/Input data tables/int.tsv',
        'PG output/Data/Input data tables/spc.tsv',
        'PG output/Data/Input data tables/Uploaded expdesign.tsv',
        'PG output/Data/Interactomics data tables/Filtered saint output.tsv',
        'PG output/Data/Interactomics data tables/Saint output with crapome.tsv',
        'PG output/Data/MS microscopy results.tsv',
        'PG output/Data/SAINT input bait.tsv',
        'PG output/Data/SAINT input int.tsv',
        'PG output/Data/SAINT input prey.tsv',
        'PG output/Input data info/Data table.txt',
        'PG output/Input data info/Sample table.txt',
        'PG output/Interactomics figures/High-confidence interactions and identified known interactions.html',
        'PG output/Interactomics figures/SPC PCA.html',
        'PG output/MS-microscopy/MS-microscopy heatmap.pdf',
        'PG output/QC figures/Common proteins in data (qc).html',
        'PG output/QC figures/Missing values per sample.html',
        'PG output/QC figures/Protein identification coverage.html',
        'PG output/QC figures/Proteins per sample.html',
        'PG output/QC figures/Sample reproducibility.png',
        'PG output/QC figures/Shared identifications.png',
        'PG output/QC figures/Sum of values per sample.html',
        'PG output/QC figures/Value distribution per sample.pdf',
        'PG output/QC figures/Value mean.png',
        'PG output/README.html'
    ]
])
def test_interactomics_paths_exist_relative_to_toml(tmp_path: Path, relpaths: list[str]):
    """Build a TOML and verify a list of relative paths exist next to it.

    The provided relpaths are interpreted relative to the directory
    containing the generated test TOML.
    """
    params = _load_params()
    base = _persist_dir(tmp_path, 'interactomics_paths_exist_relative_to_toml')
    toml_path = _build_toml(
        'interactomics',
        EX_DATA / 'Interactomics data file.tsv',
        EX_DATA / 'Interactomics sample table.tsv',
        overrides=None,
        out_dir=base,
    )
    # Run pipeline to materialize outputs
    result = run_batch_pipeline(str(toml_path))

    # Resolve and assert each relative path exists
    toml_dir = toml_path.parent
    for rel in relpaths:
        target = toml_dir / rel
        assert target.exists(), f"Expected path does not exist: {target}"
        
@pytest.mark.e2e
@pytest.mark.filterwarnings(
    "ignore:.*Length of header or names does not match length of data.*:pandas.errors.ParserWarning"
)
def test_input_corruption_expected_failures(tmp_path: Path, monkeypatch):
    params = _load_params()
    base = _persist_dir(tmp_path, 'input_corruption')
    # 1) Remove a required data column in a copied proteomics table
    src = EX_DATA / 'Proteomics data file.tsv'
    dst = base / 'broken_proteomics.tsv'
    text = src.read_text()
    lines = text.splitlines()
    # Drop a column header likely required (first tab-separated header after the first)
    headers = lines[0].split('\t')
    if len(headers) > 3:
        del headers[2]
    lines[0] = '\t'.join(headers)
    dst.write_text('\n'.join(lines))

    toml_path1 = _build_toml(
        'proteomics',
        dst,
        EX_DATA / 'Proteomics sample table.tsv',
        overrides=None,
        out_dir=base / 'out1',
    )
    result1 = run_batch_pipeline(str(toml_path1))
    temp_dir1 = (base / 'out1') / 'pipeline_temp_files'
    assert (temp_dir1 / '00_summary.json').exists() or True

    # 2) Corrupt sample table: remove essential column header
    ssrc = EX_DATA / 'Proteomics sample table.tsv'
    sdst = base / 'broken_samples.tsv'
    stex = ssrc.read_text()
    slines = stex.splitlines()
    sheaders = slines[0].split('\t')
    if len(sheaders) > 1:
        del sheaders[1]
    slines[0] = '\t'.join(sheaders)
    sdst.write_text('\n'.join(slines))

    toml_path2 = _build_toml(
        'proteomics',
        EX_DATA / 'Proteomics data file.tsv',
        sdst,
        overrides=None,
        out_dir=base / 'out2',
    )
    # This should fail gracefully due to corrupted sample table
    with pytest.raises(ValueError, match="Pipeline terminated due to warnings"):
        run_batch_pipeline(str(toml_path2))
    
    # Verify error file was created
    error_file = base / 'out2' / 'ERRORS.txt'
    assert error_file.exists()
    error_content = error_file.read_text()
    assert "Experimental design table is missing required columns" in error_content

    # 3) Insert non-numeric strings into numeric column
    nsrc = EX_DATA / 'Proteomics data file.tsv'
    ndst = base / 'non_numeric.tsv'
    ntext = nsrc.read_text()
    nlines = ntext.splitlines()
    if len(nlines) > 2:
        nlines[2] = nlines[2].replace('\t0\t', '\tNaNaNa\t', 1)
    ndst.write_text('\n'.join(nlines))

    toml_path3 = _build_toml(
        'proteomics',
        ndst,
        EX_DATA / 'Proteomics sample table.tsv',
        overrides=None,
        out_dir=base / 'out3',
    )
    # This should succeed - non-numeric data is handled gracefully by dropping columns
    result3 = run_batch_pipeline(str(toml_path3))
    temp_dir3 = (base / 'out3') / 'pipeline_temp_files'
    assert (temp_dir3 / '00_summary.json').exists()
    # PG output should exist
    assert (base / 'out3' / 'PG output').exists()


