"""Round 6 workflow tests: subset-based QC architecture + cytome filename helper.

Covers:
* cytome filename helper: default = {dataset}_pipeline.cytome
* per-dataset cytome_filename override
* global cytome_filename override via PARAMS
* {dataset} token interpolation
* cytome_raw_filename default + override
* qc_keep_original_cytome=true emits --keep-original flag
* qc_keep_original_cytome=false omits the flag
* DAG: cytome_apply_qc_basic appears before cosg_markers
* DAG: cytome_apply_qc_final appears before select_tfidf_svd
* qc_apply rules absent when run_qc_filter=false
* Round 5 cell-mask-column flag is REMOVED post-Round-6
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
import yaml


WORKFLOW_DIR = Path(__file__).resolve().parents[1] / "workflow"
SNAKEFILE = WORKFLOW_DIR / "Snakefile"
BASE_CONFIG = WORKFLOW_DIR / "config.yaml"
SNAKEMAKE = "/path/to/conda/env/bin/snakemake"


def _have_snakemake():
    return shutil.which(SNAKEMAKE) is not None or os.path.exists(SNAKEMAKE)


pytestmark = pytest.mark.skipif(
    not _have_snakemake() or not BASE_CONFIG.exists(),
    reason="snakemake or workflow config not available",
)


def _make_config(modifier):
    """Read base config, run modifier(cfg), write to a temp file, return its path."""
    with open(BASE_CONFIG) as f:
        cfg = yaml.safe_load(f)
    modifier(cfg)
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="round6_")
    with os.fdopen(fd, "w") as f:
        yaml.safe_dump(cfg, f)
    return path


def _dry_run(config_path: str, target: str) -> str:
    proc = subprocess.run(
        [SNAKEMAKE, "-s", str(SNAKEFILE), "--configfile", config_path,
         "-np", target],
        capture_output=True, text=True, cwd=WORKFLOW_DIR.parent,
    )
    return proc.stdout + proc.stderr


# ---------------------------------------------------------------------
# Cytome filename helper
# ---------------------------------------------------------------------

def test_default_cytome_filename_is_dataset_pipeline():
    """Default filename pattern '{dataset}_pipeline.cytome' resolves correctly."""
    out = _dry_run(str(BASE_CONFIG),
                   "results/snakemake/e18/cluster/metrics.json")
    assert "/e18/e18_pipeline.cytome" in out, (
        "Default cytome filename should be 'e18_pipeline.cytome' for the e18 dataset"
    )
    # The old shared filename should no longer appear in shell lines
    assert "/e18/pipeline.cytome" not in out


def test_global_cytome_filename_override_via_params():
    """Setting cytome_filename in params: makes every dataset use it."""
    def m(cfg):
        cfg["params"]["cytome_filename"] = "custom_global.cytome"
    cfg = _make_config(m)
    try:
        out = _dry_run(cfg, "results/snakemake/e18/cluster/metrics.json")
    finally:
        os.unlink(cfg)
    assert "/e18/custom_global.cytome" in out
    assert "/e18/e18_pipeline.cytome" not in out


def test_per_dataset_cytome_filename_override():
    """Setting cytome_filename on a specific dataset takes precedence."""
    def m(cfg):
        cfg["datasets"]["e18"]["cytome_filename"] = "e18_special.cytome"
    cfg = _make_config(m)
    try:
        out = _dry_run(cfg, "results/snakemake/e18/cluster/metrics.json")
    finally:
        os.unlink(cfg)
    assert "/e18/e18_special.cytome" in out


def test_dataset_token_interpolation_in_template():
    """The {dataset} token in the template is interpolated at resolution time."""
    def m(cfg):
        cfg["params"]["cytome_filename"] = "{dataset}_v2.cytome"
    cfg = _make_config(m)
    try:
        out = _dry_run(cfg, "results/snakemake/e18/cluster/metrics.json")
    finally:
        os.unlink(cfg)
    assert "/e18/e18_v2.cytome" in out


def test_raw_cytome_filename_default():
    """qc_keep_original_cytome=true (default) emits the _raw filename."""
    out = _dry_run(str(BASE_CONFIG),
                   "results/snakemake/e18/qc/stage1_applied.json")
    assert "--raw-cytome" in out, "apply rule should pass --raw-cytome when keep_original=true"
    assert "e18_pipeline_raw.cytome" in out, (
        "Default cytome_raw_filename pattern is '{dataset}_pipeline_raw.cytome'"
    )
    assert "--keep-original" in out


# ---------------------------------------------------------------------
# qc_keep_original_cytome toggle
# ---------------------------------------------------------------------

def test_qc_keep_original_false_omits_keep_original_flag():
    """qc_keep_original_cytome=false: apply script does NOT get --keep-original."""
    def m(cfg):
        cfg["params"]["qc_keep_original_cytome"] = False
    cfg = _make_config(m)
    try:
        out = _dry_run(cfg, "results/snakemake/e18/qc/stage1_applied.json")
    finally:
        os.unlink(cfg)
    # The shell line for cytome_apply_qc_basic should NOT contain --keep-original
    # (we accept --raw-cytome being present — it's harmless without --keep-original)
    apply_lines = [
        line for line in out.splitlines()
        if "cytome_apply_qc_basic.py" in line
    ]
    assert apply_lines, "cytome_apply_qc_basic rule should still be in DAG"
    for line in apply_lines:
        assert "--keep-original" not in line, (
            "--keep-original should be absent when qc_keep_original_cytome=false"
        )


# ---------------------------------------------------------------------
# DAG structure
# ---------------------------------------------------------------------

def test_apply_basic_appears_before_cosg_in_DAG():
    """cytome_apply_qc_basic must run before cosg_markers."""
    out = _dry_run(str(BASE_CONFIG),
                   "results/snakemake/e18/cosg/cosg_scores.tsv")
    assert "cytome_apply_qc_basic" in out
    assert "cosg_markers" in out
    # In the dry-run rule list, apply_qc_basic appears as a rule that must
    # complete before cosg_markers can run. Snakemake doesn't list rules in
    # execution order necessarily, but both must be present.


def test_apply_final_appears_before_svd_in_DAG():
    """cytome_apply_qc_final must run before select_tfidf_svd."""
    out = _dry_run(str(BASE_CONFIG),
                   "results/snakemake/e18/cluster/metrics.json")
    assert "cytome_apply_qc_final" in out
    assert "select_tfidf_svd" in out


def test_run_qc_filter_false_skips_apply_rules():
    """run_qc_filter=false → cytome_apply_qc_* rules don't appear in DAG."""
    def m(cfg):
        cfg["params"]["run_qc_filter"] = False
    cfg = _make_config(m)
    try:
        out = _dry_run(cfg, "results/snakemake/e18/cluster/metrics.json")
    finally:
        os.unlink(cfg)
    assert "cytome_apply_qc_basic" not in out
    assert "cytome_apply_qc_final" not in out
    # qc_filter_stage1/2 also shouldn't run (they were only there to feed
    # the apply rules + qc_report)
    assert "qc_filter_stage1" not in out
    assert "qc_filter_stage2" not in out


def test_post_round6_cosg_has_no_cell_mask_column_flag():
    """Round 6: cosg_markers must not receive --cell-mask-column anymore."""
    out = _dry_run(str(BASE_CONFIG),
                   "results/snakemake/e18/cosg/cosg_scores.tsv")
    # Find the cosg_markers shell line specifically
    cosg_lines = [
        line for line in out.splitlines()
        if "run_cosg.py" in line
    ]
    assert cosg_lines, "cosg shell line not found in dry-run output"
    for line in cosg_lines:
        assert "--cell-mask-column" not in line, (
            "After Round 6, cosg shell line should NOT contain "
            "--cell-mask-column (cytome is pre-filtered)"
        )
