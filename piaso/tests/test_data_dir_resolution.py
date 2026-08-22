"""Tests for the PIASO data-root resolution chain and the v2 registry."""

import json
import os
from pathlib import Path

import pytest

from piaso.data._genome import PIASO_DATA_DIR, resolve_data_dir


class TestResolutionOrder:
    def test_default(self, monkeypatch):
        import piaso.settings as settings
        monkeypatch.setattr(settings, "data_dir", None)
        monkeypatch.delenv("PIASO_DATA_DIR", raising=False)
        assert resolve_data_dir() == PIASO_DATA_DIR

    def test_env_var(self, monkeypatch, tmp_path):
        import piaso.settings as settings
        monkeypatch.setattr(settings, "data_dir", None)
        monkeypatch.setenv("PIASO_DATA_DIR", str(tmp_path / "env_root"))
        assert resolve_data_dir() == tmp_path / "env_root"

    def test_settings_beats_env(self, monkeypatch, tmp_path):
        import piaso.settings as settings
        monkeypatch.setenv("PIASO_DATA_DIR", str(tmp_path / "env_root"))
        monkeypatch.setattr(settings, "data_dir", str(tmp_path / "session_root"))
        assert resolve_data_dir() == tmp_path / "session_root"

    def test_argument_beats_everything(self, monkeypatch, tmp_path):
        import piaso.settings as settings
        monkeypatch.setenv("PIASO_DATA_DIR", str(tmp_path / "env_root"))
        monkeypatch.setattr(settings, "data_dir", str(tmp_path / "session_root"))
        assert resolve_data_dir(tmp_path / "call_root") == tmp_path / "call_root"

    def test_expanduser(self, monkeypatch):
        assert resolve_data_dir("~/somewhere") == Path.home() / "somewhere"

    def test_datasets_dir_follows_root(self, monkeypatch, tmp_path):
        from piaso.data._datasets import _datasets_dir
        import piaso.settings as settings
        monkeypatch.setattr(settings, "data_dir", None)
        monkeypatch.delenv("PIASO_DATA_DIR", raising=False)
        assert _datasets_dir(tmp_path) == tmp_path / "datasets"
        monkeypatch.setenv("PIASO_DATA_DIR", str(tmp_path / "envd"))
        assert _datasets_dir() == tmp_path / "envd" / "datasets"


class TestRegistryV2:
    @pytest.fixture(scope="class")
    def registry(self):
        path = Path(__file__).parent.parent / "data" / "_registry_builtin.json"
        return json.load(open(path))

    def test_version_and_record(self, registry):
        assert registry["version"] == 2
        assert "22012620" in registry["zenodo_record"]

    def test_all_urls_on_new_record(self, registry):
        for name, e in registry["datasets"].items():
            assert "/records/22012620/" in e["url"], name

    def test_five_cytome_entries(self, registry):
        cy = {n: e for n, e in registry["datasets"].items()
              if e.get("format") == "cytome"}
        assert set(cy) == {
            "adult_cortex_multiome_rna_cytome", "sea_ad_mtg_20k_cytome",
            "allen_devvis_rna", "humandevcx_38_rna", "humanlifespan_pfc_rna",
        }
        for name, e in cy.items():
            assert e["md5"] and e["size_bytes"] > 0 and e["filename"].endswith(".cytome"), name

    def test_no_placeholder_record_ids(self, registry):
        assert "<RECORD_ID>" not in json.dumps(registry)

    def test_names_unique_after_suffixing(self, registry):
        # the two datasets available in both formats have distinct names
        assert "adult_cortex_multiome_rna" in registry["datasets"]
        assert "adult_cortex_multiome_rna_cytome" in registry["datasets"]
        assert "sea_ad_mtg_20k" in registry["datasets"]
        assert "sea_ad_mtg_20k_cytome" in registry["datasets"]
