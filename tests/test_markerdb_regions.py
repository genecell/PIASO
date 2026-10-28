"""getMarkerRegions: PIASOmarkerDB's marker regions, as a DataFrame or as a dict of a PIASO store's tile IDs.

A region study ranks the 500 bp tiles of a chromatin atlas by COSG for each cell type, on one genome build. The
client asks the server's /api/v1/regions in rank order, page by page, keeps the top_n of every cell type, and with
as_dict=True names each region as a PIASO store names its tiles ('chr1:3000500-3001000'), so the dict feeds
predictCellTypeByMarker on tile counts. A study on another build than the caller's raises instead of matching
nothing. The server is faked here (the endpoint's semantics, from PIASOmarkerDB's regions.py); nothing touches the
network.
"""
from __future__ import annotations

import warnings

import pandas as pd
import pytest

from piaso.tools import _markerdb

TILE = 500
ZU, WANG = "ZuRen2023_WholeBrain_ATAC", "WangKriegstein2025_Neocortex_ATAC"
COMMA_TYPE = "Macrophage (General,Alveolar)"


def _rows():
    """Two studies on two builds; every region one tile of the 500 bp grid, ranked within its cell type."""
    out = []
    spec = [(ZU, "Mouse", "mm10", "brain", {"Astro": 5, "L2/3 IT": 4, COMMA_TYPE: 3}),
            (WANG, "Human", "hg38", "brain; cortex", {"EN": 3, "IN": 3})]
    pos = 0
    for study, species, genome, tissue, types in spec:
        for ct, n in types.items():
            for rank in range(1, n + 1):
                pos += 7                                  # tiles spread over chr1 and chr2
                chrom, start = ("chr1" if pos % 2 else "chr2"), pos * TILE
                if ct == "Astro" and rank <= 2:           # Astro's ranks 1 and 2 share one 1 kb tile
                    chrom, start = "chr1", 1_000_000 + (rank - 1) * TILE
                out.append(dict(cell_type=ct, chrom=chrom, start=start, end=start + TILE,
                                cosg_score=round(10.0 / rank, 4), cosg_rank=rank, study_publication=study,
                                species=species, tissue=tissue, condition="normal", genome=genome,
                                ccre_class="dELS" if rank % 2 else "pELS", ccre_accession=None,
                                nearest_gene=f"G{ct[:2]}{rank}", tss_distance=1000 * rank,
                                nearby_genes=f"G{ct[:2]}{rank};Gfap" if ct == "Astro" else None,
                                phylop_mean=None, phastcons_mean=None, bases_scored=0, ortholog=None,
                                ortholog_markers=0, ortholog_contexts=None))
    for r in out:
        r["region"] = f"{r['chrom']}:{r['start']}-{r['end']}"
    return out


ROWS = _rows()
ALIASES = [{"old_name": "ZuRen2023_WholeBrain", "new_name": ZU}]


def fake_server(rows=ROWS, honour_sort=True, regions_status=None):
    """``_request`` of a server with the regions endpoints, and the calls it received."""
    calls = []

    def request(method, endpoint, params=None, **kw):
        params = dict(params or {})
        calls.append((endpoint, params))
        if endpoint == "study-aliases":
            return ALIASES
        if endpoint == "region-studies":
            build = params.get("species")
            seen = {}
            for r in rows:
                if build and r["genome"] != build:
                    continue
                k = (r["study_publication"], r["species"], r["genome"], r["tissue"], r["condition"])
                d = seen.setdefault(k, dict(zip(("study_publication", "species", "genome", "tissue", "condition"), k),
                                            cell_types=set(), regions=0))
                d["cell_types"].add(r["cell_type"])
                d["regions"] += 1
            return {"data": [dict(d, cell_types=len(d["cell_types"])) for d in seen.values()]}
        if endpoint != "regions":
            raise AssertionError(f"unexpected endpoint {endpoint}")
        if regions_status:
            raise _markerdb.APIError("not found", status_code=regions_status)
        build = params["species"]
        sel = [r for r in rows if r["genome"] == build]
        if params.get("study"):
            names = [s for s in params["study"].split(",")]
            names = [{a["old_name"]: a["new_name"] for a in ALIASES}.get(s, s) for s in names]
            sel = [r for r in sel if r["study_publication"] in names]
        if params.get("cell_type"):
            v = params["cell_type"]
            whole = any(r["cell_type"] == v for r in sel)
            cts = [v] if whole else [x.strip() for x in v.split(",")]
            sel = [r for r in sel if r["cell_type"] in cts]
        if params.get("gene"):
            gs = {g.upper() for g in params["gene"].split(",")}
            sel = [r for r in sel if r["nearest_gene"].upper() in gs
                   or gs & {x.upper() for x in (r["nearby_genes"] or "").split(";") if x}]
        if params.get("ccre"):
            sel = [r for r in sel if params["ccre"] in (r["ccre_class"] or "").split(";")]
        if params.get("min_score") is not None:
            sel = [r for r in sel if r["cosg_score"] >= float(params["min_score"])]
        sort = params.get("sort", "score")
        if not honour_sort:                               # an older server: genome order, ranks interleaved
            sel.sort(key=lambda r: (r["chrom"], r["start"], r["cell_type"]))
        elif sort == "rank":
            sel.sort(key=lambda r: (r["cosg_rank"], -r["cosg_score"], r["chrom"], r["start"], r["cell_type"]))
        else:
            sel.sort(key=lambda r: (-r["cosg_score"], r["cosg_rank"], r["chrom"], r["start"], r["cell_type"]))
        limit, offset = int(params.get("limit", 1000)), int(params.get("offset", 0))
        page = sel[offset:offset + limit]
        out = {"data": page, "count": len(page), "total": len(sel), "limit": limit, "offset": offset,
               "genome": build, "has_more": offset + len(page) < len(sel)}
        if honour_sort:
            out["sort"] = sort
        return out

    return request, calls


@pytest.fixture
def client(monkeypatch, tmp_path):
    def build(**kw):
        c = _markerdb.PIASOmarkerDB(cache_dir=tmp_path / "cache")
        request, calls = fake_server(**kw)
        monkeypatch.setattr(c, "_request", request)
        return c, calls
    return build


def region_calls(calls):
    return [p for e, p in calls if e == "regions"]


def store_tiles(tile_size=TILE, chrom_sizes={"chr1": 248_956_422, "chr2": 242_193_529}):
    """A store's tile IDs, as a PIASO store names its tiles: '{chrom}:{start}-{start + tile_size}'."""
    return {f"{c}:{i * tile_size}-{(i + 1) * tile_size}" for c, n in chrom_sizes.items()
            for i in range(min(-(-n // tile_size), 3_000_000 // tile_size))}   # the first 3 Mb are enough


# ---------------------------------------------------------------------------------------------- the DataFrame
def test_the_dataframe_keeps_the_servers_fields_sorted_by_study_type_and_rank(client):
    c, _ = client()
    df = c.getMarkerRegions(study=ZU)
    assert list(df.columns) == _markerdb._REGION_COLUMNS
    assert set(df.genome) == {"mm10"} and set(df.study_publication) == {ZU}
    assert len(df) == 5 + 4 + 3
    key = list(zip(df.study_publication, df.cell_type, df.cosg_rank))
    assert key == sorted(key)
    assert (df.region == df.chrom + ":" + df.start.astype(str) + "-" + df.end.astype(str)).all()


def test_top_n_keeps_each_cell_types_best_and_stops_paging_early(client):
    c, calls = client()
    df = c.getMarkerRegions(study=ZU, top_n=2, batch_size=3)
    assert df.groupby("cell_type").cosg_rank.max().to_dict() == {"Astro": 2, "L2/3 IT": 2, COMMA_TYPE: 2}
    assert len(df) == 6
    asked = region_calls(calls)
    assert all(p["sort"] == "rank" and p["order"] == "asc" for p in asked)
    # 12 rows in rank order, pages of 3: ranks 1 1 1 | 2 2 2 | 3 ... the third page passes rank 2, so no fourth
    assert len(asked) == 3


def test_a_server_that_ignores_the_sort_still_gives_the_top_n(client):
    c, calls = client(honour_sort=False)
    df = c.getMarkerRegions(study=ZU, top_n=2, batch_size=3)
    assert df.groupby("cell_type").cosg_rank.max().to_dict() == {"Astro": 2, "L2/3 IT": 2, COMMA_TYPE: 2}
    assert len(region_calls(calls)) == 4                       # every page read: no early stop without rank order


# --------------------------------------------------------------------------------------------- the tile dict
def test_the_dict_names_regions_as_a_stores_tiles_in_rank_order(client):
    c, _ = client()
    d = c.getMarkerRegions(study=ZU, top_n=3, as_dict=True)
    assert set(d) == {"Astro", "L2/3 IT", COMMA_TYPE}
    tiles = store_tiles()
    for ct, ids in d.items():
        assert ids and set(ids) <= tiles, ct
        want = [r["region"] for r in sorted((r for r in ROWS if r["cell_type"] == ct and r["cosg_rank"] <= 3),
                                            key=lambda r: r["cosg_rank"])]
        assert ids == list(dict.fromkeys(want))                   # at 500 bp, the tile ID is the server's region


def test_a_coarser_tile_grid_maps_and_merges_regions(client):
    c, _ = client()
    d = c.getMarkerRegions(study=ZU, cell_type="Astro", top_n=2, tile_size=1000, as_dict=True)
    assert d == {"Astro": ["chr1:1000000-1001000"]}               # ranks 1 and 2 share one 1 kb tile
    assert set(d["Astro"]) <= store_tiles(1000)


def test_tile_ids_follow_the_store(client):
    assert _markerdb._tile_ids("chr1", 3000500, 3001000, 500) == ["chr1:3000500-3001000"]
    assert _markerdb._tile_ids("chr1", 3000400, 3000900, 500) == ["chr1:3000000-3000500", "chr1:3000500-3001000"]
    assert _markerdb._tile_ids("chrX", 0, 500, 2000) == ["chrX:0-2000"]


# ----------------------------------------------------------------------------------------------- the build
def test_a_study_on_another_build_raises(client):
    c, calls = client()
    with pytest.raises(_markerdb.ValidationError, match=r"genome hg38 .*ZuRen2023_WholeBrain_ATAC \(mm10\).*lifted"):
        c.getMarkerRegions(study=ZU, genome="hg38")
    assert not region_calls(calls)


def test_species_and_genome_must_agree(client):
    c, _ = client()
    with pytest.raises(_markerdb.ValidationError, match="mm10"):
        c.getMarkerRegions(species="Mouse", genome="hg38")


def test_the_build_comes_from_the_study(client):
    c, calls = client()
    c.getMarkerRegions(study=WANG)
    assert {p["species"] for p in region_calls(calls)} == {"hg38"}


def test_a_matching_genome_and_the_builds_names_are_accepted(client):
    c, calls = client()
    for g in ("mm10", "MM10", "Mouse"):
        assert len(c.getMarkerRegions(study=ZU, genome=g, top_n=1)) == 3
    assert {p["species"] for p in region_calls(calls)} == {"mm10"}


def test_studies_on_two_builds_raise(client):
    c, _ = client()
    with pytest.raises(_markerdb.ValidationError, match="one genome build at a time"):
        c.getMarkerRegions(study=[ZU, WANG])


def test_an_unknown_study_lists_the_region_studies(client):
    c, _ = client()
    with pytest.raises(_markerdb.ValidationError, match=f"{WANG}.*{ZU}.*getMarkers"):
        c.getMarkerRegions(study="SEAAD2024_MTG_subclass")


def test_nothing_to_choose_a_build_from_raises(client):
    c, _ = client()
    with pytest.raises(_markerdb.ValidationError, match="Pass species"):
        c.getMarkerRegions()
    with pytest.raises(_markerdb.ValidationError, match="Human .hg38. and Mouse .mm10."):
        c.getMarkerRegions(species="Zebrafish")


def test_a_former_study_name_and_another_case_work(client):
    c, calls = client()
    with pytest.warns(UserWarning, match=f"renamed to '{ZU}'"):
        df = c.getMarkerRegions(study="ZuRen2023_WholeBrain", top_n=1)
    assert set(df.study_publication) == {ZU}
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        df = c.getMarkerRegions(study=ZU.lower(), top_n=1)
    assert not [w for w in seen if "renamed" in str(w.message)]
    assert set(df.study_publication) == {ZU}
    assert {p["study"] for p in region_calls(calls)} == {ZU}


# ------------------------------------------------------------------------------------------------- filters
def test_cell_types_with_a_comma_in_the_name_are_asked_for_whole(client):
    c, calls = client()
    df = c.getMarkerRegions(study=ZU, cell_type=COMMA_TYPE)
    assert set(df.cell_type) == {COMMA_TYPE} and len(df) == 3
    calls.clear()
    df = c.getMarkerRegions(study=ZU, cell_type=["Astro", COMMA_TYPE], top_n=1)
    assert sorted(df.cell_type) == sorted(["Astro", COMMA_TYPE])
    assert [p["cell_type"] for p in region_calls(calls)] == ["Astro", COMMA_TYPE]
    calls.clear()
    c.getMarkerRegions(study=ZU, cell_type=["Astro", "L2/3 IT"], top_n=1)
    assert [p["cell_type"] for p in region_calls(calls)] == ["Astro,L2/3 IT"]


def test_the_filters_reach_the_server(client):
    c, calls = client()
    df = c.getMarkerRegions(species="Mouse", gene=["gfap", "Aqp4"], ccre="dELS", min_score=0.5,
                            region=["chr1:0-5000", "chr2:0-1,000"], tissue="brain", condition="normal")
    p = region_calls(calls)[0]
    assert p["gene"] == "gfap,Aqp4" and p["ccre"] == "dELS" and p["min_score"] == 0.5
    assert p["region"] == "chr1:0-5000;chr2:0-1,000" and p["tissue"] == "brain" and p["condition"] == "normal"
    assert set(df.cell_type) == {"Astro"} and set(df.ccre_class) == {"dELS"}


def test_no_match_gives_the_columns_or_an_empty_dict(client):
    c, _ = client()
    df = c.getMarkerRegions(study=ZU, cell_type="Microglia")
    assert df.empty and list(df.columns) == _markerdb._REGION_COLUMNS
    assert c.getMarkerRegions(study=ZU, cell_type="Microglia", as_dict=True) == {}


@pytest.mark.parametrize("kw, err", [
    (dict(as_dict="both"), ValueError), (dict(as_dict=1), ValueError), (dict(top_n=0), _markerdb.ValidationError),
    (dict(top_n=2.5), _markerdb.ValidationError), (dict(tile_size=-500), _markerdb.ValidationError),
    (dict(batch_size=0), _markerdb.ValidationError), (dict(min_score=-1), _markerdb.ValidationError),
])
def test_invalid_arguments(client, kw, err):
    c, _ = client()
    with pytest.raises(err):
        c.getMarkerRegions(species="Mouse", **kw)


def test_a_server_without_regions_says_so(client):
    c, _ = client(regions_status=404)
    with pytest.raises(_markerdb.APIError, match="has no marker regions"):
        c.getMarkerRegions(species="Mouse")


# ---------------------------------------------------------------------------------- the studies and piaso.tl
def test_list_region_studies(client):
    c, _ = client()
    allv = c.listRegionStudies()
    assert list(allv.columns[:7]) == _markerdb._REGION_STUDY_COLUMNS
    assert set(allv.study_publication) == {ZU, WANG}
    human = c.listRegionStudies(species="Human")
    assert set(human.study_publication) == {WANG} and set(human.genome) == {"hg38"}


def test_piaso_tl_getMarkerRegions_uses_the_default_client(client, monkeypatch):
    import piaso
    c, _ = client()
    monkeypatch.setattr(_markerdb, "_default_client", c)
    assert piaso.tl.getMarkerRegions is _markerdb.getMarkerRegions
    assert set(piaso.tl.getMarkerRegions(list_studies=True).study_publication) == {ZU, WANG}
    d = piaso.tl.getMarkerRegions(study=WANG, genome="hg38", top_n=2, as_dict=True)
    assert set(d) == {"EN", "IN"} and all(len(v) == 2 for v in d.values())
    df = piaso.tl.getMarkerRegions(study=WANG, top_n=1)
    assert isinstance(df, pd.DataFrame) and len(df) == 2
