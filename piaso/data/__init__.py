from ._genome import (
    fetch_genome,
    resolve_genome_files,
    list_available_genomes,
    list_downloaded_genomes,
    list_available_gtf_sources,
    GTF_SOURCES,
    GTF_PRESETS,
    DEFAULT_GTF_RELEASE,
)
from ._datasets import (
    list_datasets,
    dataset_info,
    fetch_dataset,
    load_dataset,
    refresh_registry,
)

# --- GRN resources: genome sequence (.2bit), motif DBs, TF lists ---
# PWM stays in PIASO (backs piaso.pp.scan_motifs); the .2bit + motif-DB loaders
# moved to the CytoRete package (pip install cytorete) — lazy shims.
from ._pwm import PWM
from .._grn_shim import (
    fetch_2bit,
    resolve_2bit_path,
    extract_sequences,
    revcomp,
    load_meme,
    load_jaspar_meme,
    load_cisbp_meme,
    load_cisbp,
    load_tf_list,
    fetch_jaspar,
    resolve_jaspar_path,
    fetch_cisbp,
    resolve_cisbp_meme_path,
    fetch_cistarget_motifs,
    load_cistarget_motifs,
    resolve_cistarget_paths,
    write_meme,
    fetch_animaltfdb_tf_list,
    build_tf_motif_map,
)

# camelCase public aliases (PIASO convention)
fetchGenomeFasta = fetch_2bit
loadMotifs = load_jaspar_meme        # JASPAR .meme; use load_cisbp_meme / load_cisbp for CIS-BP
loadTFList = load_tf_list
buildTFMotifMap = build_tf_motif_map
fetchJASPAR = fetch_jaspar
fetchCISBP = fetch_cisbp

from ._screen import fetch_screen, resolve_screen_path, load_screen_ccres, ccres_near_tss
fetchSCREEN = fetch_screen
