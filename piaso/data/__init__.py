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

# --- Sequence and motif resources: genome sequence (.2bit), motif DBs, TF lists ---
# These back piaso.pp.scan_motifs. They live here rather than in a downstream
# package because scanning is useless without a way to obtain sequences and
# PWMs: shipping the scanner alone left `piaso.pp.scan_motifs` with no supported
# route to its own inputs.
from ._pwm import PWM
from ._fasta import (
    fetch_2bit,
    resolve_2bit_path,
    extract_sequences,
    revcomp,
)
from ._motifs import (
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
from ._lr import (fetch_lr_database, load_lr_database, resolve_lr_path,
                  LR_URLS, ANNOTATION_CLASSES)
from ._chembl import (fetch_chembl, load_chembl_targets, resolve_chembl_path,
                      filter_chembl_activities, chembl_targets_to_dict,
                      CHEMBL_URL, PCHEMBL_THRESHOLDS)
fetchChEMBL = fetch_chembl
loadChEMBLTargets = load_chembl_targets
fetchLRDatabase = fetch_lr_database
loadLRDatabase = load_lr_database
fetchSCREEN = fetch_screen
