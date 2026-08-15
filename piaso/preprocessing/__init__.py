# --- not in the public distribution: import if present, else install a
# forwarder that raises an actionable ImportError at call time (see
# piaso._internal_shim). `import piaso` must succeed either way.
try:
    from ._selectPeaks import selectPeaks
except ImportError:      # pragma: no cover - the public-package path
    from .._internal_shim import forward_many as _forward_many
    globals().update(_forward_many("piaso.preprocessing._selectPeaks", ['selectPeaks'], "pp"))
from ._filtering import filter_cells, filter_features, subset_cells
from ._normalize_log1p import normalize_log1p

try:
    from ._importFragments import importFragments
except ImportError:
    pass

try:
    from ._importCellRanger import importCellRanger
except ImportError:
    pass

try:
    from ._processFragment import (
        filterFragments, filterFragmentByBarcode, filterFragmentByRegion,
        filterFragmentByLength, loadFragmentAsTile, mergeFragmentFiles,
        preprocessFragmentFile, generateInsertionSiteFragmentFile,
        fragment_length_distribution, count_fragments_per_cell,
    )
except ImportError:
    pass

try:
    from ._quantifyPeakActivity import quantifyPeakActivity
except ImportError:      # pragma: no cover - not in the public distribution
    from .._internal_shim import forward_many as _forward_many
    globals().update(_forward_many(
        "piaso.preprocessing._quantifyPeakActivity", ['quantifyPeakActivity'], "pp"))

try:
    from ._sweep import load_peaks, sweep_intersect
except ImportError:
    pass

try:
    from ._streaming_io import _open_fragments, _build_barcode_index
except ImportError:
    pass

try:
    from ._calculateMetrics import (
        calculateCellMetrics,
        calculateGroupMetrics,
        calculateFeatureMetrics,
        calculatePeakMetrics,  # deprecated K-HARD stub → raises TypeError
        calculateTSSEnrichmentScore,
    )
except ImportError:
    pass

try:
    from ._processPeakFile import processTSSbed_python
except ImportError:
    pass

try:
    from ._interval_overlap import (
        intersect_peaks_with_genes,
        intersect_peaks_with_promoters,
        parse_peak_coords,
        load_bed,
        subtract_intervals,
    )
except ImportError:
    pass

try:
    from ._dataFrame_processing import table, getCrossCategories
except ImportError:
    pass

try:
    from ._spatialPreprocessing import rotateSpatialCoordinates
except ImportError:
    pass

from ._scrublet import scrublet

# --- GRN preprocessing transforms ---
# The motif SCANNER stays in PIASO (backs piaso.pp.scan_motifs); the cistrome +
# promoter BUILDERS moved to the CytoRete package (pip install cytorete) — lazy shims.
from .grn._scan import scan_motifs, scan_motifs_numpy, estimate_background
# Public PWM-scan primitives (used by the CytoRete GRN package).
from .grn._scan import pvalue_to_threshold
from .grn._scan import _rust_ext_available as rust_ext_available
from .grn._scan_rust import scan_motifs_rust
from .._grn_shim import (
    extract_promoter_sequences, build_cistrome, build_peak_cistrome, bulk_base_cistrome,
)
extractPromoterSequences = extract_promoter_sequences
scanMotifs = scan_motifs
buildCistrome = build_cistrome
buildPeakCistrome = build_peak_cistrome
bulkBaseCistrome = bulk_base_cistrome
