from ._runSVD import runSVD, infog_svd, runSVDLazy

from ._projectGDR import projectGDR
from ._runGDR import runGDR, calculateScoreParallel, calculateScoreParallel_multiBatch, runGDRParallel, runCOSGParallel

from ._clustering import leiden_local

from ._normalization import infog, score

from ._predictCellType import predictCellTypeByGDR, smoothCellTypePrediction, predictCellTypeByMarker

from ._integration import stitchSpace

from ._ligandReceptor import runSCALAR

from ._markerdb import queryPIASOmarkerDB, getMarkers, analyzeMarkers, PIASOmarkerDB

# --- not in the public distribution: import if present, else install a
# forwarder that raises an actionable ImportError at call time (see
# piaso._internal_shim). `import piaso` must succeed either way.
try:
    from ._generateBigWigByCellType import generateBigWigByCellType, splitBamByCellType, generateBigWigFromCytome
except ImportError:      # pragma: no cover - the public-package path
    from .._internal_shim import forward_many as _forward_many
    globals().update(_forward_many("piaso.tools._generateBigWigByCellType", ['generateBigWigByCellType', 'splitBamByCellType', 'generateBigWigFromCytome'], "tl"))

# --- not in the public distribution: import if present, else install a
# forwarder that raises an actionable ImportError at call time (see
# piaso._internal_shim). `import piaso` must succeed either way.
try:
    from ._runMACS2 import runMACS2, runMACS2Parallel
except ImportError:      # pragma: no cover - the public-package path
    from .._internal_shim import forward_many as _forward_many
    globals().update(_forward_many("piaso.tools._runMACS2", ['runMACS2', 'runMACS2Parallel'], "tl"))

from ._runTFIDF import run_TFIDF, compute_tfidf_stats

# --- not in the public distribution: import if present, else install a
# forwarder that raises an actionable ImportError at call time (see
# piaso._internal_shim). `import piaso` must succeed either way.
try:
    from ._runATACLazy import runATACLazy
except ImportError:      # pragma: no cover - the public-package path
    from .._internal_shim import forward_many as _forward_many
    globals().update(_forward_many("piaso.tools._runATACLazy", ['runATACLazy'], "tl"))

from ._neighbors import neighbors
from ._umap import umap
from ._leiden import leiden
# --- not in the public distribution: import if present, else install a
# forwarder that raises an actionable ImportError at call time (see
# piaso._internal_shim). `import piaso` must succeed either way.
try:
    from ._picco import picco, quantify_peaks
except ImportError:      # pragma: no cover - the public-package path
    from .._internal_shim import forward_many as _forward_many
    globals().update(_forward_many("piaso.tools._picco", ['picco', 'quantify_peaks'], "tl"))

# --- not in the public distribution: import if present, else install a
# forwarder that raises an actionable ImportError at call time (see
# piaso._internal_shim). `import piaso` must succeed either way.
try:
    from ._inferGeneActivity import inferGeneActivity
except ImportError:      # pragma: no cover - the public-package path
    from .._internal_shim import forward_many as _forward_many
    globals().update(_forward_many("piaso.tools._inferGeneActivity", ['inferGeneActivity'], "tl"))
# GRN inference moved to the CytoRete package (pip install cytorete); lazy shims.
from .._grn_shim import inferTFActivity, inferGRN

try:
    from ._cospecificity import cospecificity_map, cospecificity_genome_wide, cospecificity_trans
except ImportError:      # pragma: no cover - public-package path
    from .._internal_shim import forward_many as _forward_many
    globals().update(_forward_many("piaso.tools._cospecificity", ['cospecificity_map', 'cospecificity_genome_wide', 'cospecificity_trans'], "tl"))
# Public co-specificity / feature-specificity primitives (used by the CytoRete
# GRN package; promoted from private on the CytoRete extraction).
try:
    from ._cospecificity import _specificity_matrix as specificity_matrix
except ImportError:      # pragma: no cover - public-package path
    from .._internal_shim import forward_many as _forward_many
    globals().update(_forward_many("piaso.tools._cospecificity", ['_specificity_matrix'], "tl"))
try:
    from ._inferGeneActivity import _infer_gene_specificity as infer_feature_specificity
except ImportError:      # pragma: no cover - public-package path
    from .._internal_shim import forward_many as _forward_many
    globals().update(_forward_many("piaso.tools._inferGeneActivity", ['_infer_gene_specificity'], "tl"))
from ._clustering import _read_selected_mask as read_selected_mask
try:
    from ._specificity_hotspot import (
        specificity_hotspot,
        merge_adjacent_hotspots,
    )
except ImportError:      # pragma: no cover - public-package path
    from .._internal_shim import forward_many as _forward_many
    globals().update(_forward_many("piaso.tools._specificity_hotspot", ['specificity_hotspot', 'merge_adjacent_hotspots'], "tl"))

from .external import runHarmony

from .._grn_shim import inferRegulon, regulonActivity, regulonSpecificity
from ._normalize_resolve import ensure_infog_params, ensure_tfidf_params
