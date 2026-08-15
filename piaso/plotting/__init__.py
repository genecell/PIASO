from ._plotEmbedding import plot_embeddings_split, plotEmbeddingsSplit, plotEmbedding, plotUMAP
from ._plotByCluster import plot_features_violin, plotFeaturesViolin

# Scanpy-style short aliases (primary public API)
embedding = plotEmbedding
umap = plotUMAP
violin = plot_features_violin
split_embedding = plot_embeddings_split

from . import color

from ._plotLigandReceptor import plotLigandReceptorInteraction
from ._plotLigandReceptorLollipop import plotLigandReceptorLollipop


from ._plotCellMetaInfo import plotConfusionMatrix

# --- not in the public distribution: import if present, else install a
# forwarder that raises an actionable ImportError at call time (see
# piaso._internal_shim). `import piaso` must succeed either way.
try:
    from ._plotBigWig import plotBigWig
except ImportError:      # pragma: no cover - the public-package path
    from .._internal_shim import forward_many as _forward_many
    globals().update(_forward_many("piaso.plotting._plotBigWig", ['plotBigWig'], "pl"))
# --- not in the public distribution: import if present, else install a
# forwarder that raises an actionable ImportError at call time (see
# piaso._internal_shim). `import piaso` must succeed either way.
try:
    from ._plotCoverage import plotCoverage, plotWeightedPileup
except ImportError:      # pragma: no cover - the public-package path
    from .._internal_shim import forward_many as _forward_many
    globals().update(_forward_many("piaso.plotting._plotCoverage", ['plotCoverage', 'plotWeightedPileup'], "pl"))

from ._plotDotplot import dotplot, plotDotplot
from ._plotStackedBar import stacked_barplot, stackedBarplot
from ._plotHeatmap import heatmap, plotHeatmap
from ._plotSankey import sankey, plotSankey
from ._plotDendrogram import plot_dendrogram, plotDendrogram
from ._plotScatter import scatter, plotScatter
from ._plotGroupMetrics import plotGroupMetrics, plot_group_metrics
try:
    from ._plotCospecificity import (
        plot_cospecificity_map,
        plot_specificity_hotspot_manhattan,
        plot_specificity_hotspot_recurrence,
    )
except ImportError:      # pragma: no cover - not in the public distribution
    from .._internal_shim import forward_many as _forward_many
    globals().update(_forward_many("piaso.plotting._plotCospecificity", ['plot_cospecificity_map', 'plot_specificity_hotspot_manhattan', 'plot_specificity_hotspot_recurrence'], "pl"))

### Explicitly import
from .color import createCustomCmapFromHex


# Regulon plots moved to the CytoRete package (pip install cytorete) — lazy shims.
from .._grn_shim import (
    pl_regulonActivity as regulonActivity,
    regulonNetwork, regulonEmbedding, regulonSpecificityScatter,
)
