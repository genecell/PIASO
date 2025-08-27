from ._runSVD import runSVD, runSVDLazy

from ._runGDR import runGDR, calculateScoreParallel, calculateScoreParallel_multiBatch, runGDRParallel, runCOSGParallel

from ._clustering import leiden_local

from ._normalization import infog, score

from ._predictCellType import predictCellTypeByGDR, smoothCellTypePrediction, predictCellTypeByMarker

from ._integration import stitchSpace

from ._ligandReceptor import runSCALAR