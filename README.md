[![Stars](https://img.shields.io/github/stars/genecell/PIASO?logo=GitHub&color=yellow)](https://github.com/genecell/PIASO/stargazers)
[![PyPI](https://img.shields.io/pypi/v/piaso-tools?logo=PyPI)](https://pypi.org/project/piaso-tools)
[![Total downloads](https://static.pepy.tech/personalized-badge/piaso-tools?period=total&units=international_system&left_color=black&right_color=orange&left_text=downloads)](https://pepy.tech/project/piaso-tools)
[![Monthly downloads](https://static.pepy.tech/personalized-badge/piaso-tools?period=month&units=international_system&left_color=black&right_color=orange&left_text=downloads/month)](https://pepy.tech/project/piaso-tools)

# PIASO

#### Precise Integrative Analysis of Single-cell Omics

### Documentation

[PIASO documentation](https://genecell.github.io/PIASO/) 

#### Current available functionalities

1. perform dimensionality reduction with GDR
2. integration of scRNA-seq datasets with GDR
3. integration of scRNA-seq data and MERFISH/Xenium data with GDR
4. normalization of scRNA-seq data with INFOG
5. selection of highly-variable genes in scRNAseq data with INFOG
6. a novel gene set scoring method
7. run clustering on selected cluster(s)
8. side-by-side cell embedding plots, e.g., split by Conditions
9. stacked violin plots for multiple features, including genes and cell metrics

#### Coming functionalities

1. preprocessing of scATAC-seq datasets
2. integration of scRNA-seq and scATAC-seq datasets (not relying on gene activities)
3. inference of cell type-specific gene regulatory networks
4. and others

### Installation

You could simply install PIASO via `pip` in your conda environment:
```bash
pip install piaso-tools
```
For the development version in GitHub, you could install via:
```bash
pip install git+https://github.com/genecell/PIASO.git
```

### Citation

If PIASO is useful for your research, please consider citing [Wu et al., Pyramidal neurons proportionately alter the identity and survival of specific cortical interneuron subtypes, bioRxiv (2024)](https://www.biorxiv.org/content/10.1101/2024.07.20.604399v1). 

### Contact
Min Dai
dai@broadinstitute.org