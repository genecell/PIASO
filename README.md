[![Stars](https://img.shields.io/github/stars/genecell/PIASO?logo=GitHub&color=yellow)](https://github.com/genecell/PIASO/stargazers)
[![PyPI](https://img.shields.io/pypi/v/piaso-tools?logo=PyPI)](https://pypi.org/project/piaso-tools)
[![Total downloads](https://static.pepy.tech/personalized-badge/piaso-tools?period=total&units=international_system&left_color=black&right_color=orange&left_text=downloads)](https://pepy.tech/project/piaso-tools)
[![Monthly downloads](https://static.pepy.tech/personalized-badge/piaso-tools?period=month&units=international_system&left_color=black&right_color=orange&left_text=downloads/month)](https://pepy.tech/project/piaso-tools)

# PIASO

#### Precise Integrative Analysis of Single-cell Omics

PIASO is a Python toolkit for single-cell omics analysis: marker-gene-guided
dimensionality reduction (GDR), INFOG normalization, gene-set scoring,
cell-type annotation and label transfer, and a plotting suite built for
publication figures.

It works on an `AnnData` in memory, and on
[cytome](https://github.com/genecell/cytome) datasets by streaming from disk in
chunks — peak memory is set by the batch size rather than by the number of
cells, so the same functions run on a few thousand cells or on several million.

### Documentation

[piaso.org](https:///piaso.org) 

### Installation

Install from PyPI (stable release):
```bash
pip install piaso-tools
```

This also installs [cytome](https://github.com/genecell/cytome), the on-disk
dataset format PIASO reads and writes. Nothing extra to install to work with
`.cytome` files.

Install from bioconda (stable release):
```bash
conda install -c conda-forge -c bioconda piaso
```

Install from GitHub (latest development version):
```bash
pip install git+https://github.com/genecell/PIASO.git
```

### Using PIASO with a coding agent

[PIASO-for-agents](https://github.com/genecell/PIASO-for-agents) makes the
PIASO ecosystem available to coding agents from one canonical knowledge base,
generating Claude skills, Cursor rules, `AGENTS.md`, `llms.txt`, and an MCP
server. Useful if you work in Claude Code, Cursor, Copilot, Codex, Windsurf,
Cline, or Aider and want the agent to know the current API rather than guess it.

### Citation

If PIASO is useful for your research, please consider citing Wu, S.J., Dai, M. *et al*. Pyramidal neurons proportionately alter cortical interneuron subtypes. *Nature* (2026). https://doi.org/10.1038/s41586-025-09996-8

### Contact
Min Dai
dai@broadinstitute.org
