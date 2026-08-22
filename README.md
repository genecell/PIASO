[![Stars](https://img.shields.io/github/stars/genecell/PIASO?logo=GitHub&color=yellow)](https://github.com/genecell/PIASO/stargazers)
[![PyPI](https://img.shields.io/pypi/v/piaso-tools?logo=PyPI)](https://pypi.org/project/piaso-tools)
[![Total downloads](https://static.pepy.tech/personalized-badge/piaso-tools?period=total&units=international_system&left_color=black&right_color=orange&left_text=downloads)](https://pepy.tech/project/piaso-tools)
[![Monthly downloads](https://static.pepy.tech/personalized-badge/piaso-tools?period=month&units=international_system&left_color=black&right_color=orange&left_text=downloads/month)](https://pepy.tech/project/piaso-tools)

# PIASO

#### Precise Integrative Analysis of Single-cell Omics

PIASO is a **Python and Rust** toolkit for single-cell omics — scRNA-seq,
scATAC-seq and spatial transcriptomics — covering the analysis from raw counts
through to the figures in a paper. Performance-critical routines are implemented
in Rust; pre-compiled wheels ship for Linux, macOS (Intel and Apple Silicon) and
Windows, so there is nothing to build.

```python
import piaso, cosg

piaso.tl.infog(adata, n_top_genes=3000)                       # normalize + select
piaso.tl.runSVD(adata, layer="infog", n_components=50, key_added="X_svd")
piaso.tl.neighbors(adata, use_rep="X_svd"); piaso.tl.leiden(adata)
piaso.tl.umap(adata, use_rep="X_svd")
cosg.cosg(adata, groupby="leiden")                            # marker genes
piaso.pl.embedding(adata, color="leiden")
```

That whole workflow runs on a plain `pip install piaso-tools` — no scanpy
required.

### Who this is for

- **You have a single-cell dataset and want an analysis, not a toolchain.**
  Reading, QC, normalization, dimensionality reduction, clustering, marker
  genes, annotation and plotting are one package with one set of conventions.
- **Your data outgrew memory.** The same function calls run on an `AnnData` in
  RAM or stream from a [cytome](https://github.com/genecell/cytome) file on
  disk, where peak memory is set by the batch size instead of the cell count.
  Validated to several million cells.
- **You care what the figure looks like.** The plotting suite and
  `piaso.settings` are built for publication figures rather than for quick
  looks.
- **You work with a coding agent.** The API is published in an agent-readable
  form (see below), so the agent works from the current signatures.

If you only need one method, the pieces are usable on their own — `cosg` for
markers, `cytome` for storage.

### The ecosystem

PIASO is the analysis layer of a small set of packages that fit together, and
each is useful alone:

| | |
|---|---|
| **[PIASO](https://github.com/genecell/PIASO)** | analysis: normalization, dimensionality reduction, clustering, annotation, plotting |
| **[cytome](https://github.com/genecell/cytome)** | a single-file format for single-cell multi-omics; what PIASO streams from |
| **[COSG](https://github.com/genecell/COSG)** | fast, accurate marker gene and marker region identification |
| **[PIASO-data](https://github.com/genecell/PIASO-data)** | genome references and tutorial datasets, fetched and cached on demand |
| **[PIASO-for-agents](https://github.com/genecell/PIASO-for-agents)** | the ecosystem in a form coding agents can read |
| **[LARIS](https://github.com/genecell/LARIS)** · **[Emergene](https://github.com/genecell/Emergene)** | spatial ligand–receptor analysis · per-cell differential analysis across conditions |

### Documentation

**[piaso.org](https://piaso.org)** — tutorials, API reference and release notes.

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

Any model with web access can be pointed straight at:

```
https://piaso.org/llms.txt
https://piaso.org/llms-full.txt
```

### Contributing

Issues and pull requests are welcome at
[github.com/genecell/PIASO](https://github.com/genecell/PIASO/issues). Bug
reports are most useful with the output of `piaso.__version__` and a minimal
example.

### Citation

If PIASO is useful for your research, please consider citing Wu, S.J., Dai, M. *et al*. Pyramidal neurons proportionately alter cortical interneuron subtypes. *Nature* (2026). https://doi.org/10.1038/s41586-025-09996-8

### Contact
Min Dai
dai@broadinstitute.org
