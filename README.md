
# Higashi: Multiscale and integrative scHi-C analysis
<img src="https://github.com/ma-compbio/Higashi/blob/main/figs/logo2.png" align="right"
     alt="logo" width="290">

https://doi.org/10.1038/s41587-021-01034-y

As a computational framework for scHi-C analysis, Higashi has the following features:

-  Higashi represents the scHi-C dataset as a **hypergraph**
     - Each cell and each genomic bin are represented as the cell node and the genomic bin node.
     - Each non-zero entry in the single-cell contact map is modeled as a hyperedge. 
     - The read count for each chromatin interaction is used as the attribute of the hyperedge. 
- Higashi uses a **hypergraph neural network** to unveil high-order interaction patterns within this constructed hypergraph.
- Higashi can produce the **embeddings** for the scHi-C for downstream analysis.
-  Higashi can **impute single-cell Hi-C contact maps**, enabling detailed characterization of 3D genome features such as **TAD-like domain boundaries** and **A/B compartment scores** at single-cell resolution.

--------------------------

![figs/Overview.png](https://github.com/ma-compbio/Higashi/blob/main/figs/short_overview.png)

# Installation

We now have Fast-Higashi on conda.
`conda install -c ruochiz fasthigashi`

This repository is now configured for `uv`. From a clean checkout:

```{bash}
git clone https://github.com/ma-compbio/Higashi/
cd Higashi
uv sync --python 3.12
```

The default `uv` environment is pinned to the currently working `fasthigashi` micromamba stack that has been verified with this repository:

- Python 3.12
- numpy 2.4.3
- scipy 1.16.3
- pandas 2.3.3
- scikit-learn 1.8.0
- torch 2.11.0 (verified in `fasthigashi` as the `2.11.0+cu126` wheel)
- cooler 0.10.4
- h5py 3.16.0
- matplotlib 3.10.8
- seaborn 0.13.2
- umap-learn 0.5.11

Optional extras:

```{bash}
uv sync --extra gpu
uv sync --extra vis
```

- `gpu` adds the verified CuPy wheel (`cupy-cuda13x==14.0.1`)
- `vis` adds the optional `Higashi_vis` stack (`bokeh`, `cachetools`, `cmocean`)

Most legacy entry scripts still assume they are launched from the repository root. For example:

```{bash}
uv run python higashi/Process.py -c ./config.JSON
uv run python higashi/main_cell.py -c ../config_dir/config_ramani.JSON
```

# Documentation
Please see [the wiki](https://github.com/ma-compbio/Higashi/wiki) for extensive documentation and example tutorials.

Higashi is constantly being updated, see [change log](https://github.com/ma-compbio/Higashi/wiki/Change-Log) for the updating history

# Tutorial
- Higashi on [4DN sci-Hi-C (Kim et al.)](https://github.com/ma-compbio/Higashi/blob/main/tutorials/4DN_sci-Hi-C_Kim%20et%20al.ipynb)
- Higashi on [Ramani et al.](https://github.com/ma-compbio/Higashi/blob/main/tutorials/Ramani%20et%20al.ipynb)
- Fast-Higashi on [Lee et al.](https://github.com/ma-compbio/Fast-Higashi/blob/main/PFC%20tutorial.ipynb)

# Cite

Cite our paper by

```
@article {Zhang2020multiscale,
	author = {Zhang, Ruochi and Zhou, Tianming and Ma, Jian},
	title = {Multiscale and integrative single-cell Hi-C analysis with Higashi},
	year={2021},
	publisher = {Nature Publishing Group},
	journal = {Nature biotechnology}
}
```

![figs/Overview.png](https://github.com/ma-compbio/Higashi/blob/main/figs/higashi_title.png)

# See also

Fast-Higashi for more efficient and robust scHi-C embeddings
https://www.cell.com/cell-systems/fulltext/S2405-4712(22)00395-7

https://github.com/ma-compbio/Fast-Higashi

# Contact

Please contact ruochiz@andrew.cmu.edu or raise an issue in the github repo with any questions about installation or usage. 
