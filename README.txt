HECTOR Pro — Multi-Species Single Cell MODEL
================================================

HECTOR Pro is an extended variant of the HECTOR architecture, inspired by
TranscriptFormer and UCE. It replaces gene embeddings with protein embeddings
and is trained on large-scale public data from CELLxGENE and scBaseCount(over 30 Million cells).

The model follows the same simple usage pattern as the original HECTOR, but
is significantly larger and therefore runs more slowly.


SUPPORTED SPECIES
-----------------
The production checkpoint supports six species. Both the Latin name and
common-name aliases are accepted as the species argument:

  Latin name           Common name
  -------------------  -------------
  homo_sapiens         human
  mus_musculus         mouse
  macaca_mulatta       macaque
  callithrix_jacchus   marmoset
  pan_troglodytes      chimpanzee
  danio_rerio          zebrafish

Use one species per AnnData object.


INSTALLATION
------------
1. Set up the environment:

    conda create -n hector-pro -c conda-forge python=3.12 pip -y
    conda activate hector-pro
    python -m pip install scanpy pyarrow scikit-image networkx matplotlib plotly igraph leidenalg

Then install the backend for your machine, only choose one backend:

    Linux / CPU:
        python -m pip install "tensorflow==2.20.*"

    Linux with an NVIDIA GPU (requires an NVIDIA driver):
        python -m pip install "tensorflow[and-cuda]==2.20.*"

    Apple Silicon Mac:
        python -m pip install mlx

HECTOR automatically uses MLX on macOS when available;
otherwise it uses TensorFlow. You can also use an existing environment
with these dependencies already installed.

2. Install HECTOR Pro:

Download or clone the GitHub repository, then open a terminal in the
folder containing pyproject.toml and hector_pro/. With the environment
above activated, run:

    python -m pip install ".[extras]"

The model downloads automatically on first use. To use a local model
instead, see the example below.


QUICK START
-----------
Keep raw counts in adata.X and retain the measured genes. HECTOR Pro
performs gene alignment and normalization internally — do not normalize
or log-transform the data beforehand.

    import scanpy as sc
    import hector_pro

    species  = "homo_sapiens"
    adata    = sc.read_h5ad("input_counts.h5ad")

    predictor = hector_pro.HECTOR(species=species)

    hector_pro.filter_cells(adata, species=species)       # optional QC
    hector_pro.map_gene_symbols(adata, species=species)   # gene alignment

    predictions = predictor.predict(adata, label_format="name")
    predictor.write_predictions(adata, predictions)

    adata.write_h5ad("annotated.h5ad")

On first run, HECTOR Pro automatically downloads the model from Hugging Face
and caches it locally at:
    ~/.cache/hector_pro/models/multi_species_model.h5

To use a local checkpoint instead of downloading the model:

    predictor = hector_pro.HECTOR(
        model_path="/path/to/your/model.h5",
        species=species,
        auto_download=False,
    )


KNOWN LIMITATIONS & RECOMMENDATIONS
-------------------------------------
- This model was developed as a personal experiment to explore multi-species
  cell-type annotation. It is not an official or production-grade release.
- It will perform worse than the dedicated single-species HECTOR models on
  human or mouse data. For human or mouse datasets, the single-species
  models are strongly recommended.
- Cross-species transfer to species not listed above has not been validated
  and is not currently supported.
- Because the model is larger than the original HECTOR, inference is slower.
  Plan accordingly for large datasets.
