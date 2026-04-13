# VEGA Pipeline

Experimental pipeline for gene module activity analysis across biological datasets, built on top of [VEGA (VAE Enhanced by Gene Annotations)](https://github.com/LucasESBS/vega) by Lucas Seninge et al.

## Credits

The core VAE model and gene module masking logic are taken directly from the original VEGA package by [Lucas Seninge](https://github.com/LucasESBS). Please cite the original work if you use this pipeline:

> Seninge, L., Anastopoulos, I., Ding, H. & Stuart, J. VEGA is an interpretable generative model for inferring biological network activity in single-cell transcriptomics. *Nature Communications* 12, 5684 (2021). https://doi.org/10.1038/s41467-021-26017-0

## Overview

This repository extends the original VEGA package with am experimental pipeline that includes:

- Multiple GMT gene sets (Reactome, Hallmark, etc.)
- Three mask conditions per run: **true**, **random**, and **degree-preserving** controls
- Different Datasets
- Donor-level splitting for single-cell data (SEAAD)
- Per-seed latent embedding export (`z_train`, `z_test`) as CSV files

## Repository Structure

```
VEGA/
├── Vega-Runs.py              # Main experiment runner
├── dataset_configs.py        # Dataset configurations (TCGA, SEAAD)
├── data_splits.py            # Train/val/test splits for TCGA
├── data_splits_SEAAD.py      # Donor-level splits for SEAAD
├── GeneSymbol2UniprotGMT.py  # Convert gene symbols to UniProt GMT files
├── environment.yaml          # Conda environment for local runs
├── vega/                     # VEGA package (Seninge et al.)
└── data/                     # Gene sets and expression data
```

## Usage

### 1. Generate data splits

```bash
# TCGA
python data_splits.py

# SEAAD (donor-level splitting)
python data_splits_SEAAD.py
```

### 2. Run experiments

```bash
# Single dataset, multiple GMT files
python Vega-Runs.py --datasets TCGA --gmt_files data/reactomes_uniprot.gmt data/hallmark_v2026_1_Hs_uniprot.gmt

# All datasets
python Vega-Runs.py --gmt_files data/reactomes_uniprot.gmt data/hallmark_v2026_1_Hs_uniprot.gmt
```

Key arguments:
- `--datasets` — one or more dataset names (default: all in `dataset_configs.py`)
- `--gmt_files` — one or more GMT files to use as gene module sets
- `--seeds` — random seeds (default: 2 3 4 5 6)
- `--n_epochs` — number of training epochs (default: 300)
- `--batch_size` — batch size (default: 128)

### 3. Output structure

For each dataset and seed, the following files are produced:

```
splits_{DATASET}/
├── metrics.txt
└── run_seed-{N}/
    ├── best_model_{condition}_{gmt_name}/   # Saved model
    ├── z_train_{condition}_{gmt_name}.csv   # Train embeddings
    └── z_test_{condition}_{gmt_name}.csv    # Test embeddings
```

Conditions: `true`, `random`, `degree_preserving`.

## Adding a New Dataset

Add an entry to `dataset_configs.py`:

```python
MY_DATASET = {
    'name': 'MY_DATASET',
    'expr_data_path': 'data/my_data.csv',
    'split_dir': 'splits_MY_DATASET',
    'id_col': 'sample_id',
    'label_col': 'cell_type',
    'donor_col': None,           # set to donor column name if single-cell
    'nan_cols': ['cell_type'],
    'obs_metadata_cols': ['sample_id', 'cell_type'],
    'drop_genes': None,
}
```