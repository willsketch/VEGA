import os
import json
import numpy as np
import pandas as pd
import anndata as ad
import torch
import torch.nn.functional as F
import warnings
import argparse


import vega
from dataset_configs import DATASETS
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

# -----------------------------
# parse arguments
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Run VEGA experiments")
    parser.add_argument("--datasets", type=str, nargs="+", default=None,
                        help="Dataset names to run e.g. --datasets TCGA SEAAD. Defaults to all in dataset_configs.py.")
    parser.add_argument("--gmt_files", type=str, nargs="+", default=["data/reactomes_uniprot.gmt"],
                        help="One or more paths to GMT files")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2,3,4,5,6],
                        help="List of run seeds to execute")
    parser.add_argument("--n_epochs", type=int, default=300,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Training batch size")
    parser.add_argument("--train_size", type=float, default=0.82,
                        help="Fraction of training data used inside VEGA")
    # add_nodes: extra fully-connected latent dims appended to the GMT mask (named UNANNOTATED_N).
    # They capture variance from genes not annotated to any pathway. Keep at 1.
    parser.add_argument("--add_nodes", type=int, default=1,
                        help="Number of fully-connected unannotated latent nodes added to the GMT mask")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing model directories")
    return parser.parse_args()

# -----------------------------
# data util functions
# -----------------------------
def load_split(expr_path, split_ids_path, obs_metadata_cols, id_col, nan_cols, donor_col=None):
    """
    Load subset of expression dataframe + metadata based on IDs in split_ids_path.

    For TCGA: split files contain patient_ids → filter by id_col directly.
    For SEAAD: split files contain donor IDs → filter by donor_col to get all
               cells from those donors, then index metadata by id_col (sample_id).

    Returns:
      expr_df: rows=samples, cols=genes (no metadata columns)
      metadata_df: indexed by id_col
    """
    df = pd.read_csv(expr_path)
    ids = np.load(split_ids_path, allow_pickle=True)

    filter_col = donor_col if donor_col is not None else id_col
    df = df[df[filter_col].isin(ids)]
    df = df.dropna(subset=nan_cols)

    metadata = df[obs_metadata_cols].copy().set_index(id_col)
    expr_df = df.drop(columns=obs_metadata_cols)

    return expr_df, metadata


def build_adata(expr_df, metadata_df):
    """
    Build AnnData with var index = gene IDs (columns).
    """
    gene_ids = expr_df.columns
    var = pd.DataFrame({"gene_id": gene_ids}).set_index("gene_id")

    adata = ad.AnnData(X=expr_df.values, obs=metadata_df, var=var)
    adata.obs_names_make_unique()
    return adata


# concatenate train and val samples
def build_adata_train(expr_path, train_ids_path, val_ids_path, obs_metadata_cols, id_col, nan_cols, donor_col=None):
    """
    Load train + val, concatenate, return adata_train
    """
    train_expr_df, train_meta = load_split(expr_path, train_ids_path, obs_metadata_cols, id_col, nan_cols, donor_col)
    val_expr_df, val_meta = load_split(expr_path, val_ids_path, obs_metadata_cols, id_col, nan_cols, donor_col)

    expr_df = pd.concat([train_expr_df, val_expr_df], axis=0)
    meta_df = pd.concat([train_meta, val_meta], axis=0)

    return build_adata(expr_df, meta_df)


# -----------------------------
# Mask randomization functions
# -----------------------------
def randomize_mask(mask, seed=None):
    """
    Randomly redistribute all edges in the mask, preserving only total edge count.
    """
    rng = np.random.default_rng(seed)
    n_rows, n_cols = mask.shape
    n_edges = int(mask.sum())
    rand_mask = np.zeros_like(mask)
    flat_indices = rng.choice(n_rows * n_cols, size=n_edges, replace=False)
    rand_mask.flat[flat_indices] = 1
    return rand_mask


def degree_preserving_mask(mask, Q=100, seed=None):
    """
    Randomize mask via edge swaps (Espinoza 2012), preserving row and column sums.
    Q controls number of swaps: n_swaps = Q * n_edges.
    """
    rng = np.random.default_rng(seed)
    m = mask.copy()
    edge_rows, edge_cols = np.where(m)
    edges = [[int(r), int(c)] for r, c in zip(edge_rows, edge_cols)]
    n_edges = len(edges)
    n_swaps = Q * n_edges
    successful = 0

    while successful < n_swaps:
        i, j = rng.choice(n_edges, size=2, replace=False)
        rA, cX = edges[i]
        rB, cY = edges[j]

        if rA == rB or cX == cY:
            continue
        if m[rA, cY] or m[rB, cX]:
            continue

        m[rA, cX] = 0
        m[rA, cY] = 1
        m[rB, cY] = 0
        m[rB, cX] = 1
        edges[i][1] = cY
        edges[j][1] = cX
        successful += 1

    return m


# -----------------------------
# Model util functions
# -----------------------------
def train_and_save_model(
    adata_train,
    save_dir_path,
    device,
    batch_size=128,
    train_size=0.82,
    n_epochs=300,
    positive_decoder=True,
    overwrite=True,
):
    """
    Train a fresh VEGA model and save it.
    The mask must already be set in adata_train.uns['_vega']['mask']
    before calling this function.
    Returns trained model in eval mode on `device`.
    """
    model = vega.VEGA(
        adata_train,
        positive_decoder=positive_decoder,
        use_cuda=(device.type == "cuda"),
    )

    model.train_vega(n_epochs=n_epochs, batch_size=batch_size, train_size=train_size, use_gpu=(device.type == "cuda"))

    model.save(path=save_dir_path, save_adata=True, save_history=True, overwrite=overwrite)

    model.eval()
    return model


def load_model(save_dir_path, device):
    """
    Load VEGA model and ensure it lives on device.
    """
    model = vega.VEGA.load(path=save_dir_path, device=torch.device(device))
    model = model.to(device)
    if hasattr(model, "use_cuda"):
        model.use_cuda = (device.type == "cuda")
    model.eval()
    return model


# -----------------------------
# Embedding util functions
# -----------------------------
def save_embeddings(model, expr_df, sample_ids, gmv_names, save_path, device):
    """
    Encode expr_df with the trained model and save μ (the mean embedding) as CSV.
    Shape: (cells × n_GMVs). Columns = GMV names, index = sample_ids.

    Parameters
    ----------
    model       : trained VEGA model in eval mode
    expr_df     : DataFrame (samples × genes)
    sample_ids  : index to assign to the output DataFrame
    gmv_names   : list of GMV names (column labels)
    save_path   : full path including filename, without extension (.csv appended)
    device      : torch device
    """
    X = torch.tensor(expr_df.values, device=device, dtype=torch.float32)
    with torch.no_grad():
        z, _, _ = model.encode(X, batch_index=None)
    z_np = z.detach().cpu().numpy()
    df = pd.DataFrame(z_np, index=sample_ids, columns=gmv_names)
    df.to_csv(save_path + '.csv')
    print(f"  Saved embeddings → {save_path}.csv  ({df.shape[0]} cells × {df.shape[1]} GMVs)")


# -----------------------------
# Evaluation util functions
# -----------------------------
def evaluate_model(model, expr_df, labels, device, drop_genes=None):
    """
    Evaluate on a matrix (expr_df) and labels.
    Returns dict with NMI, ARI, Silhouette, mse
    """
    X = torch.tensor(expr_df.values, device=device, dtype=torch.float32, requires_grad=False)

    with torch.no_grad():
        Z, mu, logvar = model.encode(X, batch_index=None)
        X_rec = model.decode(Z, batch_index=None)

    if drop_genes is not None and len(drop_genes) > 0:
        gene_mask = ~expr_df.columns.isin(drop_genes)
        # keep_idx = torch.tensor(gene_mask, device=device)

        mse = F.mse_loss(X_rec[:, gene_mask], X[:, gene_mask], reduction="mean").item()
    else:
        mse = F.mse_loss(X_rec, X, reduction="mean").item()

    # clustering on CPU numpy
    Z_np = Z.detach().cpu().numpy()
    n_clusters = int(labels.nunique())

    # if only one cluster exists in labels, set clustering metrics to Nan
    if n_clusters <= 1:
        warnings.warn(f"{n_clusters} unique label found, clustering metrics will be nan", RuntimeWarning)
        return {
            "NMI": float("nan"),
            "ARI": float("nan"),
            "Silhouette": float("nan"),
            "mse": mse,
        }

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_assignments = kmeans.fit_predict(Z_np)

    nmi = normalized_mutual_info_score(labels, cluster_assignments)
    ari = adjusted_rand_score(labels, cluster_assignments)
    sil = silhouette_score(Z_np, cluster_assignments)

    return {
        "NMI": float(nmi),
        "ARI": float(ari),
        "Silhouette": float(sil),
        "mse": float(mse),
    }


# -----------------------------
# main function for 1 run
# -----------------------------
def run_experiment(run_seed, dataset_cfg, gmt_files, device, n_epochs=300, batch_size=128,
                   train_size=0.82, overwrite=True, add_nodes=1, positive_decoder=True):

    split_dir         = dataset_cfg['split_dir']
    expr_path         = dataset_cfg['expr_data_path']
    obs_metadata_cols = dataset_cfg['obs_metadata_cols']
    id_col            = dataset_cfg['id_col']
    label_col         = dataset_cfg['label_col']
    nan_cols          = dataset_cfg['nan_cols']
    donor_col         = dataset_cfg['donor_col']
    drop_genes        = dataset_cfg['drop_genes']

    print(f"\n========== Running seed {run_seed} ==========")

    run_folder = os.path.join(split_dir, f'run_seed-{run_seed}')
    train_ids_path = os.path.join(run_folder, "train_ids.npy")
    val_ids_path = os.path.join(run_folder, "val_ids.npy")
    test_ids_path = os.path.join(run_folder, "test_ids.npy")

    print("→ Building training AnnData...")
    adata_train = build_adata_train(
        expr_path=expr_path,
        train_ids_path=train_ids_path,
        val_ids_path=val_ids_path,
        obs_metadata_cols=obs_metadata_cols,
        id_col=id_col,
        nan_cols=nan_cols,
        donor_col=donor_col,
    )

    # load test set once (shared across all GMTs and conditions)
    test_expr_df, test_meta = load_split(expr_path, test_ids_path, obs_metadata_cols, id_col, nan_cols, donor_col)
    labels_test = test_meta[label_col]
    train_expr_df = pd.DataFrame(adata_train.X, index=adata_train.obs_names, columns=adata_train.var_names)
    labels_train = adata_train.obs[label_col]

    seed_metrics = {}

    for gmt_path in gmt_files:
        gmt_name = os.path.splitext(os.path.basename(gmt_path))[0]
        print(f"\n  ===== GMT: {gmt_name} =====")

        # reset vega setup for each GMT using setup_anndata (otherwise create_mask in utils.py errors if mask already exists)
        vega.utils.setup_anndata(adata_train)
        vega.utils.create_mask(adata_train, gmt_path, add_nodes=add_nodes)
        true_mask = adata_train.uns['_vega']['mask'].copy()
        gmv_names = list(adata_train.uns['_vega']['gmv_names'])

        masks = {
            'true': true_mask,
            'random': randomize_mask(true_mask, seed=run_seed),
            'degree_preserving': degree_preserving_mask(true_mask, seed=run_seed),
        }

        gmt_metrics = {}

        for condition, mask in masks.items():
            print(f"\n  --- Condition: {condition} ---")

            adata_train.uns['_vega']['mask'] = mask

            save_dir_path = os.path.join(run_folder, f"best_model_{condition}_{gmt_name}")
            model_path = os.path.join(save_dir_path, "vega_params.pt")

            if not os.path.exists(model_path):
                print(f"  → Training {condition} model...")
                model = train_and_save_model(
                    adata_train=adata_train,
                    save_dir_path=save_dir_path,
                    device=device,
                    batch_size=batch_size,
                    train_size=train_size,
                    n_epochs=n_epochs,
                    overwrite=overwrite,
                    positive_decoder=positive_decoder,
                )
                print(f"  ✓ Training complete.")
            else:
                print(f"  → Loading existing {condition} model...")
                model = load_model(save_dir_path, device=device)
                print(f"  ✓ Model loaded.")

            print(f"  → Evaluating...")
            train_metrics = evaluate_model(model, train_expr_df, labels_train, device=device, drop_genes=drop_genes)
            test_metrics = evaluate_model(model, test_expr_df, labels_test, device=device, drop_genes=drop_genes)
            print(f"  Train NMI: {train_metrics['NMI']:.4f} | Test NMI: {test_metrics['NMI']:.4f}")

            print(f"  → Saving embeddings...")
            save_embeddings(model, train_expr_df, adata_train.obs_names, gmv_names,
                            os.path.join(run_folder, f"z_train_{condition}_{gmt_name}"), device)
            save_embeddings(model, test_expr_df, test_expr_df.index, gmv_names,
                            os.path.join(run_folder, f"z_test_{condition}_{gmt_name}"), device)

            gmt_metrics[condition] = {"train": train_metrics, "test": test_metrics}

        seed_metrics[gmt_name] = gmt_metrics

    print(f"\n✓ Seed {run_seed} done.")
    return seed_metrics



if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gmt_files = args.gmt_files
    overwrite = args.overwrite
    add_nodes = args.add_nodes
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    train_size = args.train_size
    seeds = args.seeds

    active_datasets = list(DATASETS.values())
    if args.datasets is not None:
        active_datasets = [DATASETS[n] for n in args.datasets if n in DATASETS]
        if not active_datasets:
            raise ValueError(f"No matching datasets found. Available: {list(DATASETS.keys())}")

    for dataset_cfg in active_datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_cfg['name']}")
        print(f"{'='*60}")

        out_file = os.path.join(dataset_cfg['split_dir'], "metrics.txt")

        with open(out_file, "w") as f:
            for s in seeds:
                seed_metrics = run_experiment(
                    run_seed=s,
                    dataset_cfg=dataset_cfg,
                    gmt_files=gmt_files,
                    device=device,
                    n_epochs=n_epochs,
                    batch_size=batch_size,
                    train_size=train_size,
                    overwrite=overwrite,
                    add_nodes=add_nodes,
                    positive_decoder=True,
                )

                line = f"run-{s}: {seed_metrics}\n"
                f.write(line)
                f.flush()

        print(f"\n✓ Metrics saved to {out_file}")






























