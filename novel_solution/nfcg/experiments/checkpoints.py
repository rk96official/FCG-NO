import os

from fcgno.training.checkpoint import load_model

TAG_SUFFIX = {"paper": "", "notebook": "_notebook", "transfer": ""}
TRANSFER_TRAINING_SET = 100


# fcgno checkpoint name of an SNO trained with the Notay loss on Krylov residuals: paper or notebook settings, or the cross-grid models (N_train = 100)
def sno_checkpoint_name(kind, grid, seed, tag="paper"):
    n_train = TRANSFER_TRAINING_SET if tag == "transfer" else grid
    return f"sno_{kind}_{grid}_notay_krylov_n{n_train}_upstream_seed{seed}{TAG_SUFFIX[tag]}"


# fcgno checkpoint name of any reproduction preconditioner trained with the Notay loss on Krylov residuals (sno with paper or notebook settings, fno, unet, dilresnet)
def reproduction_checkpoint_name(architecture, kind, grid, seed, tag="paper"):
    if architecture == "sno":
        return sno_checkpoint_name(kind, grid, seed, tag)
    return f"{architecture}_{kind}_{grid}_notay_krylov_n{grid}_upstream_seed{seed}{TAG_SUFFIX[tag]}"


# Existing fcgno checkpoint as (model, spec); fails loudly when it is missing
def load_checkpoint(checkpoint_dir, name):
    loaded = load_model(os.path.join(checkpoint_dir, name))
    if loaded is None:
        raise FileNotFoundError(f"missing checkpoint {name} in {checkpoint_dir}")
    return loaded
