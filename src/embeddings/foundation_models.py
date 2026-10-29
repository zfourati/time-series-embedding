import numpy as np
import math
import torch
import torch.nn.functional as F
from mantis.architecture import Mantis8M
from mantis.architecture import MantisV2   
from mantis.trainer import MantisTrainer
from sklearn.utils.class_weight import compute_class_weight

from sklearn.preprocessing import StandardScaler


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINTS = {
    "mantis":      (Mantis8M, "paris-noah/Mantis-8M", None),
    "mantis_plus": (Mantis8M, "paris-noah/MantisPlus", None),
    "utica":       (Mantis8M, "fegounna/Utica", "3cff4f954191b5bf9839b7a41117e2b24e7693ab"),
    "mantis_v2":   (MantisV2, "paris-noah/MantisV2", None),
    'Mantis-8M_pretrain_continued_7' : (Mantis8M, "mantis_checkpoints/mantis_7d_continued.pt", None),
    'MantisPlus_pretrain_continued_7' : (Mantis8M, "mantis_checkpoints/mantis_7d_continued.pt", None),
    'Mantis-Utica_pretrain_continued_7' : (Mantis8M, "mantis_checkpoints/utica_7d_continued.pt", None),
    'MantisV2_pretrain_continued_7' : (MantisV2, "mantis_checkpoints/mantis_v2_7d_continued.pt", None),
    'Mantis-8M_pretrain_scratch_7' : (Mantis8M, "mantis_checkpoints/mantis_7d_scratch.pt", None),
    'MantisV2_pretrain_scratch_7' : (MantisV2, "mantis_checkpoints/mantis_v2_7d_scratch.pt", None),
    'Mantis-8M_pretrain_continued_14' : (Mantis8M, "mantis_checkpoints/mantis_14d_continued.pt", None),
    'MantisPlus_pretrain_continued_14' : (Mantis8M, "mantis_checkpoints/mantis_14d_continued.pt", None),
    'Mantis-Utica_pretrain_continued_14' : (Mantis8M, "mantis_checkpoints/utica_14d_continued.pt", None),
    'MantisV2_pretrain_continued_14' : (MantisV2, "mantis_checkpoints/mantis_v2_14d_continued.pt", None),
    'Mantis-8M_pretrain_scratch_14' : (Mantis8M, "mantis_checkpoints/mantis_14d_scratch.pt", None),
    'MantisV2_pretrain_scratch_14' : (MantisV2, "mantis_checkpoints/mantis_v2_14d_scratch.pt", None),
    'Mantis-8M_pretrain_continued_30' : (Mantis8M, "mantis_checkpoints/mantis_30d_continued.pt", None),
    'MantisPlus_pretrain_continued_30' : (Mantis8M, "mantis_checkpoints/mantis_30d_continued.pt", None),
    'Mantis-Utica_pretrain_continued_30' : (Mantis8M, "mantis_checkpoints/utica_30d_continued.pt", None),
    'MantisV2_pretrain_continued_30' : (MantisV2, "mantis_checkpoints/mantis_v2_30d_continued.pt", None),
    'Mantis-8M_pretrain_scratch_30' : (Mantis8M, "mantis_checkpoints/mantis_30d_scratch.pt", None),
    'MantisV2_pretrain_scratch_30' : (MantisV2, "mantis_checkpoints/mantis_v2_30d_scratch.pt", None)
}

def to_mantis_input(X, target_len=512):
    X = torch.tensor(np.asarray(X, dtype=np.float32)).unsqueeze(1)
    return F.interpolate(X, size=target_len, mode="linear", align_corners=False).numpy()

def build_trainer(name):
    cls, ckpt, rev = CHECKPOINTS[name]
    net = cls(device=device)
    if ckpt.endswith(".pt"):
        state = torch.load(ckpt, map_location=device)
        net.load_state_dict(state)          # if this fails on keys, inspect `state.keys()`
    else:
        net = net.from_pretrained(ckpt, revision=rev) if rev else net.from_pretrained(ckpt)
    return MantisTrainer(device=device, network=net)

def mantis_embedding(name, X_train, X_val, X_test, target_len=512, batch_size=64):
    trainer = build_trainer(name)
    with torch.no_grad():
        train_embed = trainer.transform(to_mantis_input(X_train, target_len), batch_size=batch_size)
        val_embed = trainer.transform(to_mantis_input(X_val, target_len), batch_size=batch_size)
        test_embed = trainer.transform(to_mantis_input(X_test, target_len), batch_size=batch_size)
    del trainer
    torch.cuda.empty_cache() 
    return train_embed, val_embed, test_embed

from chronos import BaseChronosPipeline

CHRONOS_CHECKPOINTS = {
    'Chronos-T5-small':  'amazon/chronos-t5-small',
    'Chronos-Bolt-base': 'amazon/chronos-bolt-base',
}

def chronos_embedding(ckpt, X_train, X_val, X_test, batch_size=64):
    pipe = BaseChronosPipeline.from_pretrained(
        ckpt, device_map=str(device), torch_dtype=torch.float32)

    def embed(X):
        X = torch.tensor(np.asarray(X), dtype=torch.float32)       # (N, T)
        out = []
        for i in range(0, len(X), batch_size):
            emb, _ = pipe.embed(X[i:i + batch_size])               # (B, tokens, d_model)
            out.append(emb.float().mean(dim=1).cpu().numpy())      # mean-pool tokens
        return np.concatenate(out)

    return embed(X_train), embed(X_val), embed(X_test)

from transformers import CLIPVisionModel

TIVIT_CHECKPOINTS = {
    'TiViT-CLIP-B16': ('openai/clip-vit-base-patch16', 7),   # (checkpoint, layer)
    'TiViT-CLIP-H14': ('laion/CLIP-ViT-H-14-laion2B-s32B-b79K', 14),  # layer used in the repo's example
}

CLIP_MEAN = torch.tensor([0.4815, 0.4578, 0.4082]).view(1, 3, 1, 1)
CLIP_STD  = torch.tensor([0.2686, 0.2613, 0.2758]).view(1, 3, 1, 1)

def series_to_image(X, img_size=224):
    X = torch.tensor(np.asarray(X), dtype=torch.float32)            # (N, T)
    N, T = X.shape
    seg = math.ceil(math.sqrt(T))                                   # segment length ~ sqrt(T)
    rows = math.ceil(T / seg)
    X = F.pad(X, (0, rows * seg - T), mode='replicate')             # pad to a full grid
    img = X.view(N, 1, rows, seg)                                   # rows = stacked segments
    mn = img.amin(dim=(2, 3), keepdim=True)
    mx = img.amax(dim=(2, 3), keepdim=True)
    img = (img - mn) / (mx - mn + 1e-8)                             # per-sample min-max to [0, 1]
    img = F.interpolate(img, size=(img_size, img_size), mode='bilinear', align_corners=False)
    img = img.repeat(1, 3, 1, 1)                                    # grayscale -> 3 channels
    return (img - CLIP_MEAN) / CLIP_STD

@torch.no_grad()
def tivit_embedding(ckpt, layer, X_train, X_val, X_test, batch_size=64):
    vit = CLIPVisionModel.from_pretrained(ckpt).to(device).eval()

    def embed(X):
        imgs = series_to_image(X)
        out = []
        for i in range(0, len(imgs), batch_size):
            batch = imgs[i:i + batch_size].to(device)
            hs = vit(pixel_values=batch, output_hidden_states=True).hidden_states[layer]
            out.append(hs.mean(dim=1).cpu().numpy())                # mean over tokens
        return np.concatenate(out)

    return embed(X_train), embed(X_val), embed(X_test)


def concat_embeddings(blocks):
    """
    blocks: list of (train, val, test) embedding tuples.
    Each block is standardized using train statistics only, then blocks are
    concatenated along the feature axis. Returns (train, val, test).
    """
    parts = ([], [], [])
    for tr, va, te in blocks:
        scaler = StandardScaler().fit(tr)
        for part, Z in zip(parts, (tr, va, te)):
            part.append(scaler.transform(Z))
    return tuple(np.concatenate(p, axis=1) for p in parts)


import sys
sys.path.insert(0, '/home/zeineb/ts2vec')
from ts2vec import TS2Vec


def ts2vec_embedding(X_train, X_val, X_test, output_dims=320,
                     n_epochs=None, n_iters=None, batch_size=16, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    def prep(X):                                    # (N, T) -> (N, T, 1)
        return np.asarray(X, dtype=np.float32)[:, :, None]

    model = TS2Vec(input_dims=1, output_dims=output_dims,
                   device=device, batch_size=batch_size)
    model.fit(prep(X_train), n_epochs=n_epochs, n_iters=n_iters, verbose=False)

    def enc(X):                                     # (N, output_dims)
        return model.encode(prep(X), encoding_window='full_series')

    return enc(X_train), enc(X_val), enc(X_test)