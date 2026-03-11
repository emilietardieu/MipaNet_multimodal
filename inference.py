"""
Inférence du modèle MIPANet multi-branches sur de nouveaux territoires.
Pas de données de référence nécessaires — uniquement les sources d'entrée.

Structure attendue du répertoire root :
    root/
      IRC/    *.tif   (si IRC utilisé)
      MNH/    *.tif   (si MNH utilisé)
      BIOM/   *.tif   (si biomasse utilisé)
      ...
      predict.txt     (optionnel — liste des stems à traiter)

Si predict.txt est absent, tous les fichiers du premier répertoire source
sont utilisés automatiquement.

Sorties :
    output_dir/
      predictions/   {stem}.tif              — masque de classes géoréférencé (uint8 : 0/1/2)
      heatmaps/      {stem}_heatmap.tif      — probabilité vieille forêt géoréférencée (float32)
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils import data
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import rasterio

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from config import get_config
from model.datasets.transforms import build_source_transforms, get_normalized_mean_std
from model.model import get_mipanet


# ---------------------------------------------------------------------------
# Dataset d'inférence (sans masque)
# ---------------------------------------------------------------------------

class InferenceDataset(data.Dataset):
    """
    Dataset pour l'inférence sans données de référence.

    Lit les imagettes depuis les répertoires sources définis dans la config.
    Scanne automatiquement la première source, ou lit predict.txt si présent.
    """

    def __init__(self, root, sources, source_transforms):
        self.root = Path(root)
        self.sources = sources
        self.source_transforms = source_transforms
        self.source_names = list(sources.keys())

        first_name = self.source_names[0]
        first_cfg  = self.sources[first_name]

        predict_txt = self.root / 'predict.txt'
        if predict_txt.exists():
            with open(predict_txt, 'r') as f:
                candidates = [line.strip() for line in f if line.strip()]
            print(f"Fichier predict.txt trouvé — {len(candidates)} entrées")
        else:
            src_dir    = self.root / first_cfg['dir']
            ext        = first_cfg['ext']
            candidates = sorted([p.stem for p in src_dir.glob(f'*{ext}')])
            print(f"Scan de {src_dir} — {len(candidates)} fichier(s) {ext} trouvé(s)")

        # Vérifier la présence de toutes les sources pour chaque stem
        self.source_files = {name: [] for name in self.source_names}
        self.file_names   = []

        for stem in candidates:
            all_ok = True
            for src_name, cfg in self.sources.items():
                p = self.root / cfg['dir'] / f"{stem}{cfg['ext']}"
                if not p.exists():
                    print(f"  [manquant] {src_name}: {p} — ignoré")
                    all_ok = False
                    break
            if all_ok:
                self.file_names.append(stem)
                for src_name, cfg in self.sources.items():
                    self.source_files[src_name].append(
                        self.root / cfg['dir'] / f"{stem}{cfg['ext']}"
                    )

        print(f"Imagettes valides : {len(self.file_names)}\n")

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        source_tensors = {}
        for name in self.source_names:
            cfg       = self.sources[name]
            file_path = self.source_files[name][index]

            if cfg['channels'] == 1:
                try:
                    with rasterio.open(file_path) as src:
                        img = src.read(1).astype(np.float32)   # [H, W] float32
                except Exception:
                    img = Image.open(file_path)
            else:
                img = Image.open(file_path)

            source_tensors[name] = self.source_transforms[name](img)

        # On retourne aussi le chemin de la première source pour le géoréférencement
        ref_path = str(self.source_files[self.source_names[0]][index])
        return source_tensors, self.file_names[index], ref_path


# ---------------------------------------------------------------------------
# Inférencer principal
# ---------------------------------------------------------------------------

class Inferencer:
    """
    Lance l'inférence MIPANet sur un territoire sans vérité terrain.

    Produit :
      - un masque de classes GeoTIFF (uint8 : 0/1/2) géoréférencé par imagette
      - une carte de chaleur float32 GeoTIFF de probabilité vieille forêt par imagette
    """

    CLASS_COLORS = {0: (0, 0, 0), 1: (127, 127, 127), 2: (255, 255, 255)}

    def __init__(self, root, model_path, branches, output_dir=None):
        self.root       = Path(root)
        self.model_path = Path(model_path)
        self.config, _  = get_config()

        # Dépackage du format [branch_config, lr, ratio] ou branch_config seul
        if (isinstance(branches, (list, tuple))
                and len(branches) >= 2
                and isinstance(branches[1], float)):
            self.branches = branches[0]
        else:
            self.branches = branches

        self.sources      = self.config['sources']
        self.source_names = list(self.sources.keys())
        self.norm_stats   = get_normalized_mean_std(self.sources)

        source_transforms = build_source_transforms(self.sources)

        # Répertoires de sortie
        weights_name   = self.model_path.stem
        self.output_dir = Path(output_dir) if output_dir else self.root / f"inference_{weights_name}"
        self.pred_dir   = self.output_dir / "predictions"
        self.heat_dir   = self.output_dir / "heatmaps"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pred_dir.mkdir(exist_ok=True)
        self.heat_dir.mkdir(exist_ok=True)

        # Dataset et DataLoader
        self.dataset = InferenceDataset(self.root, self.sources, source_transforms)

        self.device = torch.device(
            "cuda:0" if self.config['testing']['use_cuda'] and torch.cuda.is_available() else "cpu"
        )
        kwargs = {'num_workers': self.config['testing']['workers'], 'pin_memory': True} \
            if self.device.type == 'cuda' else {}

        self.loader = data.DataLoader(
            self.dataset,
            batch_size = self.config['testing']['batch_size'],
            drop_last  = False,
            shuffle    = False,
            **kwargs,
        )

        # Modèle
        pass_rff   = [True] * len(self.branches)
        self.model = get_mipanet(
            dataset       = self.config['testing']['dataset'],
            branches      = self.branches,
            sources       = self.sources,
            pass_rff      = pass_rff,
            first_fusions = self.config['encoder']['first_fusions'],
            last_fusion   = self.config['encoder']['last_fusion'],
            use_tgcc      = self.config['TGCC']['use_TGCC'],
        )
        self._load_model()

        used = [src for branch in self.branches for src in branch]
        print(f"Modalités utilisées : {' // '.join(used)}")
        print(f"Device              : {self.device}")
        print(f"Sorties             : {self.output_dir}\n")

    # ------------------------------------------------------------------
    # Chargement des poids
    # ------------------------------------------------------------------

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modèle introuvable : {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
        else:
            state_dict = checkpoint

        try:
            self.model.load_state_dict(state_dict)
        except RuntimeError as e:
            if "Missing key(s)" in str(e) or "Unexpected key(s)" in str(e):
                model_dict  = self.model.state_dict()
                compatible  = {k: v for k, v in state_dict.items()
                               if k in model_dict and model_dict[k].shape == v.shape}
                model_dict.update(compatible)
                self.model.load_state_dict(model_dict)
                print(f"Chargement partiel : {len(compatible)}/{len(state_dict)} paramètres")
            else:
                raise

        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Utilitaires
    # ------------------------------------------------------------------

    def _sources_to_device(self, sources_dict):
        return {name: t.to(self.device) for name, t in sources_dict.items()}

    def _denormalize(self, tensor, source_name):
        stats = self.norm_stats[source_name]
        mean  = torch.tensor(stats['mean']).view(-1, 1, 1).to(tensor.device)
        std   = torch.tensor(stats['std']).view(-1, 1, 1).to(tensor.device)
        return tensor * std + mean

    def _colorize_mask(self, mask_np):
        h, w = mask_np.shape
        rgb  = np.zeros((h, w, 3), dtype=np.uint8)
        for cls, color in self.CLASS_COLORS.items():
            rgb[mask_np == cls] = color
        return rgb

    # ------------------------------------------------------------------
    # Sauvegarde des sorties
    # ------------------------------------------------------------------

    def _save_prediction(self, pred_classes_np, stem, ref_path):
        """Sauvegarde le masque de classes en GeoTIFF géoréférencé (uint8 : 0=non-forêt, 1=forêt, 2=vieille forêt)."""
        out_path = self.pred_dir / f"{stem}.tif"
        with rasterio.open(ref_path) as src:
            profile = src.profile.copy()
        profile.update(
            dtype   = rasterio.uint8,
            count   = 1,
            compress= 'lzw',
            nodata  = 255,
        )
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(pred_classes_np.astype(np.uint8), 1)

    def _save_heatmap(self, pred_logits, stem, ref_path):
        """Sauvegarde la probabilité vieille forêt en GeoTIFF géoréférencé (float32, valeurs 0–1)."""
        try:
            probs           = F.softmax(pred_logits, dim=0)           # [C, H, W]
            old_forest_prob = probs[2].cpu().numpy()                  # [H, W]
            old_forest_prob = np.nan_to_num(old_forest_prob, nan=0.0, posinf=1.0, neginf=0.0)

            out_path = self.heat_dir / f"{stem}_heatmap.tif"
            with rasterio.open(ref_path) as src:
                profile = src.profile.copy()
            profile.update(
                dtype   = rasterio.float32,
                count   = 1,
                compress= 'lzw',
                nodata  = -9999.0,
            )
            with rasterio.open(out_path, 'w', **profile) as dst:
                dst.write(old_forest_prob.astype(np.float32), 1)

        except Exception as e:
            print(f"Erreur heatmap {stem}: {e}")

    # ------------------------------------------------------------------
    # Point d'entrée principal
    # ------------------------------------------------------------------

    def run(self):
        """Lance l'inférence sur toutes les imagettes du dataset."""
        n = len(self.dataset)
        if n == 0:
            print("Aucune imagette à traiter. Vérifiez le répertoire root et la config sources.")
            return

        print("=============== Début de l'inférence ===============")

        with torch.no_grad():
            for sources_dict, file_names, ref_paths in tqdm(self.loader, desc="Inférence", total=len(self.loader)):
                sources_dict = self._sources_to_device(sources_dict)
                outputs      = self.model(sources_dict)
                pred_logits  = outputs[0]                       # [B, C, H, W]
                pred_classes = torch.argmax(pred_logits, dim=1) # [B, H, W]

                for b in range(pred_classes.shape[0]):
                    stem     = file_names[b] if isinstance(file_names, (list, tuple)) else file_names
                    ref_path = ref_paths[b]  if isinstance(ref_paths,  (list, tuple)) else ref_paths
                    self._save_prediction(pred_classes[b].cpu().numpy(), stem, ref_path)
                    self._save_heatmap(pred_logits[b], stem, ref_path)

        print("=============== Inférence terminée ===============")
        print(f"  {n} imagette(s) traitée(s)")
        print(f"  Prédictions → {self.pred_dir}")
        print(f"  Heatmaps    → {self.heat_dir}")
