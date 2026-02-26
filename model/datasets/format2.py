"""
==========
ForMat2 = Dataset Forêts Matures multi-sources
==========
Ce module définit la classe ForMat2, qui charge dynamiquement
les sources définies dans la config (IRC, MNH, biomasse, historique, etc.)
et retourne un dictionnaire de tenseurs.
"""
from PIL import Image
from pathlib import Path

from .base import BaseDataset
import rasterio
import numpy as np

class ForMat2(BaseDataset):
    """
    Classe pour gérer le dataset ForMat2 multi-sources.
    """
    NUM_CLASS = 3  # classe sol (0), classe forêt (1), classe vieille forêt (2)

    def __init__(self, root, mode, sources=None, source_transforms=None, augmented_transform=None):
        """
        Initialise le dataset ForMat2.
            :param str root: Chemin vers le répertoire racine du dataset.
            :param str mode: 'train', 'val', ou 'test'.
            :param dict sources: Dict des sources {nom: {dir, channels, ext, mean, std}}.
            :param dict source_transforms: Dict {nom_source: transform callable}.
            :param callable augmented_transform: Transformation d'augmentation unifiée.
        """
        super(ForMat2, self).__init__(root, mode, sources, source_transforms)

        self.augmented_transform = augmented_transform
        self.use_augmentation = augmented_transform is not None

        _format_root = Path(root)
        _mask_dir = _format_root / 'MASK'

        # Fichier de split
        if self.mode == 'train':
            _split_f = _format_root / 'train.txt'
        elif self.mode == 'val':
            _split_f = _format_root / 'val.txt'
        else:
            _split_f = _format_root / 'test.txt'

        # Construire les chemins pour chaque source
        self.source_names = list(self.sources.keys())
        self.source_files = {name: [] for name in self.source_names}
        self.masks = []
        self.file_names = []

        with open(_split_f, "r") as lines:
            for line in lines:
                line = line.strip()
                self.file_names.append(line)

                # Charger chaque source
                for name, cfg in self.sources.items():
                    src_dir = _format_root / cfg['dir']
                    src_file = src_dir / f"{line}{cfg['ext']}"
                    assert src_file.is_file(), f"Fichier introuvable: {src_file}"
                    self.source_files[name].append(src_file)

                _mask = _mask_dir / f"{line}.tif"
                assert _mask.is_file(), f"Masque introuvable: {_mask}"
                self.masks.append(_mask)

    def __getitem__(self, index):
        """
        Récupère un échantillon du dataset.
            :param int index: Index de l'échantillon.
            :return: (sources_dict, target, file_name)
                - sources_dict: Dict {nom_source: tensor}
                - target: Masque de segmentation
                - file_name: Nom du fichier
        """
        # Charger toutes les sources
        source_images = {}
        for name in self.source_names:
            src_cfg = self.sources[name]
            file_path = self.source_files[name][index]

            if src_cfg['channels'] == 1:
                # Pour les sources mono-canal, essayer rasterio d'abord (gère mieux les float32)
                try:
                    with rasterio.open(file_path) as src:
                        img = src.read(1).astype(np.float32)  # [H, W] float32
                except Exception:
                    img = Image.open(file_path)
            else:
                img = Image.open(file_path)

            source_images[name] = img

        _target = Image.open(self.masks[index])

        # Appliquer les transformations
        if self.use_augmentation:
            source_tensors, _target = self.augmented_transform(source_images, _target)
        else:
            _target = self._target_transform(_target)
            source_tensors = {}
            for name, img in source_images.items():
                if name in self.source_transforms:
                    source_tensors[name] = self.source_transforms[name](img)
                else:
                    raise ValueError(f"Pas de transform défini pour la source '{name}'")

        return source_tensors, _target, self.file_names[index]

    def __len__(self):
        return len(self.file_names)
