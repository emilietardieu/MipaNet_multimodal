import numpy as np
import torch
import torch.utils.data as data

class BaseDataset(data.Dataset):
    """
    Classe de base pour les datasets multi-sources.
        :param str root: Chemin racine du dataset.
        :param str mode: Type de division (train, val, test).
        :param dict sources: Dict des sources avec leurs métadonnées.
        :param dict source_transforms: Dict {nom_source: transform} pour la normalisation.
    """
    def __init__(self, root, mode, sources=None, source_transforms=None):
        self.root = root
        self.mode = mode
        self.sources = sources or {}
        self.source_transforms = source_transforms or {}

    @property
    def num_class(self):
        return self.NUM_CLASS

    def _target_transform(self, mask):
        mask = torch.from_numpy(np.array(mask)).long()
        return mask
