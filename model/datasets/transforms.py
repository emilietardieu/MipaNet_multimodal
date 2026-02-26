"""
Transformations dynamiques pour le pipeline multi-sources.
Les mean/std sont lues depuis la config SOURCES.
"""
import torch
import torchvision.transforms as transform
import torchvision.transforms.functional as TF
import numpy as np
import random


def build_source_transforms(sources):
    """
    Construit un dict de transforms (ToTensor + Normalize) pour chaque source.
    Les mean/std sont normalisées pour des images en [0, 1] si les valeurs sont > 1.

        :param dict sources: Dict des sources {nom: {channels, mean, std, ...}}.
        :returns: Dict {nom_source: Compose transform}
    """
    source_transforms = {}
    for name, cfg in sources.items():
        mean = cfg['mean']
        std = cfg['std']

        # Normaliser mean/std si les valeurs sont dans l'échelle [0, 255]
        # (torchvision.ToTensor convertit les images PIL uint8 en [0, 1])
        norm_mean = [m / 255.0 if m > 1.0 else m for m in mean]
        norm_std = [s / 255.0 if s > 1.0 else s for s in std]

        source_transforms[name] = transform.Compose([
            transform.ToTensor(),
            transform.Lambda(lambda x: x.to(torch.float)),
            transform.Normalize(norm_mean, norm_std)
        ])

    return source_transforms


def get_normalized_mean_std(sources):
    """
    Retourne les mean/std normalisées pour chaque source (utile pour la dénormalisation).
        :param dict sources: Dict des sources.
        :returns: Dict {nom_source: {'mean': [...], 'std': [...]}}
    """
    result = {}
    for name, cfg in sources.items():
        mean = cfg['mean']
        std = cfg['std']
        norm_mean = [m / 255.0 if m > 1.0 else m for m in mean]
        norm_std = [s / 255.0 if s > 1.0 else s for s in std]
        result[name] = {'mean': norm_mean, 'std': norm_std}
    return result


class AugmentedTransform:
    """
    Classe pour appliquer des transformations d'augmentation cohérentes
    sur toutes les sources et les masques.

        :param dict sources: Dict des sources avec mean/std.
        :param str mode: 'train' pour l'augmentation, 'val'/'test' sans augmentation.
        :param float noise_std: Écart-type pour le bruit gaussien.
        :param float flip_prob: Probabilité d'appliquer les flips.
    """
    def __init__(self, sources, mode='train', noise_std=0.01, flip_prob=0.5):
        self.sources = sources
        self.mode = mode
        self.noise_std = noise_std
        self.flip_prob = flip_prob

        # Pré-calculer les mean/std normalisées
        self.norm_stats = get_normalized_mean_std(sources)

    def __call__(self, source_images, mask):
        """
        Applique les transformations sur toutes les sources et le masque.

            :param dict source_images: Dict {nom_source: PIL.Image ou np.ndarray}.
            :param PIL.Image mask: Masque de segmentation.
            :returns: (source_tensors, mask_tensor)
                - source_tensors: Dict {nom_source: tensor}
                - mask_tensor: Tensor du masque
        """
        # Décider des augmentations une seule fois (cohérence entre sources)
        do_hflip = self.mode == 'train' and random.random() < self.flip_prob
        do_vflip = self.mode == 'train' and random.random() < self.flip_prob

        # Appliquer les flips au masque
        if do_hflip:
            mask = TF.hflip(mask)
        if do_vflip:
            mask = TF.vflip(mask)

        mask_tensor = torch.from_numpy(np.array(mask)).long()

        # Transformer chaque source
        source_tensors = {}
        for name, img in source_images.items():
            # Flips
            if do_hflip:
                if isinstance(img, np.ndarray):
                    img = np.fliplr(img).copy()
                else:
                    img = TF.hflip(img)
            if do_vflip:
                if isinstance(img, np.ndarray):
                    img = np.flipud(img).copy()
                else:
                    img = TF.vflip(img)

            # Conversion en tenseur
            if isinstance(img, np.ndarray):
                tensor = torch.from_numpy(img.copy()).unsqueeze(0).to(torch.float)
            else:
                tensor = TF.to_tensor(img).to(torch.float)

            # Bruit gaussien (seulement en train)
            if self.mode == 'train':
                noise = torch.randn_like(tensor) * self.noise_std
                tensor = tensor + noise

            # Normalisation
            stats = self.norm_stats[name]
            tensor = TF.normalize(tensor, stats['mean'], stats['std'])

            source_tensors[name] = tensor

        return source_tensors, mask_tensor


def build_augmented_transforms(sources):
    """
    Construit les transforms d'augmentation pour train et val.
        :param dict sources: Dict des sources.
        :returns: Dict avec 'train_augmented_transform' et 'val_augmented_transform'.
    """
    return {
        'train_augmented_transform': AugmentedTransform(
            sources=sources,
            mode='train',
            noise_std=0.01,
            flip_prob=0.5
        ),
        'val_augmented_transform': AugmentedTransform(
            sources=sources,
            mode='val',
            noise_std=0.0,
            flip_prob=0.0
        ),
    }
