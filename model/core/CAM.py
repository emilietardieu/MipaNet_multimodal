"""
Score-CAM — explicabilité pour MipaNet (modèles mono-branche).

Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks
Wang et al., CVPR 2020

Algorithme (mode non signé, signed=False) :
  1. Hook sur la couche cible pour capturer les activations [1, C, h, w].
  2. Pour chaque canal k :
       a. Upsample activation[k] à la taille de l'entrée
       b. Normalise dans [0, 1] → masque
       c. Entrée masquée = input × masque
       d. Forward masqué → score de la classe cible (prob softmax moyenne)
  3. CAM = ReLU( somme_k( score_k × masque_k ) )

Mode signé (signed=True) :
  - Un score de référence est calculé sur une entrée nulle (baseline).
  - score_k ← score_k − score_baseline  (peut être négatif)
  - Pas de ReLU → zones inhibitrices conservées (valeurs négatives)
  - CAM normalisée dans [−1, 1]

Usage :
    with ScoreCAM(model) as cam_gen:
        cam = cam_gen.generate(sources_dict, target_class=2)
        cam_signed = cam_gen.generate(sources_dict, target_class=2, signed=True)
"""

import torch
import torch.nn.functional as F
import numpy as np


# Couches disponibles dans l'encodeur mono-branche (ResNet-18)
# Clé : nom lisible → clé dans branch_layers
AVAILABLE_LAYERS = {
    'layer1': 'b0_layer1',  # 64 ch,  H/4  × W/4
    'layer2': 'b0_layer2',  # 128 ch, H/8  × W/8
    'layer3': 'b0_layer3',  # 256 ch, H/16 × W/16
    'layer4': 'b0_layer4',  # 512 ch, H/32 × W/32  (défaut)
}


class ScoreCAM:
    """
    Score-CAM pour MipaNet mono-branche.
        :param nn.Module model: MipaNet entraîné (n_branches == 1), en mode eval.
        :param nn.Module target_layer: Couche dont on capte les activations.
        Si None, utilise b0_layer4 (bottleneck de l'encodeur).
        :param int batch_size: Nombre de masques traités en parallèle lors du forward masqué.
        Réduire si OOM.
    """

    def __init__(self, model, target_layer=None, batch_size=32):
        self.model = model
        self.model.eval()
        self.batch_size = batch_size
        self._activations = None

        if target_layer is None:
            target_layer = model.encoder.encoder.branch_layers['b0_layer4']
        self.target_layer = target_layer
        self._hook = target_layer.register_forward_hook(self._hook_fn)

    # ------------------------------------------------------------------
    # Méthodes internes
    # ------------------------------------------------------------------

    def _hook_fn(self, module, input, output):
        self._activations = output.detach()

    def _normalize_maps(self, maps):
        """
        Normalise chaque carte d'activation dans [0, 1] indépendamment.
            :param torch.Tensor maps: Cartes d'activation [N, H, W].
            :return maps_norm: Tenseur normalisé [N, H, W].
            :rtype : torch.Tensor
        """
        flat = maps.view(maps.shape[0], -1)
        mn = flat.min(dim=1)[0].view(-1, 1, 1)
        mx = flat.max(dim=1)[0].view(-1, 1, 1)
        return (maps - mn) / (mx - mn + 1e-8)

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def generate(self, sources_dict, target_class, signed=False):
        """
        Génère la carte Score-CAM pour une image et une classe cible.
            :param dict sources_dict: Dictionnaire {nom_source: tensor [1, C, H, W]}, batch size = 1.
            :param int target_class: Indice de la classe cible.
            :param bool signed: Si True, retourne une CAM signée dans [−1, 1] montrant
                aussi les zones inhibitrices (score < baseline). Si False (défaut),
                applique un ReLU et retourne des valeurs dans [0, 1].
            :return cam: Carte de chaleur [H, W].
            :rtype: numpy.ndarray
        """
        first = next(iter(sources_dict.values()))
        _, _, H, W = first.shape
        device = first.device

        # --- 1. Forward de référence pour capturer les activations ---
        with torch.no_grad():
            _ = self.model(sources_dict)

        acts = self._activations          # [1, n_channels, h, w]
        n_channels = acts.shape[1]

        # --- 2. Upsample + normalisation → masques binaires doux ---
        upsampled = F.interpolate(acts, size=(H, W), mode='bilinear', align_corners=False)
        upsampled = upsampled.squeeze(0)  # [n_channels, H, W]
        masks = self._normalize_maps(upsampled)

        # --- 3a. Score baseline (entrée nulle) — uniquement si signed=True ---
        if signed:
            baseline_sources = {k: torch.zeros_like(v) for k, v in sources_dict.items()}
            with torch.no_grad():
                out_base = self.model(baseline_sources)
            baseline_score = F.softmax(out_base[0], dim=1)[:, target_class].mean().item()
        else:
            baseline_score = 0.0

        # --- 3b. Score de chaque masque ---
        scores = torch.zeros(n_channels, device=device)

        for i in range(0, n_channels, self.batch_size):
            batch_masks = masks[i : i + self.batch_size]   # [bs, H, W]
            bs = batch_masks.shape[0]

            # Appliquer chaque masque à chaque source : input × masque
            masked_sources = {}
            for src_name, src_tensor in sources_dict.items():
                # src_tensor : [1, C, H, W]  →  broadcast avec [bs, 1, H, W]
                masked_sources[src_name] = (
                    src_tensor.expand(bs, -1, -1, -1) * batch_masks.unsqueeze(1)
                )

            with torch.no_grad():
                out = self.model(masked_sources)

            # Probabilité softmax moyenne pour la classe cible, centrée sur baseline
            probs = F.softmax(out[0], dim=1)               # [bs, n_classes, H, W]
            scores[i : i + bs] = probs[:, target_class].mean(dim=(1, 2)) - baseline_score

        # --- 4. Combinaison pondérée ---
        cam = (scores.view(n_channels, 1, 1) * masks).sum(dim=0)   # [H, W]

        if signed:
            # Normalisation symétrique dans [−1, 1]
            abs_max = cam.abs().max()
            if abs_max > 1e-8:
                cam = cam / abs_max
        else:
            # ReLU + normalisation dans [0, 1]
            cam = F.relu(cam)
            cam_min, cam_max = cam.min(), cam.max()
            if cam_max > cam_min:
                cam = (cam - cam_min) / (cam_max - cam_min)

        return cam.cpu().numpy()

    def remove_hook(self):
        """Supprime le hook de forward. À appeler après utilisation."""
        self._hook.remove()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove_hook()


# ------------------------------------------------------------------
# Utilitaire : sélectionner une couche par son nom
# ------------------------------------------------------------------

def get_layer(model, layer_name='layer4'):
    """
    Retourne la couche de l'encodeur mono-branche correspondant à layer_name.
        :param MipaNet model: Le modèle MipaNet mono-branche.
        :param str layer_name: Nom de la couche cible : 'layer1', 'layer2', 'layer3' ou 'layer4'.
        :return layer: La couche sélectionnée dans l'encodeur.
        :rtype: nn.Module

    Exemple::
        layer = get_layer(model, 'layer3')
        with ScoreCAM(model, target_layer=layer) as cam_gen:
            cam = cam_gen.generate(sources_dict, target_class=2)
    """
    if layer_name not in AVAILABLE_LAYERS:
        raise ValueError(
            f"layer_name doit être l'un de {list(AVAILABLE_LAYERS.keys())}, reçu '{layer_name}'."
        )
    key = AVAILABLE_LAYERS[layer_name]
    return model.encoder.encoder.branch_layers[key]
