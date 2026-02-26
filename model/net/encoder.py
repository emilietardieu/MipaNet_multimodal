"""
Module d'encodeur multi-branches pour le modèle MIPANet.
Supporte N branches avec fusion pairwise configurable.
Chaque branche est un ResNet-18 dont le nombre de canaux d'entrée
dépend des sources qui y sont concaténées.
"""
import torch
import torch.nn as nn

from .fuse import PairwiseFusion, FUSE_MODULE_DICT
from .util import get_resnet18


class MultibranchEncoder(nn.Module):
    """
    Encodeur multi-branches basé sur ResNet-18.
        :param list branches: Liste de listes de noms de sources, ex: [['IRC', 'MNH'], ['historique']]
        :param dict sources: Dict des sources avec leurs métadonnées (channels, etc.)
        :param list pass_rff: Liste de booléens, un par branche. Si True, la branche reçoit les features raffinées.
        :param str first_fusions: Type de fusion pour les niveaux 1-3 ('PAM', 'MIM', 'MIPA').
        :param str last_fusion: Type de fusion pour le niveau 4.
        :param bool use_tgcc: Si True, utilise les chemins TGCC pour les poids pré-entraînés.
    """
    def __init__(self, branches, sources, pass_rff=None, first_fusions="PAM", last_fusion="MIPA", use_tgcc=False):
        super().__init__()

        allowed_fusions = ["PAM", "MIM", "MIPA"]
        if first_fusions not in allowed_fusions or last_fusion not in allowed_fusions:
            raise ValueError(f"Les fusions autorisées sont {allowed_fusions}. Vous avez fourni {first_fusions} et {last_fusion}.")

        self.branches_config = branches
        self.sources         = sources
        self.n_branches      = len(branches)
        self.pass_rff        = pass_rff if pass_rff is not None else [True] * self.n_branches
        self.first_fusions   = first_fusions
        self.last_fusion     = last_fusion

        # Calculer le nombre de canaux d'entrée pour chaque branche
        self.branch_channels = []
        for branch in branches:
            channels = sum(sources[src]['channels'] for src in branch)
            self.branch_channels.append(channels)

        # Créer un ResNet-18 par branche
        self.branch_bases = nn.ModuleList() # Liste des ResNet-18 de base, un par branche
        for i, channels in enumerate(self.branch_channels):
            # Poids pré-entraînés seulement si 3 canaux
            if channels == 3:
                base = get_resnet18(input_dim=3, use_tgcc=use_tgcc)
            else:
                base = get_resnet18(input_dim=channels, use_tgcc=use_tgcc)
            self.branch_bases.append(base)

        # Extraire les couches de chaque branche
        self.branch_layer0 = nn.ModuleList() # Liste des Sequential (conv1 + bn1 + relu) pour chaque branche
        self.branch_inpool = nn.ModuleList() # Liste des maxpool pour chaque branche
        self.branch_layers = nn.ModuleDict() # Dictionnaire des couches layer1, layer2, layer3, layer4 pour chaque branche, ex: 'b0_layer1', 'b1_layer1', etc.

        for i, base in enumerate(self.branch_bases):
            self.branch_layer0.append(nn.Sequential(base.conv1, base.bn1, base.relu))
            self.branch_inpool.append(base.maxpool)
            for j in range(1, 5):
                self.branch_layers[f'b{i}_layer{j}'] = getattr(base, f'layer{j}')

        # Dimensions des features pour chaque niveau (ResNet-18)
        self.fuse_feats = [64, 64, 128, 256, 512]

        # Créer les modules de fusion pairwise pour chaque niveau
        if self.n_branches >= 2:
            for level in range(len(self.fuse_feats)):
                fusion_type = self.last_fusion if level == 4 else self.first_fusions
                self.add_module(
                    f'fuse{level}',
                    PairwiseFusion(self.n_branches, self.fuse_feats[level], fusion_type)
                )

    def forward(self, branch_inputs):
        """
        Passage avant de l'encodeur multi-branches.
            :param list branch_inputs: Liste de tenseurs, un par branche [B, C_i, H, W].
            :returns: Dict contenant les features de chaque branche et les features fusionnées.
            :rtype: dict

        # Note sur les features retournées :
        - 'b{i}_{level}': features de la branche i au niveau layer{level}
        - 'x{level}': features fusionnées au niveau {level} (si n_branches >= 2), sinon les features de la branche unique   
        """
        feats = {}

        # Niveau 0 => [B, 64, h/2, w/2]
        branch_feats_0 = []
        for i in range(self.n_branches):
            f0 = self.branch_layer0[i](branch_inputs[i])
            branch_feats_0.append(f0)

        # Niveau 1 => [B, 64, h/4, w/4]
        branch_feats = []
        for i in range(self.n_branches):
            f = self.branch_inpool[i](branch_feats_0[i])
            f = self.branch_layers[f'b{i}_layer1'](f)
            branch_feats.append(f)
            feats[f'b{i}_1'] = f

        if self.n_branches >= 2:
            x1, refined = self.fuse1(branch_feats)
            feats['x1'] = x1
        else:
            feats['x1'] = branch_feats[0]
            refined = None

        # Niveaux 2, 3, 4
        for level in range(2, 5):
            new_branch_feats = []
            for i in range(self.n_branches):
                # Choisir l'entrée : features raffinées ou brutes
                if refined is not None and self.pass_rff[i]:
                    inp = refined[i]
                else:
                    inp = branch_feats[i]
                f = self.branch_layers[f'b{i}_layer{level}'](inp)
                new_branch_feats.append(f)
                feats[f'b{i}_{level}'] = f

            branch_feats = new_branch_feats

            if self.n_branches >= 2:
                fuse_module = getattr(self, f'fuse{level}')
                fused, refined = fuse_module(branch_feats)
                feats[f'x{level}'] = fused
            else:
                feats[f'x{level}'] = branch_feats[0]
                refined = None

        return feats


class Encoder(nn.Module):
    """
    Classe wrapper de l'encodeur.
        :param list branches: Config des branches.
        :param dict sources: Config des sources.
        :param encoder_kwargs: Arguments pour MultibranchEncoder.
    """
    def __init__(self, branches, sources, **encoder_kwargs):
        super().__init__()
        self.encoder = MultibranchEncoder(branches=branches, sources=sources, **encoder_kwargs)

    def forward(self, branch_inputs):
        return self.encoder(branch_inputs)
