import torch
from torch import nn

from model.net.encoder import Encoder
from model.net.decoder import Decoder

class MipaNet(nn.Module):
    """
    Main class for the MIPANet model — multi-branches configurable.
        :param int n_classes: Number of output classes.
        :param list branches: Liste de listes de sources par branche.
        :param dict sources: Dictionnaire des sources avec leurs métadonnées.
        :param dict encoder_kwargs: Arguments pour l'encodeur (pass_rff, first_fusions, last_fusion, etc.).
    """

    def __init__(self, n_classes, branches, sources, **encoder_kwargs):
        super().__init__()

        self.branches   = branches
        self.sources    = sources
        self.n_branches = len(branches)

        self.encoder = Encoder(
            branches=branches,
            sources =sources,
            **encoder_kwargs
        )

        self.decoder = Decoder(
            n_classes=n_classes,
            fuse_feats=self.encoder.encoder.fuse_feats,
            n_branches=self.n_branches,
        )

    def forward(self, sources_dict):
        """
        Passage avant du modèle.
            :param dict sources_dict: Dict {nom_source: tensor [B, C, H, W]}
            :returns: Tuple de prédictions (main + auxiliaires).
        """
        # Grouper les sources par branche et concaténer (early fusion)
        branch_inputs = []
        for branch in self.branches:
            tensors = [sources_dict[src] for src in branch]
            if len(tensors) == 1:
                branch_inputs.append(tensors[0])
            else:
                branch_inputs.append(torch.cat(tensors, dim=1))

        feats = self.encoder(branch_inputs)
        feats = self.decoder(feats)
        return tuple(feats)


def get_mipanet(dataset, branches, sources, **encoder_kwargs):
    from .datasets import datasets
    model = MipaNet(datasets[dataset.lower()].NUM_CLASS, branches, sources, **encoder_kwargs)
    return model
