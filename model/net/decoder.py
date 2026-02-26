"""
Module de décodeur pour le modèle MIPANet multi-branches.

IRB_block (inverted residual block):
C'est une implémentation classique de bloc type MobileNetV2
    Conv1x1 pour étendre l'espace des canaux
    Depthwise Conv3x3 pour capturer des motifs spatiaux
    Conv1x1 pour réduire les canaux
"""
import torch
import torch.nn as nn

from .util import IRB_Block, LearnedUpUnit

class Decoder(nn.Module):
    """
    Décodeur multi-branches.
    Reçoit les features fusionnées (x) ET les features de chaque branche (b0, b1, ...).
    Les skip connections concatènent toutes les features puis réduisent via conv 1x1.
        :param int n_classes: Nombre de classes de sortie.
        :param list fuse_feats: Liste des dimensions des caractéristiques [64, 64, 128, 256, 512].
        :param int n_branches: Nombre de branches de l'encodeur.
    """
    def __init__(self, n_classes, fuse_feats, n_branches=2):
        super().__init__()

        self.n_branches = n_branches

        # Inverser les dimensions des caractéristiques pour le décodage
        decoder_feats = fuse_feats[-2:0:-1]  # [256, 128, 64]

        # Niveaux correspondants dans l'encodeur (pour les skip connections)
        # decoder level 0 fuse f4→f3, level 1 fuse →f2, level 2 fuse →f1
        self.skip_levels = [3, 2, 1]  # niveaux de skip connection

        # Blocs de fusion (skip connections)
        # Chaque skip concat: decoder_feats + fused_feats + n_branches * branch_feats
        # Toutes les features ont la même dimension à un niveau donné
        skip_fuse_feats = fuse_feats[-2:0:-1]  # [256, 128, 64] — dimensions des skip connections
        for i in range(len(decoder_feats)):
            # Entrée = features du décodeur + features fusionnées + features de chaque branche
            concat_channels = decoder_feats[i] + skip_fuse_feats[i] * (1 + n_branches)
            self.add_module(f'skip_reduce{i}',
                nn.Sequential(
                    nn.Conv2d(concat_channels, decoder_feats[i], kernel_size=1, bias=False),
                    nn.BatchNorm2d(decoder_feats[i]),
                    nn.ReLU(inplace=True)
                )
            )

        # Blocs de Upsampling
        for i in range(len(decoder_feats)):
            self.add_module('up%d' % i,
                IRB_Up_Block(decoder_feats[i])
            )

        # Couches auxiliaires
        for i in range(len(decoder_feats)):
            self.add_module('aux%d' % i,
                nn.Conv2d(decoder_feats[i], n_classes, kernel_size=1, stride=1, padding=0, bias=True),
            )

        # Couches de fusion finale
        self.out_conv = nn.Sequential(
            nn.Conv2d(min(decoder_feats), n_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.out_up = nn.Sequential(
            LearnedUpUnit(n_classes),
            LearnedUpUnit(n_classes)
        )

    def forward(self, in_feats):
        # Récupérer f4 (fused au niveau 4 = bottleneck)
        f4 = in_feats['x4']

        feats, aux0 = self.up0(f4)

        # Skip connection niveau 3
        skip_tensors = [feats, in_feats['x3']]
        for b in range(self.n_branches):
            skip_tensors.append(in_feats[f'b{b}_3'])
        feats = self.skip_reduce0(torch.cat(skip_tensors, dim=1))

        feats, aux1 = self.up1(feats)

        # Skip connection niveau 2
        skip_tensors = [feats, in_feats['x2']]
        for b in range(self.n_branches):
            skip_tensors.append(in_feats[f'b{b}_2'])
        feats = self.skip_reduce1(torch.cat(skip_tensors, dim=1))

        feats, aux2 = self.up2(feats)

        # Skip connection niveau 1
        skip_tensors = [feats, in_feats['x1']]
        for b in range(self.n_branches):
            skip_tensors.append(in_feats[f'b{b}_1'])
        feats = self.skip_reduce2(torch.cat(skip_tensors, dim=1))

        aux3 = self.out_conv(feats)

        out_feats = [self.out_up(aux3), self.aux2(aux2), self.aux1(aux1), self.aux0(aux0)]

        return out_feats


class IRB_Up_Block(nn.Module):
    def __init__(self, in_feats):
        super().__init__()
        self.conv_unit = nn.Sequential(
            IRB_Block(2*in_feats, 2*in_feats),
            IRB_Block(2*in_feats, 2*in_feats),
            IRB_Block(2*in_feats, in_feats)
        )
        self.up_unit = LearnedUpUnit(in_feats) #UpSampling + Conv

    def forward(self, x):
        """
        :param torch.Tensor x: Caractéristiques d'entrée.
        :returns: Caractéristiques upsamplées et caractéristiques intermédiaires.
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        feats = self.conv_unit(x)
        return (self.up_unit(feats), feats)
