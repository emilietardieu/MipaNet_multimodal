"""
Module de fusion pour MIPANet
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# PAM
class ChannelAttention(nn.Module):
    """
    Module d'attention par canal.
        :attributs int in_channels : Nombre de canaux des caractéristiques d'entrée.
    """
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.in_channels = in_channels

        self.to_avg_pool = nn.AdaptiveAvgPool2d(2)
        self.to_max_pool = nn.MaxPool2d(kernel_size=2)
        self.to_conv = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1)
        self.to_sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Passage avant du module ChannelAttention.
            :param torch.Tensor x : Caractéristiques d'entrée.
            :returns: Caractéristiques re-pondérées.
            :rtype: torch.Tensor
        """
        avg_pooled_x = self.to_avg_pool(x)
        max_pooled_x = self.to_max_pool(avg_pooled_x)
        weighted_x = self.to_conv(max_pooled_x)
        attention_x = self.to_sigmoid(weighted_x)
        out_x = x * attention_x + x

        return out_x

#PAM
class SpatialAttention(nn.Module):
    """
    Module d'attention spatiale.
        :attributs int in_channels : Nombre de canaux des caractéristiques d'entrée.
    """
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Passage avant du module SpatialAttention.
            :param torch.Tensor x : Caractéristiques d'entrée.
            :returns: Caractéristiques re-pondérées.
            :rtype: torch.Tensor
        """
        # Spatial attention
        spatial_attention = self.sigmoid(self.conv1(x))
        out_x = x * self.relu(spatial_attention)

        return out_x

#PAM
class to_Attention(nn.Module):
    """
    Module combinant l'attention par canal et l'attention spatiale.
    On peut tester quelle combinaison est intérressante. Initialement le code est spacial(x) + channel(y),
    Le module PAM comme présenté dans le papier est channel(x) + channel(y).
        :attributs int in_channels : Nombre de canaux des caractéristiques d'entrée.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.Self1 = ChannelAttention(in_channels)
        self.Self2 = SpatialAttention(in_channels)

    def forward(self, x, y):
        """
        Passage avant du module to_Attention.
            :param torch.Tensor x : Caractéristiques d'entrée branche A.
            :param torch.Tensor y : Caractéristiques d'entrée branche B.
            :returns: Caractéristiques fusionnées.
            :rtype: torch.Tensor
        """
        out_1 = self.Self1(x)
        out_2 = self.Self1(y)
        result = out_1 + out_2
        return  result, out_1, out_2


#MIM
class CrossAttention(nn.Module):
    """
    Module d'attention croisée
    Ce module applique une attention croisée multi-tête entre une entrée de requête (x_q)
    et une entrée clé/valeur (x_kv), avec un résiduel sur x_kv.
        :attribut int in_channels : Nombre de canaux des caractéristiques d'entrée.
        :attribut int heads : Nombre de têtes d'attention.
        :attribut int dim_head : Dimension de chaque tête d'attention.
        :attribut float dropout : Taux de dropout sur la sortie.
    """
    def __init__(self, in_channels, heads=8, dim_head=64, dropout=0.1):
        super().__init__()

        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head  # dimension totale après projection multi-tête
        self.scale = dim_head ** -0.5  # facteur de normalisation pour l'attention

        # Projections linéaires pour Q, K et V (requête, clé, valeur)
        self.to_q = nn.Linear(in_channels, self.inner_dim, bias=False)
        self.to_k = nn.Linear(in_channels, self.inner_dim, bias=False)
        self.to_v = nn.Linear(in_channels, self.inner_dim, bias=False)

        # Projection finale (optionnelle) pour ramener à la dimension d'origine
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, in_channels),
            nn.Dropout(dropout)
        ) if self.inner_dim != in_channels else nn.Identity()

    def forward(self, x_q, x_kv):
        """
        Passage avant du module CrossAttention
            :param torch.Tensor x_q : Entrée requête (query), de forme [B, C, H, W].
            :param torch.Tensor x_kv : Entrée clé/valeur (key/value),
            :returns: Résultat de l'attention croisée, de forme [B, C, H, W].
            :rtype: torch.Tensor
        """
        B, C, H, W = x_q.shape  # B: batch, C: canaux, HxW: spatial

        # Sauvegarder x_kv original pour la connexion résiduelle
        x_kv_original = x_kv.clone()

        # Mise à plat spatiale : (B, C, H, W) → (B, N, C), avec N = H * W
        x_q = x_q.view(B, C, -1).transpose(1, 2)   # [B, N, C]
        x_kv = x_kv.view(B, C, -1).transpose(1, 2) # [B, N, C]
        N = x_q.size(1)  # Nombre total de positions spatiales

        # Projections Q, K, V
        q = self.to_q(x_q)  # [B, N, heads * dim_head]
        k = self.to_k(x_kv)
        v = self.to_v(x_kv)

        # Découpe en têtes : [B, N, H*D] → [B, H, N, D]
        q = q.view(B, N, self.heads, self.dim_head).transpose(1, 2)  # [B, H, N, D]
        k = k.view(B, N, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(B, N, self.heads, self.dim_head).transpose(1, 2)

        # Attention scores : produit scalaire entre Q et K transposé
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, N, N]

        # Normalisation par softmax
        attn_probs = F.softmax(attn_scores, dim=-1)  # [B, H, N, N]

        # Pondération des valeurs : A x V
        out = torch.matmul(attn_probs, v)  # [B, H, N, D]

        # Fusion des têtes : [B, H, N, D] → [B, N, H*D]
        out = out.transpose(1, 2).contiguous().view(B, N, self.inner_dim)

        # Projection finale (si nécessaire) : [B, N, H*D] → [B, N, C]
        out = self.to_out(out)

        # Reshape inverse : [B, N, C] → [B, C, H, W]
        out = out.transpose(1, 2).view(B, C, H, W)

        # Ajout résiduel de x_kv original
        if out.size() != x_kv_original.size():
            x_kv_original = F.interpolate(x_kv_original, size=out.shape[2:], mode='bilinear', align_corners=False)
        return out + x_kv_original


#MIPA
class MIPA_Module(nn.Module):
    """
    Module MIPA combinant attention croisée et pooling pyramidal.
    Ce module applique une attention croisée bidirectionnelle entre deux entrées
    et peut optionnellement utiliser un pooling pyramidal pour la réduction de dimensionnalité.
        :attribut int in_feats : Nombre de canaux des caractéristiques d'entrée.
        :attribut tuple pp_size : Tailles pour le pooling pyramidal.
        :attribut int descriptor : Dimension du descripteur pour la réduction (-1 pour désactiver).
        :attribut int mid_feats : Dimension des caractéristiques intermédiaires.
        :attribut str sp_feats : Type de fusion spatiale ('x', 'y', ou 'u' pour somme).
    """
    def __init__(self, in_channels, pp_size=(1, 2, 4, 8), descriptor=8, mid_feats=16, sp_feats='u'):
        super().__init__()
        self.in_feats = in_channels

        # Utilisation de l'attention croisée définie dans ce fichier
        self.Corss = CrossAttention(in_channels)  # Note: typo "Corss" conservé pour compatibilité

        self.sp_feats = sp_feats
        self.pp_size = pp_size
        self.feats_size = sum([(s ** 2) for s in self.pp_size])
        self.descriptor = descriptor

        # Sans réduction de dimension
        if (descriptor == -1) or (self.feats_size < descriptor):
            self.des = nn.Identity()
            self.fc = nn.Sequential(
                nn.Linear(in_channels * self.feats_size, mid_feats, bias=False),
                nn.BatchNorm1d(mid_feats),
                nn.ReLU(inplace=True)
            )
        # Avec réduction de dimension
        else:
            self.des = nn.Conv2d(self.feats_size, self.descriptor, kernel_size=1)
            self.fc = nn.Sequential(
                nn.Linear(in_channels * descriptor, mid_feats, bias=False),
                nn.BatchNorm1d(mid_feats),
                nn.ReLU(inplace=True)
            )

        self.fc_x = nn.Linear(mid_feats, in_channels)
        self.fc_y = nn.Linear(mid_feats, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        """
        Passage avant du module MIPA_Module.
            :param torch.Tensor x : Première entrée (ex: RGB), de forme [B, C, H, W].
            :param torch.Tensor y : Deuxième entrée (ex: Depth), de forme [B, C, H, W].
            :returns: Tuple contenant (fusion_totale, sortie_1, sortie_2).
            :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        batch_size, ch, _, _ = x.size()
        sp_dict = {'x': x, 'y': y, 'u': x + y}

        # Attention croisée bidirectionnelle
        out_1 = self.Corss(sp_dict['y'], sp_dict['x'])  # y->x (Depth vers RGB)
        out_2 = self.Corss(sp_dict['x'], sp_dict['y'])  # x->y (RGB vers Depth)

        return out_1 + out_2


class PairwiseFusion(nn.Module):
    """
    Module de fusion pairwise pour N branches.
    Crée un module de fusion pour chaque paire (i, j) avec i < j.

    Pour PAM : retourne (fused, refined_i, refined_j) par paire.
    Pour MIM/MIPA : retourne un seul tensor fusionné par paire.

    La sortie fusionnée finale est la moyenne de toutes les fusions pairwise.
    Chaque branche reçoit la moyenne de ses features raffinées.
        :param int n_branches: Nombre de branches.
        :param int in_channels: Nombre de canaux des caractéristiques.
        :param str fusion_type: Type de fusion ('PAM', 'MIM', 'MIPA').
    """
    def __init__(self, n_branches, in_channels, fusion_type):
        super().__init__()
        self.n_branches = n_branches
        self.fusion_type = fusion_type

        # Créer un module de fusion pour chaque paire
        self.pairs = []
        self.fuse_modules = nn.ModuleDict()
        for i in range(n_branches):
            for j in range(i + 1, n_branches):
                pair_key = f'{i}_{j}'
                self.pairs.append((i, j, pair_key))
                self.fuse_modules[pair_key] = FUSE_MODULE_DICT[fusion_type](in_channels=in_channels)

    def forward(self, branch_feats):
        """
        Fusion pairwise de N branches.
            :param list branch_feats: Liste de tenseurs [B, C, H, W], un par branche.
            :returns: (fused, refined_list)
                - fused: Moyenne des features fusionnées [B, C, H, W]
                - refined_list: Liste de tenseurs raffinés, un par branche (ou None si MIM/MIPA)
            :rtype: tuple
        """
        fused_sum = None
        n_pairs = len(self.pairs)

        # Pour PAM, accumuler les features raffinées par branche
        if self.fusion_type == 'PAM':
            refined_accum = [None] * self.n_branches
            refined_count = [0] * self.n_branches

            for i, j, pair_key in self.pairs:
                fuse_result, ref_i, ref_j = self.fuse_modules[pair_key](branch_feats[i], branch_feats[j])

                if fused_sum is None:
                    fused_sum = fuse_result
                else:
                    fused_sum = fused_sum + fuse_result

                # Accumuler les features raffinées
                if refined_accum[i] is None:
                    refined_accum[i] = ref_i
                else:
                    refined_accum[i] = refined_accum[i] + ref_i
                refined_count[i] += 1

                if refined_accum[j] is None:
                    refined_accum[j] = ref_j
                else:
                    refined_accum[j] = refined_accum[j] + ref_j
                refined_count[j] += 1

            # Moyenner
            fused = fused_sum / n_pairs
            refined = [refined_accum[b] / refined_count[b] for b in range(self.n_branches)]
            return fused, refined

        else:  # MIM ou MIPA
            for i, j, pair_key in self.pairs:
                fuse_result = self.fuse_modules[pair_key](branch_feats[i], branch_feats[j])

                if fused_sum is None:
                    fused_sum = fuse_result
                else:
                    fused_sum = fused_sum + fuse_result

            fused = fused_sum / n_pairs
            return fused, None


FUSE_MODULE_DICT = {
    'PAM': to_Attention,
    'MIM': CrossAttention,
    'MIPA': MIPA_Module }
