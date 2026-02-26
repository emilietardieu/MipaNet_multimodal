import torch
import torch.nn as nn
import torch.nn.functional as F

def dice_loss(pred, target, smooth=1e-7, ignore_index=-1):
    """
    Calcul de la Dice Loss pour la segmentation sémantique.  
        param torch.Tensor pred: Prédictions du modèle (logits) de forme [B, C, H, W]
        param torch.Tensor target: Labels de vérité terrain de forme [B, H, W]
        param float smooth: Terme de lissage pour éviter la division par zéro
        param int ignore_index: Index à ignorer dans le calcul
        return:dice_loss_value: Valeur de la Dice Loss
        rtype: torch.Tensor
    """
    # Conversion des logits en probabilités
    pred_probs = F.softmax(pred, dim=1)
    
    # Création du masque pour ignorer certains pixels
    if ignore_index >= 0:
        mask   = (target != ignore_index)
        target = target * mask
    else:
        mask = torch.ones_like(target, dtype=torch.bool)
    
    # Conversion du target en one-hot encoding
    num_classes    = pred.shape[1]
    target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    
    # Application du masque
    if ignore_index >= 0:
        mask           = mask.unsqueeze(1).expand_as(target_one_hot)
        pred_probs     = pred_probs * mask
        target_one_hot = target_one_hot * mask
    
    # Calcul du coefficient de Dice pour chaque classe
    intersection = torch.sum(pred_probs * target_one_hot, dim=(2, 3))
    pred_sum     = torch.sum(pred_probs, dim=(2, 3))
    target_sum   = torch.sum(target_one_hot, dim=(2, 3))
    
    dice_coeff = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
    
    # Moyenne sur les classes et le batch
    dice_loss_value = 1.0 - dice_coeff.mean()
    
    return dice_loss_value

def focal_tversky_loss(pred, target, alpha=0.3, beta=0.7, gamma=1.33, smooth=1e-7, ignore_index=-1):
    """
    Calcul de la Focal Tversky Loss pour la segmentation sémantique.
        param torch.Tensor pred: Prédictions du modèle (logits) de forme [B, C, H, W]
        param torch.Tensor target: Labels de vérité terrain de forme [B, H, W]
        param float alpha: Poids pour les faux négatifs
        param float beta: Poids pour les faux positifs
        param float gamma: Exposant pour la focalisation
        param float smooth: Terme de lissage pour éviter la division par zéro
        param int ignore_index: Index à ignorer dans le calcul
        return:focal_tversky_loss_value: Valeur de la Focal Tversky Loss
        rtype: torch.Tensor

    si alpha = beta = gamma = 1, on retouve l'iou
    si alpha = beta = 0.5 et gamma = 1, on retouve le dice
    si gamma = 1, on retouve le tversky standard
    """
    pred_probs = F.softmax(pred, dim=1)
    
    if ignore_index >= 0:
        mask   = (target != ignore_index)
        target = target * mask
    else:
        mask = torch.ones_like(target, dtype=torch.bool)
    
    num_classes    = pred.shape[1]
    target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    
    if ignore_index >= 0:
        mask           = mask.unsqueeze(1).expand_as(target_one_hot)
        pred_probs     = pred_probs * mask
        target_one_hot = target_one_hot * mask
    
    # Calcul des composantes du Tversky index
    intersection    = torch.sum(pred_probs * target_one_hot, dim=(2, 3))
    false_negatives = torch.sum(target_one_hot * (1 - pred_probs), dim=(2, 3))
    false_positives = torch.sum((1 - target_one_hot) * pred_probs, dim=(2, 3))
    
    tversky_index = (intersection + smooth) / (intersection + alpha * false_negatives + beta * false_positives + smooth)
    
    # Moyenne sur les classes et le batch
    focal_tversky_loss_value = torch.pow((1 - tversky_index), gamma).mean()
    
    return focal_tversky_loss_value

class CombinedLoss(nn.Module):
    """
    Fonction de perte combinée utilisant Cross Entropy Loss et Dice Loss.
    Particulièrement utile pour maximiser l'IoU d'une classe spécifique.
    
    total_loss = α * CrossEntropyLoss + β * DiceLoss
    """
    def __init__(self, config):

        """
        Initialisation de la perte combinée.
            param dict config: Dictionnaire de configuration contenant les poids et paramètres pour les différentes composantes de la perte.
                - config['aux_weight']: Poids pour les pertes auxiliaires
                - config['first_loss']['ce_weight']: Poids pour la Cross Entropy Loss
                - config['second_loss']['loss_type']: Type de la seconde perte ("dice" ou "focal_tversky")
                - config['second_loss']['focal_tversky']: Dictionnaire contenant les paramètres alpha, beta, gamma pour la Focal Tversky Loss   
        """
        super(CombinedLoss, self).__init__()
        self.nclass       = 3
        self.ignore_index = -1
        
        self.aux          = True
        self.aux_weight   = config['aux_weight']

        self.ce_weight    = config['first_loss']['ce_weight']
        self.sec_weight   = 1 - self.ce_weight
    
        self.loss_type    = config['second_loss']['loss_type']
        self.ft_alpha     = config['second_loss']['focal_tversky']['ft_alpha']
        self.ft_beta      = config['second_loss']['focal_tversky']['ft_beta']
        self.ft_gamma     = config['second_loss']['focal_tversky']['ft_gamma']
        self.dice_smooth  = 1e-7

        # Conversion des poids de classes en tenseur  
        cw = config["first_loss"]["class_weight"]
        if cw is None:
            self.register_buffer("class_weight", None)
        else:
            self.register_buffer("class_weight", torch.tensor(cw, dtype=torch.float))

        self.ce_loss = nn.CrossEntropyLoss(
            weight=self.class_weight,
            ignore_index=self.ignore_index,
            reduction="mean",
)
    
    def forward(self, *inputs):
        """
        Calcul de la perte combinée.
            param tuple inputs: Tuple contenant les prédictions principales, les prédictions auxiliaires (si présentes) et les cibles.
            return total_loss: Perte totale combinée
            rtype: torch.Tensor
        """
        if self.aux:
            out_feats, target = inputs[0], inputs[-1]
            aux_feats = inputs[1:-1]
            
            # Perte principale
            ce_main = self.ce_loss(out_feats, target)
            # Perte secondaire
            if self.loss_type == "dice":
                secondary_main = dice_loss(out_feats, target, self.dice_smooth, self.ignore_index)
            elif self.loss_type == "focal_tversky":
                secondary_main = focal_tversky_loss(out_feats, target, self.ft_alpha, self.ft_beta, self.ft_gamma, self.dice_smooth, self.ignore_index)
            main_loss = self.ce_weight * ce_main + self.sec_weight * secondary_main
            
            # Pertes auxiliaires
            aux_losses = []
            for aux in aux_feats:
                _, _, h, w = aux.size()
                aux_target = F.interpolate(target.unsqueeze(1).float(), 
                                         size=(h, w)).long().squeeze(1)
                if self.ce_loss.weight is not None:
                    self.ce_loss.weight = self.ce_loss.weight.to(aux.device)
                ce_aux = self.ce_loss(aux, aux_target)
                if self.loss_type   == "dice":
                    secondary_aux = dice_loss(aux, aux_target, self.dice_smooth, self.ignore_index)
                elif self.loss_type == "focal_tversky":
                    secondary_aux = focal_tversky_loss(aux, aux_target, self.ft_alpha, self.ft_beta, self.ft_gamma, self.dice_smooth, self.ignore_index)
                aux_combined = self.ce_weight * ce_aux + self.sec_weight * secondary_aux
                aux_losses.append(aux_combined)
            
            # Combinaison des pertes
            aux_loss_avg = sum(aux_losses) / len(aux_losses)
            total_loss = main_loss + self.aux_weight * aux_loss_avg
            
            return total_loss
        
        else:
            # Pas de pertes auxiliaires
            out_feats, target = inputs[0], inputs[-1]
            ce_loss_val = self.ce_loss(out_feats, target)
            if self.loss_type   == "dice":
                secondary_loss_val = dice_loss(out_feats, target, self.dice_smooth, self.ignore_index)
            elif self.loss_type == "focal_tversky":
                secondary_loss_val = focal_tversky_loss(out_feats, target, self.ft_alpha, self.ft_beta, self.ft_gamma, self.dice_smooth, self.ignore_index)
            
            total_loss = self.ce_weight * ce_loss_val + self.sec_weight * secondary_loss_val
            return total_loss