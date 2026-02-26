##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""
Metriques pour l'évaluation de la segmentation sémantique.
    - Précision pixel (Pixel Accuracy)
    - Intersection over Union (IoU)
"""
import torch
import numpy as np

def batch_pix_accuracy(output, target):
    """
    Batch Pixel Accuracy
        :param torch.Tensor output: input 4D tensor, masque prédit par le modèle
        :param torch.Tensor target: label 3D tensor, masque de vérité terrain
        :returns: pixel_correct, pixel_labeled
        :rtype: int, int
    """
    _, predict = torch.max(output, 1)

    predict = predict.cpu().numpy().astype('int64') + 1
    target  = target.cpu().numpy().astype('int64') + 1

    pixel_labeled = np.sum(target > 0)
    pixel_correct = np.sum((predict == target)*(target > 0))
    assert pixel_correct <= pixel_labeled, \
        "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, nclass):
    """
    Batch Intersection of Union
        :param torch.Tensor output: 4D tensor of model outputs
        :param torch.Tensor target: 3D tensor of ground truth labels
        :param int nclass: number of categories (int)
        :returns: area_inter and area_union
        :rtype: np.ndarray, np.ndarray
    """
    _, predict = torch.max(output, 1)
    mini    = 1
    maxi    = nclass
    nbins   = nclass
    predict = predict.cpu().numpy().astype('int64') + 1
    target  = target.cpu().numpy().astype('int64') + 1

    predict = predict * (target > 0).astype(predict.dtype)
    intersection = predict * (predict == target)
    # areas of intersection and union
    area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
    area_pred, _  = np.histogram(predict, bins=nbins, range=(mini, maxi))
    area_lab, _   = np.histogram(target, bins=nbins, range=(mini, maxi))
    area_union    = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all(), \
        "Intersection area should be smaller than Union area"
    return area_inter, area_union