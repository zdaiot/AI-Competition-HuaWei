"""Pytorch implementation of Class-Balanced-Loss
   Reference: "Class-Balanced Loss Based on Effective Number of Samples"
   Authors: Yin Cui and
               Menglin Jia and
               Tsung Yi Lin and
               Yang Song and
               Serge J. Belongie
   https://arxiv.org/abs/1901.05555, CVPR'19.
   https://github.com/vandit15/Class-balanced-loss-pytorch/blob/master/class_balanced_loss.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_loss(labels, logits, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.

    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).

    Args:
        labels: A float tensor of size [batch, num_classes].
        logits: A float tensor of size [batch, num_classes].
        alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
        gamma: A float scalar modulating loss from hard and easy examples.

    Returns:
        focal_loss: A float32 scalar representing normalized total loss.
    """
    BCLoss = F.binary_cross_entropy_with_logits(input=logits, target=labels, reduction="none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 +
                                                                           torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss


class CB_Loss(nn.Module):
    def __init__(self, samples_per_class, num_of_classes, loss_type, beta, gamma):
        super(CB_Loss, self).__init__()
        """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
    
        Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
        where Loss is one of the standard losses used for Neural Networks.
    
        Args:
            samples_per_class: A python list of size [num_of_classes].
            num_of_classes: total number of classes. int
            loss_type: string. One of "CB_Sigmoid", "CB_Focal", "CB_Softmax".
            beta: float. Hyperparameter for Class balanced loss.
            gamma: float. Hyperparameter for Focal loss.
        """
        self.samples_per_class = samples_per_class
        self.num_of_classes = num_of_classes
        self.loss_type = loss_type
        self.beta = beta
        self.gamma = gamma

        effective_num = 1.0 - np.power(self.beta, self.samples_per_class)
        self.weights = (1.0 - self.beta) / np.array(effective_num)
        # Normalization operation
        self.weights = self.weights / np.sum(self.weights) * self.num_of_classes
        self.weights = torch.tensor(self.weights).float()
        self.weights = self.weights.unsqueeze(0)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: A int tensor of size [batch].
            targets: A float tensor of size [batch, num_of_classes].

        Returns:
          cb_loss: A float tensor representing class balanced loss
        """
        labels_one_hot = F.one_hot(targets, self.num_of_classes).float()
        # after repeat, dim [batch_size, num_of_classes], 相乘后只有真实类标对应的位置有值
        weights = self.weights.repeat(labels_one_hot.shape[0], 1).to(labels_one_hot.device) * labels_one_hot
        # dim [batch_size]
        weights = weights.sum(1)
        # dim [batch_size, 1]
        weights = weights.unsqueeze(1)
        # dim [batch_size, num_of_classes], 每一行的元素均相同
        weights = weights.repeat(1, self.num_of_classes)

        if self.loss_type == "CB_Focal":
            cb_loss = focal_loss(labels_one_hot, inputs, weights, self.gamma)
        elif self.loss_type == "CB_Sigmoid":
            cb_loss = F.binary_cross_entropy_with_logits(input=inputs, target=labels_one_hot, weight=weights)
        elif self.loss_type == "CB_Softmax":
            pred = inputs.softmax(dim=1)
            cb_loss = F.binary_cross_entropy(input=pred, target=labels_one_hot, weight=weights)
        else:
            assert NotImplementedError
        return cb_loss


def CB_loss_function(labels, logits, samples_per_cls, num_of_classes, loss_type, beta, gamma):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.

    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.

    Args:
        labels: A int tensor of size [batch].
        logits: A float tensor of size [batch, num_of_classes].
        samples_per_cls: A python list of size [num_of_classes].
        num_of_classes: total number of classes. int
        loss_type: string. One of "CB_Sigmoid", "CB_Focal", "CB_Softmax".
        beta: float. Hyperparameter for Class balanced loss.
        gamma: float. Hyperparameter for Focal loss.

    Returns:
        cb_loss: A float tensor representing class balanced loss
    """
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    # Normalization operation
    weights = weights / np.sum(weights) * num_of_classes

    labels_one_hot = F.one_hot(labels, num_of_classes).float()

    weights = torch.tensor(weights).float()
    weights = weights.unsqueeze(0)
    # after repeat, dim [batch_size, num_of_classes], 相乘后只有真实类标对应的位置有值
    weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
    # dim [batch_size]
    weights = weights.sum(1)
    # dim [batch_size, 1]
    weights = weights.unsqueeze(1)
    # dim [batch_size, num_of_classes], 每一行的元素均相同
    weights = weights.repeat(1, num_of_classes)

    if loss_type == "CB_Focal":
        cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
    elif loss_type == "CB_Sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels_one_hot, weight=weights)
    elif loss_type == "CB_Softmax":
        pred = logits.softmax(dim=1)
        cb_loss = F.binary_cross_entropy(input=pred, target=labels_one_hot, weight=weights)
    else:
        assert NotImplementedError
    return cb_loss


if __name__ == '__main__':
    num_of_classes = 5
    logits = torch.rand(10, num_of_classes).float()
    labels = torch.randint(0, num_of_classes, size=(10,))
    beta = 0.9999
    gamma = 2.0
    samples_per_cls = [2, 3, 1, 2, 2]
    loss_type = "CB_Softmax"
    cb_loss = CB_loss_function(labels, logits, samples_per_cls, num_of_classes, loss_type, beta, gamma)
    print(cb_loss)

    my_CB_loss = CB_Loss(samples_per_cls, num_of_classes, loss_type, beta, gamma)
    cb_loss = my_CB_loss(logits, labels)
    print(cb_loss)
