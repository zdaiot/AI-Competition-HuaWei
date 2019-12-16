import torch
import torch.nn as nn


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: q_i = (1 - epsilon) * a_i + epsilon / N.
    """
    def __init__(self, num_classes, epsilon=0.1, alpha=None, use_gpu=True):
        """

        Args:
            num_classes: int, 类别总数
            epsilon: float, 系数
            alpha: list, float or list, 类别的权重
            use_gpu: bool, 是否使用gpu
        """
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.epsilon = epsilon

        # alpha = [1 for x in range(self.num_classes)]
        # alpha[25], alpha[26], alpha[28] = 1.5, 0.9, 0.95
        # alpha[30], alpha[32], alpha[31] = 1.3, 1.2, 1.1
        # alpha[33], alpha[37] = 2, 0.9
        # alpha[48], alpha[51] = 1.2, 0.95
        # alpha[49], alpha[50] = 2.5, 1.3

        if isinstance(alpha, (float, int)):
            self.alpha = torch.FloatTensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.FloatTensor(alpha)

        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)

        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            log_probs = log_probs * self.alpha

        '''
        scatter_第一个参数为1表示分别对每行填充；targets.unsqueeze(1)得到的维度为[num_classes, 1]；
        填充方法为：取出targets的第i行中的第一个元素（每行只有一个元素），记该值为j；则前面tensor中的(i,j)元素填充1；
        最终targets的维度为[batch_size, num_classes]，每一行代表一个样本，若该样本类别为j，则只有第j元素为1，其余元素为0
        '''
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu:
            targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        # mean(0)表示缩减第0维，也就是按列求均值，得到维度为[num_classes]，得到该batch内每一个类别的损失，再求和
        loss = (- targets * log_probs).mean(0).sum()
        return loss


if __name__ == '__main__':
    num_of_classes = 5
    logits = torch.rand(10, num_of_classes).float()
    labels = torch.randint(0, num_of_classes, size=(10,))

    my_CB_loss = CrossEntropyLabelSmooth(num_of_classes, alpha=[1 for x in range(num_of_classes)], use_gpu=False)
    cb_loss = my_CB_loss(logits, labels)
    print(cb_loss)
