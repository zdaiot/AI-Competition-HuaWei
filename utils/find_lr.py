import math
import tqdm
import torch
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from losses.get_loss import Loss
from models.build_model import PrepareModel
from config import get_classify_config
from datasets.create_dataset import GetDataloader
from datasets.data_augmentation import DataAugmentation


def prepare(config, train_labels_number):
    """

    Args:
        config: 配置参数
        train_labels_number: list, 某一折的[number_class0, number__class1, ...]

    Returns:
        optimizer: 优化器
        model: 模型
        criterion: 损失函数
    """
    # 加载模型
    prepare_model = PrepareModel()
    model = prepare_model.create_model(
        model_type=config.model_type,
        classes_num=config.num_classes,
        drop_rate=config.drop_rate,
        pretrained=True,
        bn_to_gn=config.bn_to_gn
    )
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
        model = model.cuda()

    # 加载优化器
    optimizer = prepare_model.create_optimizer(config.model_type, model, config)

    # 加载损失函数
    criterion = Loss(config.model_type, config.loss_name, config.num_classes, train_labels_number, config.beta_CB,
                     config.gamma)
    return optimizer, model, criterion


def find_lr(trn_loader, optimizer, net, criterion, init_value=1e-8, final_value=10., beta=0.98):
    """
    Args:
        trn_loader: 训练数据集的dataloader
        optimizer: 优化器
        net: 模型
        criterion: 损失函数
        init_value: float，学习率初始值
        final_value: float，学习率最终值
        beta: float，使用平滑平均计算损失时的系数

    Returns:
        log_lrs: list，从小到大的学习率
        losses: list，对应于log_lrs学习率的损失

    Note: 请根据param_groups的实际数量调整下面的初始化学习率以及更新学习率
    """
    num = len(trn_loader) - 1
    mult = (final_value / init_value) ** (1 / num)

    lr = init_value
    optimizer.param_groups[0]['lr'] = lr * 0.1
    optimizer.param_groups[1]['lr'] = lr

    avg_loss, best_loss, batch_num = 0., 0., 0
    losses, log_lrs = [], []

    tbar = tqdm.tqdm(train_loader)
    for _, inputs, labels in tbar:
        batch_num += 1

        # As before, get the loss for this mini-batch of inputs/outputs
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        # Compute the smoothed loss
        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta ** batch_num)

        # Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            return log_lrs, losses

        # Record the best loss
        if smoothed_loss < best_loss or batch_num == 1:
            best_loss = smoothed_loss

        # Store the values
        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))

        # Do the SGD step
        loss.backward()
        optimizer.step()

        # Update the lr for the next step
        lr *= mult

        # 注意我这里有两组学习率
        optimizer.param_groups[0]['lr'] = lr * 0.1
        optimizer.param_groups[1]['lr'] = lr

        params_groups_lr = str()
        for group_ind, param_group in enumerate(optimizer.param_groups):
            params_groups_lr = params_groups_lr + 'pg_%d' % group_ind + ': %.10f, ' % param_group['lr']

        descrip = params_groups_lr + 'loss： {}'.format(loss.item())
        tbar.set_description(desc=descrip)
    return log_lrs, losses


if __name__ == "__main__":
    config = get_classify_config()
    data_root = config.data_url
    folds_split = config.n_splits
    test_size = config.val_size
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # 挑选学习率时关闭数据增强以及多尺度训练
    config.image_size = [256, 256]
    config.multi_scale = False
    config.augmentation_flag = False
    config.cut_mix = False
    config.weight_decay = 0
    config.optimizer = 'Adam'

    if config.augmentation_flag:
        transforms = DataAugmentation(config.erase_prob, full_aug=True, gray_prob=config.gray_prob)
    else:
        transforms = None

    get_dataloader = GetDataloader(
        data_root, 
        folds_split=folds_split, 
        test_size=test_size,
        choose_dataset=config.choose_dataset,
        load_split_from_file=config.load_split_from_file
        )

    train_dataloaders, val_dataloaders, train_labels_number_folds, _ = get_dataloader.get_dataloader(
        config.batch_size,
        config.image_size,
        mean, std,
        transforms=transforms,
        multi_scale=config.multi_scale
    )

    for fold_index, [train_loader, _, train_labels_number] in enumerate(
            zip(train_dataloaders, val_dataloaders, train_labels_number_folds)):
        if fold_index in config.selected_fold:
            optimizer, model, criterion = prepare(config, train_labels_number)
            log_lrs, losses = find_lr(train_loader, optimizer, model, criterion)

            plt.figure(figsize=(20, 8))
            plt.plot(log_lrs, losses)
            my_x_ticks = np.arange(-9, 1.4, 0.2)
            plt.xticks(my_x_ticks)
            plt.savefig('readme/find_lr')
            plt.show()
