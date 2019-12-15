import shutil
import os
import tqdm
from utils.data_analysis import DatasetStatistic
import json
import random
import numpy as np


#########################
# 依据类别得分动态拷贝样本
#########################
def combine_dataset(download_root, official_root, combine_root, labels_to_complement_number):
    """

    Args:
        download_root: str, 自己下载的数据集路径
        official_root: str, 官方数据集路径
        combine_root: str, 合并后的文件夹路径
        labels_to_complement_number: dict, {类别1：补充样本数, ...}

    Returns:

    """
    if os.path.exists(combine_root):
        print('Removing %s' % combine_root)
        shutil.rmtree(combine_root)
        print('Making %s' % combine_root)
        os.mkdir(combine_root)
    else:
        print('Making %s' % combine_root)
        os.mkdir(combine_root)

    download_files = os.listdir(download_root)
    tbar = tqdm.tqdm(labels_to_complement_number.items())
    for label, number in tbar:
        # 筛选出该类别的文件名称列表
        files_name = [file_name for file_name in download_files if label.split('/')[1] in file_name]
        selected_number = int(min(number, len(files_name) / 2))

        selected_sample_files = get_selected_file(download_root, files_name, selected_number)
        selected_label_files = [file_name.replace('jpg', 'txt') for file_name in selected_sample_files]
        descript = '%s: sampled: %d / %d' % (label, selected_number, len(files_name) / 2)
        tbar.set_description(desc=descript)
        for file_name in selected_sample_files + selected_label_files:
            scr_file = os.path.join(download_root, file_name)
            target_file = os.path.join(combine_root, file_name)
            shutil.copy(scr_file, target_file)

    official_files = os.listdir(official_root)
    tbar = tqdm.tqdm(official_files)
    for official_file in tbar:
        scr_file = os.path.join(official_root, official_file)
        target_file = os.path.join(combine_root, official_file)
        shutil.copy(scr_file, target_file)


def get_selected_file(folder_root, files_name, selected_number):
    """从给定的文件列表中按照分数大小选取指定数目个文件

    Args:
        folder_root: str, 文件夹目录
        files_name: list, 文件名称列表
        selected_number: int, 挑选的文件个数
    Returns:
        selected_files: list, 被选中的文件列表
    """
    sample_files_name = [file_name for file_name in files_name if file_name.endswith('.jpg')]
    samples_score = []
    for sample_file_name in sample_files_name:
        score_file_name = sample_file_name.replace('.jpg', '_score.txt')
        with open(os.path.join(folder_root, score_file_name), 'r') as f:
            score = float(f.readline()[0])
        samples_score.append(score)
    selected_index = np.argsort(np.asarray(samples_score))[:selected_number]
    selected_files = [sample_files_name[index] for index in selected_index]

    return selected_files


def calculate_complement_number(labels_scores, max_number, min_number):
    """ 按照分数计算需要补充的样本数目
    Args:
        labels_scores: dict, 各个类别的得分
        max_number: int, 最大补充样本数
        min_number: int, 最少补充样本上数

    Returns:
        labels_to_complement_number: dict, {类别1：补充样本数, ...}
    """
    max_score = sorted(labels_scores.values())[-1]
    min_score = sorted(labels_scores.values())[0]
    labels_to_complement_number = {}
    for key, value in labels_scores.items():
        # 得分越高，补充的样本数目越少
        complement_number = int((max_score - value) / (max_score - min_score) * (max_number - min_number) + min_number)
        labels_to_complement_number[key] = complement_number

    return labels_to_complement_number


if __name__ == "__main__":
    data_root = 'data/huawei_data/train_data'
    download_root = 'data/huawei_data/psudeo'
    combine_root = 'data/huawei_data/combine_dynamic'
    label_id_json = 'data/huawei_data/label_id_name.json'
    score_file = 'checkpoints/se_resnext101_32x4d/log-2019-12-06T00-15-51/classes_acc.json'
    with open(score_file, 'r') as f:
        labels_scores = json.load(f)

    dataset_statistic = DatasetStatistic(data_root, label_id_json)
    thresh = 100
    # 高于样本数目阈值的类别的补充数目
    more_than_thresh_number = 150
    # 低于样本数目阈值的类别的补充数目
    less_than_thresh_number = 200
    # 依据官方数据计算各个类别补充样本的数目
    # labels_to_complement_number = dataset_statistic.get_expand_number(thresh, more_than_thresh_number, less_than_thresh_number)

    # 依据分数计算需要补充的样本数目
    labels_to_complement_number = calculate_complement_number(labels_scores, max_number=75, min_number=0)

    print(labels_to_complement_number)
    combine_dataset(download_root, data_root, combine_root, labels_to_complement_number)
