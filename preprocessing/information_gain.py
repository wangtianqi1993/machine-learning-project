# !/user/bin/env python
# -*- coding: utf-8
__author__ = 'wtq'

import sys
# sys.path.append('/home/wtq/develop/workspace/gitlab/android-app-security-detector')
from math import log
from detector.ad.permission.base import BasePermission
from detector.config import *
from detector.logger import DetectorLogger
from operate_data.get_permission_data import get_permission_data, base_permission

# base_permission = BasePermission()
logger = DetectorLogger(path='/home/wtq/Desktop/Android-reference/android-ad-reference/ad_premission_gain.log')


def calc_shannon_ent(dataset, sign):
    """
    使用信息增益比来替代信息增益来选择特征，避免选择取值较多的特征问题，传入sign， sign为-1时根据传入的dataset不同
    计算输出数据集D的经验商，或者特征A对于数据集D的经验条件商（输入的数据集为经过split_dataset()切分输出的数据）
    当输入为特征A在数据集中的序号时，计算得出的为训练数据集D关于特征A的商,该商在算信息增益比时用到
    :param dataset:
    :return:
    """
    num_entry = len(dataset)
    label_count = {}
    for feat_vec in dataset:
        current_label = feat_vec[sign]
        if current_label not in label_count.keys():
            label_count[current_label] = 0
        label_count[current_label] += 1
    shannon_ent = 0.0
    for key in label_count:

        prob = float(label_count[key])/num_entry
        # print prob, key, num_entry, sign
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent


def split_dataset(data_set, axis, value):
    ret_dataset = []
    for feat_vec in data_set:
        if feat_vec[axis] == value:
            reduced_feat = feat_vec[:axis]
            reduced_feat.extend(feat_vec[axis+1:])
            ret_dataset.append(reduced_feat)
    return ret_dataset


def choose_best_feature(data_set):
    num_features = len(data_set[0]) - 1
    base_entropy = calc_shannon_ent(data_set, -1)
    best_info_gain = 0.0
    best_feature = -1
    info_gain_dict = {}
    # i is the ith feature of the dataset
    for i in range(num_features):
        # get the value of the ith feature
        feat_list = [example[i] for example in data_set]
        # print feat_list
        # get single value of the ith feature then use to split dataset
        unique_vals = set(feat_list)
        # print unique_vals
        ith_shannon = calc_shannon_ent(data_set, i)
        print 'ith', ith_shannon
        new_entropy = 0.0
        for value in unique_vals:
            sub_dataset = split_dataset(data_set, i, value)
            prob = len(sub_dataset)/float(len(data_set))
            new_entropy += prob * calc_shannon_ent(sub_dataset, -1)
        # if ith_shannon != 0.0:
        #     info_gain = (base_entropy - new_entropy)/ith_shannon
        # else:
        #     info_gain = (base_entropy - new_entropy)
        info_gain = base_entropy - new_entropy
        info_gain_dict[i] = info_gain
        # print 'info_gain', info_gain

        if (info_gain > best_info_gain):
            best_info_gain = info_gain
            best_feature = i
    sort_gain_dict = sorted(info_gain_dict.items(), key = lambda info_gain_dict:info_gain_dict[1], reverse=True)
    return best_feature, info_gain_dict, sort_gain_dict


def get_permission_feature():
    train_permissions = base_permission.session.query_sort(TRAIN_PERMISSION,
                                                           'create', limit=1)
    stand_permission = base_permission.get_standard_permission_from_mongodb()

    train_permissions = train_permissions['train-permission']
    i = 0
    get_feature = []
    for permission in train_permissions[0]:
        vector = base_permission.create_permission_vector(stand_permission, permission)
        vector.extend(str(train_permissions[1][i]))
        i += 1
        get_feature.append(vector)
    # logger.info(get_feature)
    return get_feature


if __name__ == '__main__':
    data_set = [[1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'yes'],
                [0, 1, 'no'],
                [0, 1, 'no'],
                [0, 0, 'no']]
    per_feature = get_permission_data("/home/wtq/develop/workspace/gitlab/android-app-security-detector/detector/yuankong_malware/feature/benign_permission.json",
                                      "/home/wtq/develop/workspace/gitlab/android-app-security-detector/detector/yuankong_malware/feature/malware_permission.json")

    best, info_gain, sort_gain = choose_best_feature(per_feature)
    for item in sort_gain:
        print base_permission[item[0]]
    logger.info(sort_gain)
