# !/user/bin/env python
# -*- coding: utf-8
__author__ = 'wtq'

import sys
# sys.path.append('/home/wtq/develop/workspace/gitlab/android-app-security-detector')
from math import log
from detector.ad.permission.base import BasePermission
from detector.config import *
from detector.logger import DetectorLogger

base_permission = BasePermission()
logger = DetectorLogger(path='/home/wtq/Desktop/Android-reference/android-ad-reference/ad_premission_gain.log')


def calc_shannon_ent(dataset):
    num_entry = len(dataset)
    label_count = {}
    for feat_vec in dataset:
        current_label = feat_vec[-1]
        if current_label not in label_count.keys():
            label_count[current_label] = 0
            label_count[current_label] += 1
    shannon_ent = 0.0
    for key in label_count:
        prob = float(label_count[key])/num_entry
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
    base_entropy = calc_shannon_ent(data_set)
    best_info_gain = 0.0
    best_feature = -1
    info_gain_dict = {}
    for i in range(num_features):
        # get the value of the ith feature
        feat_list = [example[i] for example in data_set]
        # print feat_list
        # get single value of the ith feature then use to split dataset
        unique_vals = set(feat_list)
        # print unique_vals
        new_entropy = 0.0
        for value in unique_vals:
            sub_dataset = split_dataset(data_set, i, value)
            prob = len(sub_dataset)/float(len(data_set))
            new_entropy += prob * calc_shannon_ent(sub_dataset)
        info_gain = base_entropy - new_entropy
        info_gain_dict[i] = info_gain
        # print 'info_gain', info_gain

        if (info_gain > best_info_gain):
            best_info_gain = info_gain
            best_feature = i
    sort_gain_dict = sorted(info_gain_dict.items(), key = lambda info_gain_dict:info_gain_dict[1])
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
    # print calc_shannon_ent(data_set)
    # print split_dataset(data_set, 0, 1)
    ad_permission = get_permission_feature()
    best_fea, info_gain, sort_gain = choose_best_feature(ad_permission)
    logger.info(sort_gain)
