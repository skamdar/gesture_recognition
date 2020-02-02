# convert annotation data in csv file to dict and dump the dict in json format.
# dict looks as follow:

# {'labels':     ['ApplyEyeMakeup', 'ApplyLipstick', ...],
#
# 'database':   {'v_ApplyEyeMakeup_g08_c01': {'subset': 'training', 'annotations': {'lable': 'ApplyEyeMakeup'}},
#                'v_ApplyLipstick_g17_c05': {'subset': 'training', 'annotations': {'label': 'ApplyLipstick'}},
#                 ...
#               }
# }

from __future__ import print_function, division
import os
import sys
import json
import pandas as pd


def convert_csv_to_dict(csv_path, subset):
    data = pd.read_csv(csv_path, delimiter=' ', header=None)
    keys = []
    key_labels = []
    for i in range(data.shape[0]):
        row = data.ix[i, :]
        slash_rows = data.ix[i, 0].split('/')
        class_name = slash_rows[0]
        basename = slash_rows[1].split('.')[0]

        keys.append(basename)
        key_labels.append(class_name)

    database = {}
    for i in range(len(keys)):
        key = keys[i]
        database[key] = {}
        database[key]['subset'] = subset
        label = key_labels[i]
        database[key]['annotations'] = {'label': label}

    # database dict looks like:
    # {'v_ApplyEyeMakeup_g08_c01': {'subset': 'training', 'annotations': {'lable': 'ApplyEyeMakeup'}},
    #  'v_ApplyLipstick_g17_c05': {'subset': 'training', 'annotations': {'label': 'ApplyLipstick'}}...}

    return database


def load_labels(label_csv_path):
    data = pd.read_csv(label_csv_path, delimiter=' ', header=None)
    labels = []
    for i in range(data.shape[0]):
        labels.append(data.ix[i, 1])
    return labels


def convert_grit_csv_to_activitynet_json(label_csv_path, train_csv_path,
                                           val_csv_path, test_csv_path, dst_json_path):
    labels = load_labels(label_csv_path)
    train_database = convert_csv_to_dict(train_csv_path, 'training')
    val_database = convert_csv_to_dict(val_csv_path, 'validation')
    test_database = convert_csv_to_dict(test_csv_path, 'testing')

    dst_data = {}
    dst_data['labels'] = labels
    dst_data['database'] = {}
    dst_data['database'].update(train_database)
    dst_data['database'].update(val_database)
    dst_data['database'].update(test_database)

    # dst_data looks like:
    # {'labels': ['ApplyEyeMakeup', 'ApplyLipstick', ...],
    # 'database': {'v_ApplyEyeMakeup_g08_c01': {'subset': 'training', 'annotations': {'lable': 'ApplyEyeMakeup'}},
    #              'v_ApplyLipstick_g17_c05': {'subset': 'training', 'annotations': {'label': 'ApplyLipstick'}}...}}

    with open(dst_json_path, 'w') as dst_file:
        json.dump(dst_data, dst_file)


if __name__ == '__main__':
    csv_dir_path = sys.argv[1]

    # we have 1 split containing train, test and validation data

    label_csv_path = os.path.join(csv_dir_path, 'classInd.txt')
    train_csv_path = os.path.join(csv_dir_path, 'trainlist.txt')
    val_csv_path = os.path.join(csv_dir_path, 'vallist.txt')
    test_csv_path = os.path.join(csv_dir_path, 'testlist.txt')
    dst_json_path = os.path.join(csv_dir_path, 'grit.json')

    convert_grit_csv_to_activitynet_json(label_csv_path, train_csv_path,
                                               val_csv_path, test_csv_path, dst_json_path)
