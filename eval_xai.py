# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#

from fairseq.models.roberta import RobertaModel
#from musicbert import MusicBERTModel
import numpy as np
import torch
import torch.nn.functional as F
import sys
import os
from sklearn.metrics import r2_score
from map_midi_to_label import LABEL_LIST
import argparse

max_length = 8192 if 'disable_cp' not in os.environ else 1024
batch_size = 4
n_folds = 1

label_list = LABEL_LIST[3:]
scores = dict()
# for score in ["R2"]:
#     for label_name in label_list:
#         scores[score + "_" + label_name] = 


def label_fn(label, label_dict):
    return label_dict.string(
        [label + label_dict.nspecial]
    )

def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--task', choices=["xai_M2PF", "xai_M2PFnP", "xai_M2P"], required=True)
    parser.add_argument('--head_name', type = str, required=True)
    parser.add_argument('--checkpoint_file', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='xai_data_bin_apex_reg_cls/0')


for i in range(n_folds):

    print('loading model and data')
    print('start evaluating fold {}'.format(i))

    roberta = RobertaModel.from_pretrained(
        '.',
        checkpoint_file=sys.argv[1],
        data_name_or_path=sys.argv[2],
        user_dir='musicbert'
    )
    num_classes = 25
    roberta.task.load_dataset('valid')
    dataset = roberta.task.datasets['valid']
    label_dict = roberta.task.label_dictionary
    pad_index = label_dict.pad()
    roberta.cuda()
    roberta.eval()
    print(roberta)
    cnt = 0

    y_true = []
    y_pred = []

    def padded(seq):
        pad_length = max_length - seq.shape[0]
        assert pad_length >= 0
        return np.concatenate((seq, np.full((pad_length,), pad_index, dtype=seq.dtype)))

    for i in range(0, len(dataset), batch_size):
        # target = np.vstack(tuple(padded(dataset[j]['target'].numpy()) for j in range(
        #     i, i + batch_size) if j < len(dataset)))
        # target = torch.from_numpy(target)
        # #target = F.one_hot(target.long(), num_classes=(num_classes + 4))
        # #target = target.sum(dim=1)[:, 4:]
        # source = np.vstack(tuple(padded(dataset[j]['source'].numpy()) for j in range(
        #     i, i + batch_size) if j < len(dataset)))
        # source = torch.from_numpy(source)

        target = np.vstack(dataset[j]['target'].numpy() for j in range(
            i, i + batch_size) if j < len(dataset))
        target = torch.from_numpy(target)
        target = target[:,:-1]
        #target = F.one_hot(target.long(), num_classes=(num_classes + 4))
        #target = target.sum(dim=1)[:, 4:]
        source = np.vstack(tuple(padded(dataset[j]['source'].numpy()) for j in range(
            i, i + batch_size) if j < len(dataset)))
        source = torch.from_numpy(source)
        # if M2PF
        output = torch.sigmoid(roberta.predict('xai_head', source, True))
        # if M2PFnP:
        #output = torch.sigmoid(roberta.predict('xai_M2PFnP_res', source, True))
        y_true.append(target.detach().cpu().numpy())
        y_pred.append(output.detach().cpu().numpy())
        print('evaluating: {:.2f}%'.format(
            i / len(dataset) * 100), end='\r', flush=True)

    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)

    print()
    # for i in range(num_classes):
    #     print(i, label_fn(i, label_dict))
    print(y_true.shape)
    print(y_pred.shape)
    print()

    assert len(label_list) == y_pred.shape[1]

    for score in ["R2"]:
        result = r2_score(y_true, y_pred)
        scores [score + "_total"] = result
        for i, label_name in enumerate(label_list):
            scores[score + "_" + label_name] = r2_score(y_true[:,i], y_pred[:,i])
        
        print("{}:".format(score), result)
        

print(scores)

for k in scores.keys():
    print(f"{k} , {scores[k]}")
