"""Helper for evaluation on the Labeled Faces in the Wild dataset 
"""

# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import datetime
import os
import pickle
import random
import math

# import mxnet as mx
import numpy as np
import sklearn
import torch
import torchvision.transforms as transforms
# from mxnet import ndarray as nd
from scipy import interpolate
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from tqdm import tqdm
from PIL import Image

from backbones import get_model
from utils.utils_config import get_config

class LFold:
    def __init__(self, n_splits=2, shuffle=False):
        self.n_splits = n_splits
        if self.n_splits > 1:
            self.k_fold = KFold(n_splits=n_splits, shuffle=shuffle)

    def split(self, indices):
        if self.n_splits > 1:
            return self.k_fold.split(indices)
        else:
            return [(indices, indices)]


def distance_(embeddings0, embeddings1):
    # Distance based on cosine similarity
    dot = np.sum(np.multiply(embeddings0, embeddings1), axis=1)
    norm = np.linalg.norm(embeddings0, axis=1) * np.linalg.norm(embeddings1, axis=1)
    # shaving
    similarity = np.clip(dot / norm, -1., 1.)
    dist = np.arccos(similarity) / math.pi
    return dist

def calculate_roc(thresholds,
                  embeddings1,
                  embeddings2,
                  actual_issame,
                  nrof_folds=10,
                  pca=0,
                  subtract_mean=False):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)

    if pca == 0 and not subtract_mean:
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if subtract_mean:
            mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
            dist = distance_(embeddings1 - mean, embeddings2 - mean)

        if pca > 0:
            print('doing pca on', fold_idx)
            embed1_train = embeddings1[train_set]
            embed2_train = embeddings2[train_set]
            _embed_train = np.concatenate((embed1_train, embed2_train), axis=0)
            pca_model = PCA(n_components=pca)
            pca_model.fit(_embed_train)
            embed1 = pca_model.transform(embeddings1)
            embed2 = pca_model.transform(embeddings2)
            embed1 = sklearn.preprocessing.normalize(embed1)
            embed2 = sklearn.preprocessing.normalize(embed2)
            diff = np.subtract(embed1, embed2)
            dist = np.sum(np.square(diff), 1)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(
                                                                                threshold, dist[test_set],
                                                                                actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(
            thresholds[best_threshold_index], dist[test_set],
            actual_issame[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(
        np.logical_and(np.logical_not(predict_issame),
                       np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_val(thresholds,
                  embeddings1,
                  embeddings2,
                  actual_issame,
                  far_target,
                  nrof_folds=10):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(
                threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0
            
        val[fold_idx], far[fold_idx] = calculate_val_far(
            threshold, dist[test_set], actual_issame[test_set])

    val = list(filter(lambda x: x != None, val))
    far = list(filter(lambda x: x != None, far))

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(
        np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    # print(true_accept, false_accept)
    # print(n_same, n_diff)
    val = float(true_accept) / float(n_same) if n_same != 0 else None
    far = float(false_accept) / float(n_diff) if n_diff !=0 else None
    return val, far


def evaluate(embeddings, actual_issame, nrof_folds=10, pca=0):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = calculate_roc(thresholds,
                                       embeddings1,
                                       embeddings2,
                                       np.asarray(actual_issame),
                                       nrof_folds=nrof_folds,
                                       pca=pca)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = calculate_val(thresholds,
                                      embeddings1,
                                      embeddings2,
                                      np.asarray(actual_issame),
                                      1e-3,
                                      nrof_folds=nrof_folds)
    return tpr, fpr, accuracy, val, val_std, far

@torch.no_grad()
def test(pair_list, label_list, net, nfolds=10, data_dir=None):
    embeddings= []
    labels = []
    assert len(label_list) == len(pair_list)

    trans_list = []
    trans_list += [transforms.ToTensor()]
    trans_list += [transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
    t = transforms.Compose(trans_list)

    if len(label_list) == 0:
        return 0, 0, 0
    net.eval()
    for idx, pair in enumerate(tqdm(pair_list)):
        if data_dir is None:
            if 'png' in pair:
                path_1, path_2 = pair.split('.png /home')
                path_1 = path_1 + '.png'
            elif 'jpg' in pair:
                path_1, path_2 = pair.split('.jpg /home')
                path_1 = path_1 + '.jpg'
            elif 'JPG' in pair:
                path_1, path_2 = pair.split('.JPG /home')
                path_1 = path_1 + '.JPG'
            path_2 = '/home' + path_2
            path_2 = path_2[:-2]

        img_1 = t(Image.open(path_1)).unsqueeze(dim=0).to(net.device)
        img_2 = t(Image.open(path_2)).unsqueeze(dim=0).to(net.device)
        imgs = torch.cat((img_1, img_2), dim=0)

        embeddings.append(net(imgs).detach().cpu().numpy())
        label = int(label_list[idx])
        labels.append(label)

    embeddings = np.concatenate(embeddings, axis=0)

    _xnorm = 0 # TEMP

    embeddings = sklearn.preprocessing.normalize(embeddings)
    _, _, accuracy, val, val_std, far = evaluate(embeddings, labels, nrof_folds=nfolds)
    acc2, std2 = np.mean(accuracy), np.std(accuracy)
    return acc2, std2, acc2, std2, _xnorm, embeddings


def control_text_list(txt_root, txt_dir, data_dir=None, kist=False):
    if kist:
        img_list, pair_list = txt_root, txt_dir
        with open(img_list, 'r') as f:
            img_list = f.readlines()
        with open(pair_list, 'r') as f:
            pair_list = f.readlines()
        pairs = img_list
        labels = pair_list
    else:
        text_path = os.path.join(txt_root, txt_dir)
        lines = sorted(fixed_img_list(text_path))
        pairs = [' '.join(line.split(' ')[1:]) for line in lines]
        labels = [int(line.split(' ')[0]) for line in lines]
    return pairs, labels


def fixed_img_list(lfw_pair_text):
    f = open(lfw_pair_text, 'r')
    lines = []

    while True:
        line = f.readline()
        if not line:
            break
        lines.append(line)
    f.close()

    # random.shuffle(lines)
    return lines


@ torch.no_grad()
def verification_mag_kist(net, label_list, pair_list, transform=None, data_dir=None):
    import torchvision.transforms as T
    assert 2 * len(label_list) == len(pair_list)

    trans_list = []
    trans_list += [T.ToTensor()]
    trans_list += [T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
    t = T.Compose(trans_list)

    embeddings0, embeddings1, targets = [], [], []
    if len(label_list) == 0:
        return 0, 0

    net.eval()
    for idx in tqdm(range(len(label_list))):
        idx_1, idx_2, label = int(label_list[idx].split(' ')[0]), int(label_list[idx].split(' ')[1]), int(
            label_list[idx].split(' ')[-1].split('\n')[0])
        path_1, path_2 = pair_list[idx_1], pair_list[idx_2]
        path_1 = '/home/nas4_user/jungsoolee/FaceRecog_TestSet/img/' + '/'.join(path_1.split('/')[-2:]).split('\n')[0]
        path_2 = '/home/nas4_user/jungsoolee/FaceRecog_TestSet/img/' + '/'.join(path_2.split('/')[-2:]).split('\n')[0]
        img_1 = t(Image.open(path_1)).unsqueeze(dim=0).cuda()
        img_2 = t(Image.open(path_2)).unsqueeze(dim=0).cuda()
        imgs = torch.cat((img_1, img_2), dim=0)

        features = net(imgs)
        embeddings0.append(features[0])
        embeddings1.append(features[1])
        targets.append(label)

    embeddings0 = torch.stack(embeddings0).detach().cpu().numpy()
    embeddings1 = torch.stack(embeddings1).detach().cpu().numpy()
    targets = np.array(targets)

    thresholds = np.arange(0, 4, 0.01)
    tpr, fpr, accuracy = calculate_roc(thresholds, embeddings0, embeddings1, targets, nrof_folds=10, subtract_mean=True)
    print('EVAL with MAG - Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
    return np.mean(accuracy), np.std(accuracy)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='do verification')
    # general
    parser.add_argument("config", type=str, help="py config file")
    parser.add_argument('--model', default='./work_dirs/wf42m_pfc_r50/model.pt', help='path to load model.')
    args = parser.parse_args()

    print('loading config...')
    cfg = get_config(args.config)
    print('loading model...')
    backbone = get_model(cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).cuda()

    # load model
    backbone.load_state_dict(torch.load(args.model, map_location='cpu'))
    backbone.eval()

    print('loading test set list...')
    pairs, labels = control_text_list(txt_root='/home/nas4_user/jungsoolee/FaceRecog_TestSet/img.list',
                                    txt_dir='/home/nas4_user/jungsoolee/FaceRecog_TestSet/pair.list',
                                    kist=True)

    print('kist evaluation starts...')
    acc, std = verification_mag_kist(backbone, labels, pairs)
    # print(f'Accuracy: {acc}+-{std},')