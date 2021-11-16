from __future__ import division
from __future__ import print_function

import numpy as np
import math
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import metrics
from matplotlib import pyplot as plt

from utils import load_data, load_data2, load_data_gcn, accuracy
from model.models import GCN


import warnings
warnings.filterwarnings("ignore")

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正确显示中文标签
plt.rcParams['font.serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 用来正确显示负号


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=40, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')


args = parser.parse_args()
# print(parser)
# print(args)
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


# adj, features, labels, idx_train, idx_val, idx_test = load_data()
adj, features, labels, idx_train, idx_val, idx_test = load_data2()
# adj, features, labels, idx_train, idx_val, idx_test = load_data_gcn()
adj2, features2, labels2, idx_train2, idx_val2, idx_test2 = load_data_gcn()
# print(idx)
print(features.shape)
# print(labels.shape)



model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

print("---------------")
# print(features.shape[1])

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    loss_history = loss_train.item()
    val_acc_history = acc_val.item()
    from sklearn import metrics
    # 计算 AUC
    # fpr, tpr, thresholds = metrics.roc_curve(output[idx_val].detach().numpy().argmax(axis=1), labels[idx_val], pos_label=1)
    # auc = metrics.auc(fpr, tpr)
    # print(f"AUC: {auc}")

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_history, val_acc_history


def test():
    model.eval()
    output = model(features2, adj2)
    loss_test = F.nll_loss(output[idx_test2], labels2[idx_test2])
    acc_test = accuracy(output[idx_test2], labels2[idx_test2], test=True)
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

    # 计算 AUC
    # fpr, tpr, thresholds = metrics.roc_curve(output[idx_val].detach().numpy().argmax(axis=1), labels[idx_val], pos_label=1)
    # auc = metrics.auc(fpr, tpr)
    # print(f"AUC: {auc}")


if __name__ == '__main__':
    # Train model
    loss_history = []
    val_acc_history = []
    ep = []
    t_total = time.time()
    for epoch in range(args.epochs):
        loss, val_acc = train(epoch)
        loss_history.append(loss)
        val_acc_history.append(val_acc)
        ep.append(epoch)
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    test()

    fig, ax1 = plt.subplots()
    # 产生一个ax1镜面坐标
    ax2 = ax1.twinx()

    ax1.set_title('训练损失和验证集准确率')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2.set_ylabel("ValAcc")
    ax1.plot(ep, loss_history, 'r')
    ax2.plot(ep, val_acc_history, 'b')
    plt.tight_layout()
    # plt.legend(['Loss', 'ValAcc'])
    # plt.show()
