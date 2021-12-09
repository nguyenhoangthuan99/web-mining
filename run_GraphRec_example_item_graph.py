import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import pickle
import numpy as np
import time
import random
from collections import defaultdict
from UV_Encoders import UV_Encoder
from UV_Aggregators import UV_Aggregator
#from Social_Encoders import Social_Encoder
from Social_Encoders_v2 import Social_Encoder
from Social_Aggregators import Social_Aggregator
import torch.nn.functional as F
import torch.utils.data
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import datetime
import argparse
import os
import json
"""
GraphRec: Graph Neural Networks for Social Recommendation. 
Wenqi Fan, Yao Ma, Qing Li, Yuan He, Eric Zhao, Jiliang Tang, and Dawei Yin. 
In Proceedings of the 28th International Conference on World Wide Web (WWW), 2019. Preprint[https://arxiv.org/abs/1902.07243]

If you use this code, please cite our paper:
```
@inproceedings{fan2019graph,
  title={Graph Neural Networks for Social Recommendation},
  author={Fan, Wenqi and Ma, Yao and Li, Qing and He, Yuan and Zhao, Eric and Tang, Jiliang and Yin, Dawei},
  booktitle={WWW},
  year={2019}
}
```

"""


class GraphRec(nn.Module):

    def __init__(self, enc_u, enc_v_history, r2e):
        super(GraphRec, self).__init__()
        self.enc_u = enc_u
        self.enc_v_history = enc_v_history
        self.embed_dim = enc_u.embed_dim

        self.w_ur1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_ur2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_uv1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.w_uv2 = nn.Linear(self.embed_dim, 16)
        self.w_uv3 = nn.Linear(16, 1)
        self.r2e = r2e
        self.bn1 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn2 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn3 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn4 = nn.BatchNorm1d(16, momentum=0.5)
        self.criterion = nn.MSELoss()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, nodes_u, nodes_v):
        embeds_u = self.enc_u(nodes_u)
        embeds_v = self.enc_v_history(nodes_v)

        x_u = F.relu(self.bn1(self.w_ur1(embeds_u)))
        #x_u = self.dropout(x_u) #F.dropout(x_u, training=self.training)
        x_u = self.w_ur2(x_u)
        x_v = F.relu(self.bn2(self.w_vr1(embeds_v)))
        x_v = self.dropout(x_v) #F.dropout(x_v, training=self.training)
        x_v = self.w_vr2(x_v)

        x_uv = torch.cat((x_u, x_v), 1)
        x = F.relu(self.bn3(self.w_uv1(x_uv)))
        #x = self.dropout(x) #F.dropout(x, training=self.training)
        x = F.relu(self.bn4(self.w_uv2(x)))
        x = self.dropout(x) #F.dropout(x, training=self.training)
        scores = self.w_uv3(x)
        return scores.squeeze()

    def loss(self, nodes_u, nodes_v, labels_list):
        scores = self.forward(nodes_u, nodes_v)
        return self.criterion(scores, labels_list)

import sys
def train(model, device, train_loader, optimizer, epoch, best_rmse, best_mae):
    model.train()
    running_loss = 0.0
    expected_rmse = 999
    steps = []
    for i, data in enumerate(train_loader, 0):
        batch_nodes_u, batch_nodes_v, labels_list = data
        optimizer.zero_grad()
        loss = model.loss(batch_nodes_u.to(device), batch_nodes_v.to(device), labels_list.to(device))
        loss.backward(retain_graph=True)
        optimizer.step()
        running_loss += loss.item()
        steps.append(loss.item())
        sys.stdout.write('\r>>'+'step ' +str(i%100)+', Loss '+str(np.mean(steps)) )
         
        sys.stdout.flush()
        
        if i % 100 == 0:
            steps = []
            print('[%d, %5d] loss: %.3f, The best rmse/mae: %.6f / %.6f' % (
                epoch, i, running_loss / 100, best_rmse, best_mae))
            running_loss = 0.0
            sys.stdout.write('\n')
    return 0


def test(model, device, test_loader):
    model.eval()
    tmp_pred = []
    target = []
    with torch.no_grad():
        for test_u, test_v, tmp_target in test_loader:
          #  print(test_u.size(), test_v.size(), tmp_target.size() )
            test_u, test_v, tmp_target = test_u.to(device), test_v.to(device), tmp_target.to(device)
            val_output = model.forward(test_u, test_v)
            #print(val_output.size())
            tmp_pred.append(list(val_output.data.cpu().numpy()))
            target.append(list(tmp_target.data.cpu().numpy()))
    tmp_pred = np.array(sum(tmp_pred, []))
    target = np.array(sum(target, []))
    print(tmp_pred,target)
    expected_rmse = sqrt(mean_squared_error(tmp_pred, target))
    mae = mean_absolute_error(tmp_pred, target)
    return expected_rmse, mae

def rebuild(a):
    b = {}
    for u,v in a.items():
        b[int(u)] = v
    return b
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Social Recommendation: GraphRec model')
    parser.add_argument('--batch_size', type=int, default=2048+1024, metavar='N', help='input batch size for training')
    parser.add_argument('--embed_dim', type=int, default=256, metavar='N', help='embedding size')
    parser.add_argument('--phase', type=str, default="test", metavar='N', help='test phase')
    parser.add_argument('--dataset', type=str, default="80", metavar='N', help='80 for 80-10-10 split dataset, 60 for 60-20-20 split dataset')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N', help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    embed_dim = args.embed_dim
    dir_data = './data/toy_dataset'

    path_data = dir_data + ".pickle"
    data_file = open(path_data, 'rb')
    history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, train_u, train_v, train_r, test_u, test_v, test_r, social_adj_lists, ratings_list = pickle.load(
        data_file)
    
    
    if args.dataset == "80":
        item_split = 'history_item_split.json'
        user_split = 'history_user_split.json'
        model_checkpoint = "checkpoint/checkpoint_256_graph_80.pth"
    elif args.dataset == "60":
        item_split = 'history_item_split2.json'
        user_split = 'history_user_split2.json'
        model_checkpoint = "checkpoint/checkpoint_256_graph_60.pth"
    social_adj_lists = {i:{} for i in range(36000)}
    ratings_list = {1.0:0,2.0:1, 3.0:2,4.0:3,5.0:4}
    
    with open(item_split, 'r') as fp:
        history_item = json.load(fp)
    with open(user_split, 'r') as fp:
        history_user = json.load(fp)
    with open('dataset_split.json', 'r') as fp:
        dataset = json.load(fp)
    with open('graph_data_v2.json', 'r') as fp:
        social_adj_lists = json.load(fp)
    social_adj_lists = rebuild(social_adj_lists)
    print(len(social_adj_lists))
    history_u_lists, history_ur_lists = rebuild(history_user["user_item"]),rebuild(history_user["user_rating"])
    history_v_lists, history_vr_lists = rebuild(history_item["item_user"]),rebuild(history_item["item_rating"])
    
    train_u, train_v, train_r = dataset[args.dataset]["train"]["user"],dataset[args.dataset]["train"]["item"],dataset[args.dataset]["train"]["rate"]
    test_u, test_v, test_r = dataset[args.dataset]["test"]["user"],dataset[args.dataset]["test"]["item"],dataset[args.dataset]["test"]["rate"]
    mean = np.mean(test_r)
    
    print("predict mean", np.mean(np.abs(test_r -mean)**2)**0.5)
    print("std" , np.std(test_r))
    print("mean" , np.mean(test_r))
    print("1", np.sum(np.array(test_r)==1))
    print("2", np.sum(np.array(test_r)==2))
    print("3", np.sum(np.array(test_r)==3))
    print("4", np.sum(np.array(test_r)==4))
    print("5", np.sum(np.array(test_r)==5))
    print(len(train_u),len(test_v),len(test_r))
    res = []
    for i in range(len(test_r)):
        res.append(np.mean(history_ur_lists[int(test_u[i])] ) )
    print("user avg ",np.mean(np.abs(np.array(test_r)- np.array(res))**2)**0.5)
    
    trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_u), torch.LongTensor(train_v),
                                              torch.FloatTensor(train_r))
    testset = torch.utils.data.TensorDataset(torch.LongTensor(test_u), torch.LongTensor(test_v),
                                             torch.FloatTensor(test_r))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True)
    num_users = 34389 #history_u_lists.__len__()
    num_items = history_v_lists.__len__()
    num_ratings = 7#ratings_list.__len__()
    print(num_items)
    u2e = nn.Embedding(num_users, embed_dim).to(device)
    v2e = nn.Embedding(num_items, embed_dim).to(device)
    r2e = nn.Embedding(num_ratings, embed_dim).to(device)

    # user feature
    # features: item * rating
    agg_u_history = UV_Aggregator(v2e, r2e, u2e, embed_dim, cuda=device, uv=True)
    enc_u_history = UV_Encoder(u2e, embed_dim, history_u_lists, history_ur_lists, agg_u_history, cuda=device, uv=True)
    # neighobrs
    

    # item feature: user * rating

    agg_v_history = UV_Aggregator(v2e, r2e, u2e, embed_dim, cuda=device, uv=False)
    enc_v_history = UV_Encoder(v2e, embed_dim, history_v_lists, history_vr_lists, agg_v_history, cuda=device, uv=False)
    
    agg_v_social = Social_Aggregator(lambda nodes: enc_v_history(nodes).t(), v2e, embed_dim, cuda=device)
    enc_v = Social_Encoder(lambda nodes: enc_v_history(nodes).t(), embed_dim, social_adj_lists, agg_v_social,
                           base_model=enc_v_history, cuda=device)

    # model
    graphrec = GraphRec(enc_u_history, enc_v, r2e).to(device)
    optimizer = torch.optim.Adam(graphrec.parameters(), lr=args.lr ,weight_decay=0e-5)

    graphrec.load_state_dict(torch.load(model_checkpoint), strict=False) #RMSprop , alpha=0.9
    graphrec.eval()
    expected_rmse, mae = test(graphrec, device, test_loader)
    print("rmse: %.4f, mae:%.4f " % (expected_rmse, mae))
    best_rmse = 9.81
    best_mae = 9.48
    endure_count = 0
    """
    for epoch in range(1, args.epochs + 1):

        train(graphrec, device, train_loader, optimizer, epoch, best_rmse, best_mae)
        expected_rmse, mae = test(graphrec, device, test_loader)
        # please add the validation set to tune the hyper-parameters based on your datasets.

        # early stopping (no validation set in toy dataset)
        if best_rmse > expected_rmse:
            best_rmse = expected_rmse
            torch.save(graphrec.state_dict(), "checkpoint/checkpoint_256_graph_60.pth")
            #best_mae = mae
            endure_count = 0
        if best_mae > mae:
            best_mae = mae
            torch.save(graphrec.state_dict(), "checkpoint/checkpoint_best_mae_256_graph_60.pth")
        else:
            endure_count += 1
        print("rmse: %.4f, mae:%.4f " % (expected_rmse, mae))
        torch.save(graphrec.state_dict(), "checkpoint.pth")
        #if endure_count > 5:
        #    break
    """

if __name__ == "__main__":
    main()

## 80-20 rmse: 0.81 - 0.52
#rmse/mae: 0.892515 / 0.664306

## 80-20 - item graph: 0.82 - 0.488s
