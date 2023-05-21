import json
import time
import torch
import torch.nn as nn
import os
import sys
import pandas as pd

from utils import dataset, metrics, plot
from model import critic_net, nets

import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-conf", "--conf", help="Configuration file", default='config.json')
    parser.add_argument("-out_path", "--out_path", help="Out path", default='./runs')
    args = parser.parse_args()
    return args


def save_data(out_dict, file_path):
    if os.path.exists(file_path):
        old_df = pd.read_csv(file_path)
        new_df = pd.DataFrame.from_dict(out_dict)
        out_df = pd.concat([old_df, new_df])
        out_df.to_csv(file_path, index=False)
    else:
        out_df = pd.DataFrame.from_dict(out_dict)
        out_df.to_csv(file_path, index=False)


def optimize_fl(layer, alpha):
    for param in layer.parameters():
        if param.requires_grad:
            g = param.grad.data
            param.data.add_(alpha * g)
    return param.grad.data


def optimize_q(critic, alpha, qtar, qtar_next):
    for param in critic.parameters():
        if param.requires_grad:
            g = (qtar_next - qtar) * param.grad.data
            param.data.add_(alpha * g)
    return g


def optimize(layer, alpha, qtar_next):
    for param in layer.parameters():
        if param.requires_grad:
            g = qtar_next * param.grad.data
            param.data.add_(alpha * g)
    return g


def train(conf, train_loader, validation_loader, pt_folder, csv_folder, fig_folder):
    seed = conf['q_weight_init']['seed']
    epochs = conf['hyp']['epochs']
    alpha = conf['hyp']['alpha']
    alpha_q = conf['hyp']['alpha_q']
    run_id = int(time.time())

    csv_name = f'history_alpha_{alpha}_q_alpha_{alpha_q}_seed_{seed}_{run_id}.csv'
    csv_path = os.path.join(csv_folder, csv_name)

    # Creating model and critics

    torch.manual_seed(seed)
    net = nets.Net()

    critic_list = []
    for i, layer in enumerate(net.layerlist):
        if i != len(net.layerlist) - 1:  # Final layer does not have a critic
            critic = critic_net.Critic(layer.in_features + layer.out_features, conf=conf)
            critic_list.append(critic)

    # Training loop
    o_list = [None] * len(net.layerlist)
    a_list = [None] * len(net.layerlist)
    max_val_acc = 0
    for epoch in range(epochs):

        train_acc_list = []
        val_acc_list = []
        train_g_list = []

        for i, (X, Y) in enumerate(train_loader):
            # Forward propergation through network
            o = X

            for i, (layer, act) in enumerate(zip(net.layerlist, net.layer_act_list)):
                o_list[i] = o
                a = act(layer(o))
                a_list[i] = a
                o = a_list[i].detach()

            yhat = a_list[-1]

            # G = -nn.functional.binary_cross_entropy(yhat, Y)
            G = -nn.functional.mse_loss(yhat, Y)

            train_acc_list.append(metrics.get_batch_accuracy(yhat.detach().numpy(), Y.numpy()))
            train_g_list.append(G.detach().numpy())

            # Backward propergation

            # Final layer
            G.backward()
            optimize_fl(net.layerlist[-1], alpha)
            net.layerlist[-1].zero_grad()
            q_next = G.detach()

            # Other layers

            for i, layer in enumerate(net.layerlist[0:-1][::-1]):
                layer_index = len(net.layerlist) - 2 - i

                critic_i = critic_list[layer_index]
                q = critic_i(torch.concat((o_list[layer_index], a_list[layer_index]), dim=1).detach())
                q.backward()
                optimize_q(critic_i, alpha_q, q, q_next)
                critic_i.zero_grad()

                ln_l = torch.log(a_list[layer_index])
                ln_l.backward(ln_l)
                optimize(layer, alpha, q_next)
                layer.zero_grad()

                q_next = q.detach()

        for i, (X, Y) in enumerate(validation_loader):
            with torch.no_grad():
                pred = net(X)
                val_acc_list.append(metrics.get_batch_accuracy(pred.numpy(), Y.numpy()))

        train_acc = round(sum(train_acc_list) / len(train_acc_list), 3)
        val_acc = round(sum(val_acc_list) / len(val_acc_list), 3)
        train_g = round(sum(train_g_list) / len(train_g_list), 3)

        print('epoch: ', epoch, 'train_acc: ', train_acc, 'val_acc: ', val_acc, 'train G: ', train_g)

        out_dict = {'epoch': [epoch],
                    'train_acc': [train_acc],
                    'val_acc': [val_acc],
                    'train G': [train_g],
                    'alpha': [alpha],
                    'q_alpha': [alpha_q],
                    'q_weight_seed': [seed]}

        save_data(out_dict, csv_path)

        if max_val_acc < val_acc:
            pt_file_name = f'acc_{val_acc}_alpha_{alpha}_q_alpha_{alpha_q}_seed_{seed}_{run_id}.pt'
            pt_file_path = os.path.join(pt_folder, pt_file_name)
            torch.save(net, pt_file_path)
            max_val_acc = val_acc

    fig_name = f'alpha_{alpha}_q_alpha_{alpha_q}_seed_{seed}_{run_id}.jpg'
    fig_path = os.path.join(fig_folder, fig_name)
    plot.plot_csv(csv_path, fig_path)


if __name__ == "__main__":

    args = get_args()

    config_path = args.conf
    out_folder = args.out_path

    csv_folder = os.path.join(out_folder, 'csvs')
    plot_folder = os.path.join(out_folder, 'plots')
    pt_folder = os.path.join(out_folder, 'pt_files')

    os.makedirs(csv_folder, exist_ok=True)
    os.makedirs(plot_folder, exist_ok=True)
    os.makedirs(pt_folder, exist_ok=True)

    if not os.path.exists(config_path):
        print("Configuration file not found")
        sys.exit()

    with open(config_path, "r") as conf:
        conf = json.load(conf)

    d = dataset.DataSet(conf=conf)

    train_loader, validation_loader = d.get_binary_dataloaders()

    train(conf, train_loader, validation_loader, pt_folder, csv_folder, plot_folder)
