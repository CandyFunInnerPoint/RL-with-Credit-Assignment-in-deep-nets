import argparse
from utils import dataset, metrics
import torch
import os
import sys
import json


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-conf", "--conf", help="Configuration file", default='config.json')
    parser.add_argument("-out_path", "--out_path", help="Out path", default='./runs')
    parser.add_argument("-model_path", "--model_path", help="model path", default='./runs/pt_files/model.pt')
    args = parser.parse_args()
    return args


def eval(train_loader, validation_loader, model_path):
    model = torch.load(model_path)
    with torch.no_grad():

        train_acc_list = []
        for X, Y in train_loader:
            preds = model(X)
            train_acc_list.append(metrics.get_batch_accuracy(preds.numpy(), Y.numpy()))

        train_acc = sum(train_acc_list) / len(train_acc_list)

        val_acc_list = []
        for X, Y in validation_loader:
            preds = model(X)
            val_acc_list.append(metrics.get_batch_accuracy(preds.numpy(), Y.numpy()))

        val_acc = sum(val_acc_list) / len(val_acc_list)

        print('Validation accuracy:', val_acc, 'Train accuracy:', train_acc)

        return train_acc, val_acc


if __name__ == "__main__":

    args = get_args()

    config_path = args.conf
    out_folder = args.out_path
    model_path = args.model_path

    if not os.path.exists(config_path):
        print("Configuration file not found")
        sys.exit()

    if not os.path.exists(model_path):
        print("Model .pt file not found")
        sys.exit()

    with open(config_path, "r") as conf:
        conf = json.load(conf)

    d = dataset.DataSet(conf=conf)

    train_loader, validation_loader = d.get_binary_dataloaders()

    eval(train_loader, validation_loader, model_path)
