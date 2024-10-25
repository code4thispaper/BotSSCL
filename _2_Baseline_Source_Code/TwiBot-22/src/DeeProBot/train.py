from model import DeeProBot
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import classification_report
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import pandas as pd
import os
import json

def average_report(reports):
    outer_keys = ["0", "1", "macro avg", "weighted avg"]
    metric_keys = ["precision", "recall", "f1-score", "support"]
    # Generate final report
    final_report = {}
    for ok in outer_keys:
        final_report[ok] = {}
        for mk in metric_keys:
            final_report[ok][mk] = 0
    # Sum values
    for report in reports:
        for ok in outer_keys:
            for mk in metric_keys:
                final_report[ok][mk] += report[ok][mk]
    # Get average
    for ok in outer_keys:
        for mk in metric_keys:
            final_report[ok][mk] = final_report[ok][mk]/len(reports)
    # Add accuracy
    final_report['accuracy'] = 0
    for report in reports:
        final_report['accuracy'] += report['accuracy']
    # Average accuracy
    final_report['accuracy'] =  final_report['accuracy']/len(reports)
    return final_report
                

class training_data(Dataset):

    def __init__(self, root):
        self.root=root
        self.idx=torch.load(root+'train_idx.pt')
        self.des=torch.load(root+'des_tensor.pt')[:,self.idx,:]
        self.num_prop=torch.load(root+'num_properties_tensor.pt')[self.idx]
        self.label=torch.load(root+'label.pt')[self.idx]
        
    def __len__(self):
        return len(self.idx)
    
    def __getitem__(self,idx):
        des=self.des[:,idx,:]
        num_prop=self.num_prop[idx]
        label=self.label[idx]
        
        return des,num_prop,label

class val_data(Dataset):

    def __init__(self, root):
        self.root=root
        self.idx=torch.load(root+'val_idx.pt')
        self.des=torch.load(root+'des_tensor.pt')[:,self.idx,:]
        self.num_prop=torch.load(root+'num_properties_tensor.pt')[self.idx]
        self.label=torch.load(root+'label.pt')[self.idx]
        
    def __len__(self):
        return len(self.idx-1)
    
    def __getitem__(self,idx):
        des=self.des[:,idx,:]
        num_prop=self.num_prop[idx]
        label=self.label[idx]
        
        return des,num_prop,label
    
class test_data(Dataset):

    def __init__(self, root):
        self.root=root
        self.idx=torch.load(root+'test_idx.pt')
        self.des=torch.load(root+'des_tensor.pt')[:,self.idx,:]
        self.num_prop=torch.load(root+'num_properties_tensor.pt')[self.idx]
        self.label=torch.load(root+'label.pt')[self.idx]
        
    def __len__(self):
        return len(self.idx)
    
    def __getitem__(self,idx):
        des=self.des[:,idx,:]
        num_prop=self.num_prop[idx]
        label=self.label[idx]
        
        return des,num_prop,label

def init_weights(m):
    if type(m)==nn.Linear:
        nn.init.kaiming_uniform_(m.weight)

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def train(epoch):
    model.train()
    for des_tensor,num_prop,labels in tqdm(train_loader):
        des_tensor,num_prop,labels=des_tensor.to(device=device),num_prop.to(device=device),labels.to(device=device)
        output = model(des_tensor,num_prop)
        loss_train = loss(output, labels)
        acc_train = accuracy(output, labels)
        #acc_val = accuracy(output[val_idx], labels[val_idx])
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
    print('Epoch: {:04d}'.format(epoch+1),
        'loss_train: {:.4f}'.format(loss_train.item()),
        'acc_train: {:.4f}'.format(acc_train.item()),
        #'acc_val: {:.4f}'.format(acc_val.item()),
        )
    return acc_train,loss_train

def val():
    for des_tensor,num_prop,labels in tqdm(val_loader):
        des_tensor,num_prop,labels=des_tensor.to(device=device),num_prop.to(device=device),labels.to(device=device)
        output = model(des_tensor,num_prop)
        loss_test = loss(output, labels)
        acc_test = accuracy(output, labels)
    print("Test set results:",
            "val_loss= {:.4f}".format(loss_test.item()),
            "val_accuracy= {:.4f}".format(acc_test.item()),
            )
    
def test():
    all_confusion_matrices = []
    reports = []
    for _ in range(5):
        confusion_matrices = []
        for des_tensor,num_prop,labels in tqdm(test_loader):
            des_tensor,num_prop,labels=des_tensor.to(device=device),num_prop.to(device=device),labels.to(device=device)
            output = model(des_tensor,num_prop)
            output=output.max(1)[1].to('cpu').detach().numpy()
            label=labels.to('cpu').detach().numpy()
            confusion_matrices.append(metrics.confusion_matrix(label, output))
            report = classification_report(label, output, output_dict=True)
            # Fix up confusion matrix
            total_confusion_matrix = confusion_matrices[0]
            for i in range(1, len(confusion_matrices)):
                total_confusion_matrix = np.add(total_confusion_matrix, confusion_matrices[i])
            all_confusion_matrices.append(total_confusion_matrix)
            reports.append(report)
        
        # Fix up confusion matrix
        a1 = all_confusion_matrices[0]
        for i in range(1, len(all_confusion_matrices)):
            a1 = np.add(a1, all_confusion_matrices[i])

        # Create confusion matrix average
        average_confusion_matrix = np.rint(a1/len(all_confusion_matrices))
        average_confusion_matrix = average_confusion_matrix.astype(int)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = average_confusion_matrix, display_labels = [False, True])
        cm_display.plot()
        plt.title(f"[{args.dataset.capitalize()}] DeeProBot Confusion Matrix on Test Dataset Split")
        plt.savefig(f'{args.dataset}_cm.png')
        plt.clf()
        plt.cla()
        plt.close()

        # Make report
        final_json = {
            "report": {
                "model": average_report(reports)
            },
            'cm': average_confusion_matrix.tolist()
        }

        filename = f"{args.dataset}.json"
        with open(filename, 'w') as f:
            json.dump(final_json, f, indent=4)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help='name of the dataset used for training')
    args = parser.parse_args()
    device = 'cuda:0'
    embedding_size,dropout,lr,weight_decay=128,0.1,1e-3,1e-2
    batch_size=40
    root='./'+args.dataset+'/'
    train_loader = DataLoader(training_data(root), batch_size, shuffle=True)
    val_loader = DataLoader(val_data(root), batch_size, shuffle=True)
    test_loader = DataLoader(test_data(root), batch_size, shuffle=True)
    model=DeeProBot().to(device=device)
    loss=nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=weight_decay)
    
    model.apply(init_weights)

    epochs=200
    for epoch in range(epochs):
        train(epoch)
        val()
        
    test()
