import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from argparse import ArgumentParser
import json
from sklearn.metrics import classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import metrics

parser = ArgumentParser()
parser.add_argument('--dataset', type=str)
args = parser.parse_args()

dataset = args.dataset

assert dataset in ['twibot-s1', 'twibot-s2', 'varol-icwsm', 'gilani-2017']

split = pd.read_csv('../../datasets/{}/split.csv'.format(dataset))
idx = json.load(open('tmp/{}/idx.json'.format(dataset)))
idx = {item: index for index, item in enumerate(idx)}
features = np.load('tmp/{}/features.npy'.format(dataset), allow_pickle=True)
labels = np.load('tmp/{}/labels.npy'.format(dataset))

train_idx = []
val_idx = []
test_idx = []

for index, item in tqdm(split.iterrows(), ncols=0):
    try:
        if item['split'] == 'train':
            train_idx.append(idx[item['id']])
        if item['split'] == 'val' or item['split'] == 'valid':
            val_idx.append(idx[item['id']])
        if item['split'] == 'test':
            test_idx.append(idx[item['id']])
    except KeyError:
        continue

print('loading done')

print(len(train_idx))
print(len(val_idx))
print(len(test_idx))


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
                

if __name__ == '__main__':

    train_x = features[train_idx]
    train_y = labels[train_idx]
    val_x = features[val_idx]
    val_y = labels[val_idx]
    test_x = features[test_idx]
    test_y = labels[test_idx]
    
    print('training......')
    no_runs = 100
    confusion_matrices = []
    classification_reports = []
    for _ in range(no_runs):
        cls = RandomForestClassifier(n_estimators=100)
        cls.fit(train_x, train_y)
        # Do predictions
        test_pred = cls.predict(test_x)
        report = classification_report(test_y, test_pred, output_dict=True)
        classification_reports.append(report)
        confusion_matrices.append(metrics.confusion_matrix(test_y, test_pred))
        del cls

    # Fix up confusion matrix
    a1 = confusion_matrices[0]
    for i in range(1, len(confusion_matrices)):
        a1 = np.add(a1, confusion_matrices[i])

    # Create confusion matrix average
    average_confusion_matrix = np.rint(a1/len(confusion_matrices))
    average_confusion_matrix = average_confusion_matrix.astype(int)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = average_confusion_matrix, display_labels = [False, True])
    
    cm_display.plot()
    plt.title(f"[{dataset.capitalize()}] Bothunter Confusion Matrix on Test Dataset Split")
    plt.savefig(f"{dataset}_cm.png")
    plt.clf()
    plt.cla()
    plt.close()

    final_json = {
        "report": {
            "model": average_report(classification_reports)
        },
        'cm': average_confusion_matrix.tolist()
    }
    filename = f"{dataset}.json"
    with open(filename, 'w') as f:
        json.dump(final_json, f, indent=4)

    

    