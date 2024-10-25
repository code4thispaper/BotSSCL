import os
from argparse import ArgumentParser
from dataset import SATARDataset
import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import SATAR, BotClassifier
from utils import calc_metrics, calc_metrics_final
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--n_hidden', type=int, default=128)
parser.add_argument('--n_batch', type=int, default=32)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--max_tweet_count', type=int, default=128)
parser.add_argument('--max_tweet_length', type=int, default=64)
parser.add_argument('--max_words', type=int, default=1024)
args = parser.parse_args()

dataset_name = args.dataset
assert dataset_name in ['varol-icwsm', 'gilani-2017']
path = 'tmp/{}'.format(dataset_name)

n_hidden = args.n_hidden
n_batch = args.n_batch
dropout = args.dropout

max_tweet_count = args.max_tweet_count
max_tweet_length = args.max_tweet_length
max_words = args.max_words

begin_time = time.time()
data = {
    'tweets': np.load('{}/tweets.npy'.format(path), allow_pickle=True),
    'properties': np.load('{}/properties.npy'.format(path)),
    'neighbor_reps': np.load('{}/neighbor_reps.npy'.format(path)),
    'bot_labels': np.load('{}/bot_labels.npy'.format(path)),
    'follower_labels': np.load('{}/follower_labels.npy'.format(path))
}

word_vec = np.load('{}/vec.npy'.format(path))
word_vec = torch.tensor(word_vec)
words_size = len(word_vec)
blank_vec = torch.zeros((1, word_vec.shape[-1]))
word_vec = torch.cat((word_vec, blank_vec), dim=0)
num_embeddings = word_vec.shape[0]
embedding_dim = word_vec.shape[-1]
embedding_layer = nn.Embedding(num_embeddings, embedding_dim)
embedding_layer.weight.data = word_vec
embedding_layer.weight.requires_grad = False
embedding_layer.to(device)
print('loading done in {}s'.format(time.time() - begin_time))


def forward_one_batch(batch):
    return classifier(model(batch))


if __name__ == '__main__':
    test_set = SATARDataset(dataset_name,
                            split=['test'],
                            data=data,
                            padding_value=num_embeddings - 1,
                            max_words=max_words,
                            max_tweet_count=max_tweet_count,
                            max_tweet_length=max_tweet_length)
    test_loader = DataLoader(test_set, batch_size=n_batch, shuffle=False)
    checkpoints = os.listdir('tmp/checkpoints')
    model = SATAR(hidden_dim=n_hidden, embedding_dim=embedding_dim, dropout=dropout).to(device)
    classifier = BotClassifier(in_dim=n_hidden, out_dim=2).to(device)
    for name in checkpoints:
        if name.find(dataset_name) == -1:
            continue
        checkpoint = torch.load('tmp/checkpoints/{}'.format(name), map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        classifier.load_state_dict(checkpoint['classifier'])
        with torch.no_grad():
            model.eval()
            classifier.eval()
            all_label = []
            all_pred = []
            for batch in test_loader:
                batch_size = batch['bot_labels'].shape[0]
                out = forward_one_batch({
                    'words': embedding_layer(batch['words'].to(device)),
                    'tweets': embedding_layer(batch['tweets'].to(device)),
                    'neighbor_reps': batch['neighbor_reps'].to(device),
                    'properties': batch['properties'].to(device)
                })
                labels = batch['bot_labels'].to(device)
                all_label += labels.data
                all_pred += out
            all_label = torch.stack(all_label)
            all_pred = torch.stack(all_pred)
            report = calc_metrics_final(all_label, all_pred)

        # Make confusion matrix
        total_confusion_matrix = confusion_matrix(all_label.cpu().data, torch.argmax(all_pred, dim=-1).cpu().data)
        cm_display = ConfusionMatrixDisplay(total_confusion_matrix, display_labels = [False, True])
        cm_display.plot()
        plt.title(f"[{args.dataset.capitalize()}] SATAR Confusion Matrix on Test Dataset Split")
        plt.savefig(f'{args.dataset}_cm.png')
        plt.clf()
        plt.cla()
        plt.close()

        # Generate report
        # Make report
        final_json = {
            "report": {
                "model": report
            },
            'cm': total_confusion_matrix.tolist()
        }
        filename = f"{args.dataset}.json"
        with open(filename, 'w') as f:
            json.dump(final_json, f, indent=4)



