# Import required Python libraries
import os
import pandas as pd
import numpy as np
import torch
import json
import sys
from time import time
from glob import glob

from torch.optim import Adam
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix

# from scarf.model_constraint import SCARF
from scarf.loss import NTXent
from scarf.model_botsscl_aug_1 import BotSSCL

from example.dataset import TwitterAccountDatasetAug1
from example.utils import fix_seed, train_epoch, dataset_embeddings


def run(dataset, user, index, input_embedding_dimension, output_embedding_dimension):

    seed = 1234
    fix_seed(seed)

    # In[ ]:

    # Create directories
    dirs = ["aug1/loss", "aug1/tsne", "aug1/cm", "aug1/pretrained_model"]
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)

    # # Data

    # In[ ]:

    data = pd.read_csv(f"Data/{dataset}-clean.csv", index_col=0)
    target = data['label']

    # In[ ]:

    # Open input representation
    with open(f'Data/{dataset.split("-")[0]}_test_{input_embedding_dimension}_with_tweets.npy', 'rb') as f:
        rep = np.load(f)

    # In[ ]:

    # Get training data
    train_data, test_data, train_target, test_target = train_test_split(
        rep, 
        target, 
        test_size=0.2, 
        stratify=target, 
        random_state=seed
    )
    train_ds = TwitterAccountDatasetAug1(
        train_data, 
        train_target
    )

    # Get sample testing data
    with open(f'Data/{dataset}_test_{user}_{index}_{input_embedding_dimension}_with_sample_modified_tweets.npy', 'rb') as f:
        rep = np.load(f)
        print(rep.shape)
    test_target = [1] * rep.shape[0]
    test_ds = TwitterAccountDatasetAug1(
        rep, 
        test_target
    )

    # # Evaluate embeddings

    # In[ ]:

    # Preload model
    model = torch.load(f"aug1/pretrained_model/model_{input_embedding_dimension}x{output_embedding_dimension}_{dataset}.pt")

    batch_size = 512
    device = torch.device("cuda")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # get embeddings for training and test set
    train_embeddings = dataset_embeddings(model, train_loader, device)
    test_embeddings = dataset_embeddings(model, test_loader, device)

    # ### Prediction on embeddings

    # In[ ]:

    clf = LogisticRegression(solver='lbfgs', class_weight='balanced', max_iter=1000)

    clf.fit(train_embeddings, train_target)
    vanilla_predictions = clf.predict(test_embeddings)
    final_label = vanilla_predictions[-1]
    
    return final_label


if __name__ == "__main__":

    dataset = sys.argv[1]
    df = pd.read_csv(f'Data/{dataset}-clean-sample-modified-single-tweets.csv')
    
    if dataset == 'gilani-2017':
        input_dim = 64
    else:
        input_dim = 16
    
    output_dict = {}

    a = time()
    for user in df['id'].unique()[:20]:
        labels = [-1 for _ in range(200 + 1)]
        for i in range(df[df['id'] == user].shape[0]):
            label = run(dataset, user, i, input_dim, 64)
            labels[i] = label
            if label == 0:
                break
        output_dict[str(user)] = labels
    b = time()
    print(f"Time taken is {b - a}")

    output_df = pd.DataFrame.from_dict(output_dict, orient='index')
    output_df.reset_index(inplace=True)
    output_df.rename(columns={output_df.index.name:'id'}, inplace=True)
    output_df.to_csv(f"{dataset}-sample-modified-tweets-label-results-fast.csv", index=False)