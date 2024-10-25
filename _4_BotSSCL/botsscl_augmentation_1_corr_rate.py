# Import required Python libraries
import os
import pandas as pd
import numpy as np
import torch
import json
import sys
import time

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


def run(dataset, input_embedding_dimension, output_embedding_dimension, corr_rate):

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

    train_data, test_data, train_target, test_target = train_test_split(
        rep, 
        target, 
        test_size=0.2, 
        stratify=target, 
        random_state=seed
    )

    # to torch dataset
    train_ds = TwitterAccountDatasetAug1(
        train_data, 
        train_target
    )
    test_ds = TwitterAccountDatasetAug1(
        test_data, 
        test_target
    )

    # # Training

    # In[ ]:

    a = time.perf_counter()

    batch_size = 512
    epochs = 5000
    device = torch.device("cuda")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)

    model = BotSSCL(
        input_dim=input_embedding_dimension * 4, 
        emb_dim=output_embedding_dimension,
        corruption_rate=corr_rate,    #0.6 original
    ).to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    ntxent_loss = NTXent()

    loss_history = []

    for epoch in range(1, epochs + 1):
        epoch_loss = train_epoch(model, ntxent_loss, train_loader, optimizer, device, epoch)
        loss_history.append(epoch_loss)
    
    b = time.perf_counter()

    time_1 = b - a

    # In[ ]:

    torch.save(model, f"aug1/pretrained_model/model_{input_embedding_dimension}x{output_embedding_dimension}_{dataset}_{corr_rate}.pt")


    # In[ ]:

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(loss_history)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    fig.savefig(f'aug1/loss/loss_{input_embedding_dimension}x{output_embedding_dimension}_{dataset}_{corr_rate}.png', bbox_inches='tight', dpi=300)

    plt.clf()
    plt.close()

    # # Evaluate embeddings

    # In[ ]:

    a = time.perf_counter()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # get embeddings for training and test set
    train_embeddings = dataset_embeddings(model, train_loader, device)
    test_embeddings = dataset_embeddings(model, test_loader, device)

    b = time.perf_counter()

    time_2 = time.perf_counter()

    # ### Prediction on original data

    # In[ ]:

    clf = LogisticRegression(solver='lbfgs', class_weight='balanced', max_iter=1000)
    # vanilla dataset: train the classifier on the original data
    clf.fit(train_data, train_target)
    vanilla_predictions = clf.predict(test_data)
    report_1 = classification_report(test_target, vanilla_predictions, output_dict=True)
    cm_1 = confusion_matrix(test_target, vanilla_predictions).tolist()


    # In[ ]:

    clf = LogisticRegression(solver='lbfgs', max_iter=1000)
    # vanilla dataset: train the classifier on the original data
    clf.fit(train_data, train_target)
    vanilla_predictions = clf.predict(test_data)
    report_2 = classification_report(test_target, vanilla_predictions, output_dict=True)
    cm_2 = confusion_matrix(test_target, vanilla_predictions).tolist()


    # In[ ]:

    clf = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 51)
    clf.fit(train_data, train_target)
    vanilla_predictions = clf.predict(test_data)
    report_3 = classification_report(test_target, vanilla_predictions, output_dict=True)
    cm_3 = confusion_matrix(test_target, vanilla_predictions).tolist()


    # In[ ]:

    a = time.perf_counter()

    linear_svc = LinearSVC(random_state=42, dual=False)
    linear_svc.fit(train_data, train_target)
    vanilla_predictions = linear_svc.predict(test_data)
    report_4 = classification_report(test_target, vanilla_predictions, output_dict=True)
    cm_4 = confusion_matrix(test_target, vanilla_predictions).tolist()

    b = time.perf_counter()

    time_3 = b - a

    # ### Prediction on embeddings

    # In[ ]:

    clf = LogisticRegression(solver='lbfgs', class_weight='balanced', max_iter=1000)

    # vanilla dataset: train the classifier on the original data
    clf.fit(train_embeddings, train_target)
    vanilla_predictions = clf.predict(test_embeddings)
    report_5 = classification_report(test_target, vanilla_predictions, output_dict=True)
    cm_5 = confusion_matrix(test_target, vanilla_predictions).tolist()


    # In[ ]:

    a = time.perf_counter()

    clf = LogisticRegression(solver='lbfgs', max_iter=1000)

    # vanilla dataset: train the classifier on the original data
    clf.fit(train_embeddings, train_target)
    vanilla_predictions = clf.predict(test_embeddings)
    report_6 = classification_report(test_target, vanilla_predictions, output_dict=True)
    cm_6 = confusion_matrix(test_target, vanilla_predictions).tolist()

    b = time.perf_counter()

    time_4 = b - a

    # In[ ]:

    a = time.perf_counter()

    linear_svc = LinearSVC(random_state=42, dual=False)
    linear_svc.fit(train_embeddings, train_target)
    vanilla_predictions = linear_svc.predict(test_embeddings)
    report_7 = classification_report(test_target, vanilla_predictions, output_dict=True)
    cm_7 = confusion_matrix(test_target, vanilla_predictions).tolist()

    b = time.perf_counter()

    time_5 = b - a

    # In[ ]:

    clf = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 51)
    clf.fit(train_embeddings, train_target)
    vanilla_predictions = clf.predict(test_embeddings)
    report_8 = classification_report(test_target, vanilla_predictions, output_dict=True)
    cm_8 = confusion_matrix(test_target, vanilla_predictions).tolist()

    # # Visualzie using t-sne 

    # In[ ]:

    tsne = TSNE(n_components=2, learning_rate="auto", init="pca", perplexity=15)
    reduced = tsne.fit_transform(train_embeddings)
    positive = train_target == 1
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(reduced[positive, 0], reduced[positive, 1], label="positive")
    ax.scatter(reduced[~positive, 0], reduced[~positive, 1], label="negative")
    fig.savefig(f'aug1/tsne/tsne_train_embeddings_{input_embedding_dimension}x{output_embedding_dimension}_{dataset}_{corr_rate}.png', bbox_inches='tight', dpi=300)
    plt.legend()


    # In[ ]:

    tsne = TSNE(n_components=2, learning_rate="auto", init="pca", perplexity=15)
    reduced = tsne.fit_transform(test_embeddings)
    positive = test_target == 1
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(reduced[positive, 0], reduced[positive, 1], label="positive")
    ax.scatter(reduced[~positive, 0], reduced[~positive, 1], label="negative")
    plt.legend()
    fig.savefig(f'aug1/tsne/tsne_test_embeddings_{input_embedding_dimension}x{output_embedding_dimension}_{dataset}_{corr_rate}.png', bbox_inches='tight', dpi=300)

    plt.clf()
    plt.close()

    # In[ ]:

    tsne = TSNE(n_components=2, learning_rate="auto", init="pca", perplexity=15)
    reduced = tsne.fit_transform(train_data)
    positive = train_target == 1
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(reduced[positive, 0], reduced[positive, 1], label="positive")
    ax.scatter(reduced[~positive, 0], reduced[~positive, 1], label="negative")
    plt.legend()
    fig.savefig(f'aug1/tsne/tsne_train_data_{input_embedding_dimension}x{output_embedding_dimension}_{dataset}_{corr_rate}.png', bbox_inches='tight', dpi=300)

    plt.clf()
    plt.close()

    # In[ ]:

    tsne = TSNE(n_components=2, learning_rate="auto", init="pca", perplexity=15)
    reduced = tsne.fit_transform(test_data)
    positive = test_target == 1
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(reduced[positive, 0], reduced[positive, 1], label="positive")
    ax.scatter(reduced[~positive, 0], reduced[~positive, 1], label="negative")
    plt.legend()
    fig.savefig(f'aug1/tsne/tsne_test_data_{input_embedding_dimension}x{output_embedding_dimension}_{dataset}_{corr_rate}.png', bbox_inches='tight', dpi=300)

    plt.clf()
    plt.close()

    final_json = {
        "reports": [report_1, report_2, report_3, report_4, report_5, report_6, report_7, report_8],
        "cm": [cm_1, cm_2, cm_3, cm_4, cm_4, cm_5, cm_6, cm_7, cm_8],
        "performance_time": [time_1, time_2, time_3, time_4, time_5]
    }

    with open(f"corruption_rate_experiments/{dataset}_{input_embedding_dimension}_{output_embedding_dimension}_{corr_rate}.json", "w") as f:
        json.dump(final_json, f, indent=4)


if __name__ == "__main__":

    dataset = sys.argv[1]
    corr_rate = [0.4,0.5,0.6,0.7,0.8]

    for i in range(len(corr_rate)):
        run(dataset, 16, 64, corr_rate[i])
