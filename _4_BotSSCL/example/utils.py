import random
import numpy as np
import torch
import tqdm

def train_epoch(model, criterion, train_loader, optimizer, device, epoch):
    model.train()
    epoch_loss = 0.0
    batch = train_loader
    
    for anchor, positive in batch:
        anchor, positive = anchor.to(device), positive.to(device)

        # reset gradients
        optimizer.zero_grad()

        # get embeddings
        emb_anchor, emb_positive = model(anchor, positive)

        # compute loss
        loss = criterion(emb_anchor, emb_positive).to(device)
        loss.backward()

        # update model weights
        optimizer.step()

        # log progress
        epoch_loss += anchor.size(0) * loss.item()

    return epoch_loss / len(train_loader.dataset)


def dataset_embeddings(model, loader, device):
    model.eval()
    embeddings = []

    with torch.no_grad():
        for anchor, _ in loader:
            anchor = anchor.to(device)
            embeddings.append(model.get_embeddings(anchor))

    embeddings = torch.cat(embeddings).cpu().numpy()

    return embeddings


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)