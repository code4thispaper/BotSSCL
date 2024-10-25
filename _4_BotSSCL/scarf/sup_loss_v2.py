import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXent(nn.Module):
    def __init__(self, temperature=1.0):
        """NT-Xent loss for contrastive learning using cosine distance as similarity metric as used in [SimCLR](https://arxiv.org/abs/2002.05709).
        Implementation adapted from https://theaisummer.com/simclr/#simclr-loss-implementation

        Args:
            temperature (float, optional): scaling factor of the similarity metric. Defaults to 1.0.
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j, anchor_labels):
        """Compute NT-Xent loss using only anchor and positive batches of samples. Negative samples are the 2*(N-1) samples in the batch

        Args:
            z_i (torch.tensor): anchor batch of samples
            z_j (torch.tensor): positive batch of samples

        Returns:
            float: loss
        """
        batch_size = z_i.size(0)
        label_size = anchor_labels.size(0)
#         print(label_size)
        
        # Preparing the mask for labels
        anchor_labels = anchor_labels.contiguous().view(-1, 1)
#         print(anchor_labels.shape)
        
        labels = torch.eq(anchor_labels, anchor_labels.T).float().to("cuda")
#         print(labels.shape)
        label_ = torch.cat([labels, labels], dim=-1)
        label_ = torch.cat([label_, label_], dim=0)
        
#         print(label_.shape)
        
        mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool)).float().to("cuda")
#         print(mask.shape)
        
        label_mask = mask * label_

        # compute similarity between the sample's embedding and its corrupted view
        z = torch.cat([z_i, z_j], dim=0)
        similarity = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
        
        
#       Preparing anti-label mask for denominator
        labels = torch.eq(anchor_labels, anchor_labels.T)
        denominator = ~(torch.tensor(label_mask).to(torch.bool)) * torch.exp(similarity / self.temperature)
        denominator = mask * denominator
        
        positives = torch.exp((label_mask * denominator)/self.temperature)
        right_log_term = torch.log(positives/torch.sum(denominator, dim=1))
        all_losses = -torch.sum(right_log_term / torch.sum(label_mask)) 

#         all_losses = -(torch.log(numerator / torch.sum(denominator, dim=1)))/normalization_denominator
        loss = torch.sum(all_losses) / (2 * batch_size)

        return loss
