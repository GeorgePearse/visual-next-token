"""
SimCLR: A Simple Framework for Contrastive Learning of Visual Representations
Paper: https://arxiv.org/abs/2002.05709
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimCLR(nn.Module):
    """
    SimCLR model for self-supervised contrastive learning.
    """

    def __init__(self, encoder, temperature=0.5):
        """
        Args:
            encoder: Backbone network with projection head
            temperature: Temperature parameter for NT-Xent loss
        """
        super().__init__()
        self.encoder = encoder
        self.temperature = temperature

    def forward(self, x1, x2):
        """
        Forward pass with two augmented views.

        Args:
            x1: First augmented view
            x2: Second augmented view

        Returns:
            Projection outputs for both views
        """
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        return z1, z2

    def nt_xent_loss(self, z1, z2):
        """
        Normalized Temperature-scaled Cross Entropy Loss.

        Args:
            z1: Projections from first view [batch_size, proj_dim]
            z2: Projections from second view [batch_size, proj_dim]

        Returns:
            NT-Xent loss value
        """
        batch_size = z1.shape[0]

        # Normalize projections
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Concatenate projections
        z = torch.cat([z1, z2], dim=0)  # [2*batch_size, proj_dim]

        # Compute similarity matrix
        sim_matrix = torch.mm(z, z.T) / self.temperature  # [2*batch_size, 2*batch_size]

        # Create positive pair mask
        # For each sample i, the positive pair is at position i + batch_size (or i - batch_size)
        positive_mask = torch.zeros(
            (2 * batch_size, 2 * batch_size), dtype=torch.bool, device=z.device
        )
        for i in range(batch_size):
            positive_mask[i, batch_size + i] = True
            positive_mask[batch_size + i, i] = True

        # Create negative mask (all except self and positive pair)
        negative_mask = ~torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        negative_mask &= ~positive_mask

        # Compute loss for each sample
        losses = []
        for i in range(2 * batch_size):
            # Positive similarity
            pos_sim = sim_matrix[i][positive_mask[i]]

            # Negative similarities
            neg_sim = sim_matrix[i][negative_mask[i]]

            # Compute loss: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
            logits = torch.cat([pos_sim, neg_sim])
            labels = torch.zeros(logits.shape[0], dtype=torch.long, device=z.device)

            loss = F.cross_entropy(logits.unsqueeze(0), labels.unsqueeze(0))
            losses.append(loss)

        return torch.stack(losses).mean()


def simclr_loss(z1, z2, temperature=0.5):
    """
    Standalone function to compute SimCLR loss.

    Args:
        z1: Projections from first view [batch_size, proj_dim]
        z2: Projections from second view [batch_size, proj_dim]
        temperature: Temperature parameter

    Returns:
        NT-Xent loss value
    """
    batch_size = z1.shape[0]

    # Normalize
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # Concatenate
    representations = torch.cat([z1, z2], dim=0)

    # Similarity matrix
    similarity_matrix = (
        F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        / temperature
    )

    # Create labels: for a batch of N, positive pairs are (i, N+i) and (N+i, i)
    sim_i_j = torch.diag(similarity_matrix, batch_size)
    sim_j_i = torch.diag(similarity_matrix, -batch_size)

    positive_samples = torch.cat([sim_i_j, sim_j_i], dim=0).reshape(2 * batch_size, 1)

    # Mask to remove self-similarity
    mask = torch.ones((2 * batch_size, 2 * batch_size), dtype=torch.bool, device=z1.device)
    mask.fill_diagonal_(False)

    negative_samples = similarity_matrix[mask].reshape(2 * batch_size, -1)

    # Compute loss
    logits = torch.cat([positive_samples, negative_samples], dim=1)
    labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z1.device)

    loss = F.cross_entropy(logits, labels)
    return loss
