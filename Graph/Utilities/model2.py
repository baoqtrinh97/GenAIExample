import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import json
import networkx as nx
from collections import defaultdict
import random
import matplotlib.pyplot as plt
import numpy as np

class GraphGenerator(nn.Module):
    def __init__(self, node_features, hidden_dim=64):
        super(GraphGenerator, self).__init__()
        self.hidden_dim = hidden_dim
        self.conv1 = GATConv(node_features, hidden_dim, heads=1)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=1)
        self.conv3 = GATConv(hidden_dim, hidden_dim, heads=1)
        self.node_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_features)
        )
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        return x

    def decode_edges(self, z, edge_index, edge_attr=None):
        row, col = edge_index
        edge_features = torch.cat([z[row], z[col]], dim=1)
        if edge_attr is not None:
            edge_features = torch.cat([edge_features, edge_attr], dim=1)
        return self.edge_predictor(edge_features)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        z = self.encode(x, edge_index)
        new_node_features = self.node_predictor(z)
        edge_pred = self.decode_edges(z, edge_index)
        return new_node_features, edge_pred

def negative_sampling(edge_index, num_nodes, num_neg_samples=None):
    """
    Generates negative samples for edges in a graph.

    Negative sampling is the process of generating edges that do not exist in the graph,
    which can be used for training graph-based machine learning models.

    Args:
        edge_index (torch.Tensor): A tensor of shape (2, num_edges) representing the existing edges in the graph.
                                   Each column represents an edge (source, target).
        num_nodes (int): The total number of nodes in the graph.
        num_neg_samples (int, optional): The number of negative samples to generate. If not provided, it defaults
                                         to the number of existing edges.

    Returns:
        torch.Tensor: A tensor of shape (2, num_neg_samples) representing the negative edges. Each column
                      represents a negative edge (source, target), and the tensor is on the same device as `edge_index`.

    Note:
        - Negative edges are sampled uniformly from all possible edges that do not exist in the graph.
        - If the number of possible negative edges is less than `num_neg_samples`, all possible negative edges
          are returned without sampling.
    """
    num_neg_samples = num_neg_samples or edge_index.size(1)
    all_edges = set()
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                all_edges.add((i, j))
    existing_edges = set(tuple(e) for e in edge_index.t().tolist())
    candidate_edges = list(all_edges - existing_edges)
    if len(candidate_edges) <= num_neg_samples:
        neg_edges = candidate_edges
    else:
        neg_edges = random.sample(candidate_edges, num_neg_samples)
    neg_edge_index = torch.tensor(neg_edges, dtype=torch.long).t()
    return neg_edge_index.to(edge_index.device)


def train_model(model, train_loader, optimizer, criterion, num_epochs=100, device='cpu'):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            # Encode node features
            z = model.encode(data.x, data.edge_index)
            
            # Positive edges
            pos_edge_index = data.edge_index
            num_nodes = data.num_nodes
            num_pos = pos_edge_index.size(1)
            
            # Negative edges
            neg_edge_index = negative_sampling(pos_edge_index, num_nodes, num_neg_samples=num_pos)
            
            # Combine edges and labels
            edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
            edge_labels = torch.cat([
                torch.ones(num_pos, device=device),   # Positive
                torch.zeros(num_pos, device=device)   # Negative
            ])
            
            # Predict edges
            edge_pred = model.decode_edges(z, edge_index)
            edge_pred = edge_pred.view(-1)  # Flatten to match labels
            
            loss = criterion(edge_pred, edge_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}')

def classify_edges(model, pyg_graph, threshold=0.5, device='cpu'):
    """
    Classifies the edges of a PyG graph using the trained model.

    Args:
        model: Trained GraphGenerator model.
        pyg_graph: A PyG Data object.
        threshold: Probability threshold for classifying an edge as positive.
        device: Device to run the model on.

    Returns:
        edge_probs: Tensor of predicted probabilities for each edge.
        edge_pred_labels: Tensor of predicted labels (0 or 1) for each edge.
    """
    model.eval()
    pyg_graph = pyg_graph.to(device)
    with torch.no_grad():
        z = model.encode(pyg_graph.x, pyg_graph.edge_index)
        edge_probs = model.decode_edges(z, pyg_graph.edge_index).view(-1)
        edge_pred_labels = (edge_probs >= threshold).long()
    return edge_probs.cpu(), edge_pred_labels.cpu()
