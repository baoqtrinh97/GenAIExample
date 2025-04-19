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
            
            # Predict edges
            edge_pred = model.decode_edges(z, data.edge_index)
            edge_pred = edge_pred.view(-1)  # Flatten to match labels
            
            # Compute loss
            loss = criterion(edge_pred, data.edge_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}')

# --- Testing ---
def test_model(model, test_loader, criterion, device='cpu'):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            
            # Encode node features
            z = model.encode(data.x, data.edge_index)
            
            # Predict edges
            edge_pred = model.decode_edges(z, data.edge_index)
            edge_pred = edge_pred.view(-1)  # Flatten to match labels
            
            # Compute loss
            loss = criterion(edge_pred, data.edge_labels)
            total_loss += loss.item()
            
            # Compute accuracy
            predictions = (edge_pred > 0.5).float()  # Threshold at 0.5
            total_correct += (predictions == data.edge_labels).sum().item()
            total_samples += data.edge_labels.size(0)
    
    avg_loss = total_loss / len(test_loader)
    accuracy = total_correct / total_samples * 100  # Convert to percentage
    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")