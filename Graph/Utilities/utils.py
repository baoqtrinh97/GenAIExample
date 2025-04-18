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

def visualize_graph(nx_graph, type_encoding):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    plt.figure(figsize=(10, 8))

    # Create position layout for nodes
    pos = nx.spring_layout(nx_graph, seed=42)

    # Reverse mapping: idx -> type name
    idx_to_type = {idx: type_name for type_name, idx in type_encoding.items()}

    # Get node type indices for coloring
    node_type_indices = [nx_graph.nodes[node]['type'] for node in nx_graph.nodes()]

    # Create a color map based on type_encoding
    cmap = cm.get_cmap('tab10', len(type_encoding))
    color_map = {idx: cmap(idx) for idx in idx_to_type.keys()}
    node_colors = [color_map[type_idx] for type_idx in node_type_indices]

    # Draw the graph
    nx.draw(nx_graph, pos, with_labels=True, node_color=node_colors, node_size=500, font_size=10)

    # Add legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[idx], markersize=10, label=type_name)
               for idx, type_name in idx_to_type.items()]
    plt.legend(handles=handles, title="Room Types")

    plt.tight_layout()
    plt.show()

def analyze_graph_dataset(graph_data):
    """
    Analyze the dataset of graphs 

    Args:
        graph_data: List of graph dictionaries containing graph information.

    Returns:
        node_type_counts: Dictionary mapping node types to their counts.
        type_encoding: Dictionary mapping node types to unique indices based on frequency.
        filtered_graph_data: List of graphs with all valid nodes (no null or invalid nodes).
    """
    node_types = []
    node_type_counts = defaultdict(int)
    for graph_item in graph_data:
        if "Graph" in graph_item:
            graph_info = graph_item["Graph"]
            if "Nodes" in graph_info:
                for node in graph_info["Nodes"]:
                    if "Properties" in node:
                        node_props = node["Properties"]
                        node_type = node_props.get("Type")
                        if node_type is not None:
                            node_types.append(node_type)
                            node_type_counts[node_type] += 1

    type_encoding = {}
    for idx, (node_type, _) in enumerate(sorted(node_type_counts.items(), key=lambda x: x[1], reverse=True)):
        type_encoding[node_type] = idx
    total_graphs = 0
    graphs_with_all_valid_nodes = 0
    graphs_with_some_null_nodes = 0
    graphs_with_all_null_nodes = 0
    indices_with_null_nodes = []
    for i, graph_item in enumerate(graph_data):
        if "Graph" in graph_item:
            total_graphs += 1
            graph_info = graph_item["Graph"]
            total_nodes = 0
            null_nodes = 0
            if "Nodes" in graph_info:
                for node in graph_info["Nodes"]:
                    if "Properties" in node:
                        total_nodes += 1
                        node_props = node["Properties"]
                        node_type = node_props.get("Type")
                        if node_type is None or node_type not in type_encoding:
                            null_nodes += 1
            if total_nodes == 0:
                graphs_with_all_null_nodes += 1
                indices_with_null_nodes.append(i)
            elif null_nodes == total_nodes:
                graphs_with_all_null_nodes += 1
                indices_with_null_nodes.append(i)
            elif null_nodes > 0:
                graphs_with_some_null_nodes += 1
                indices_with_null_nodes.append(i)
            else:
                graphs_with_all_valid_nodes += 1
    filtered_graph_data = [g for i, g in enumerate(graph_data) if i not in indices_with_null_nodes]
    return node_type_counts, type_encoding, filtered_graph_data

def build_graph_list(graph_data, type_encoding):
    """
    Build a list of NetworkX graphs from the dataset, with all node and edge attributes.

    Args:
        graph_data: List of graph dictionaries containing graph information.
        type_encoding: Dictionary mapping node types to unique indices.
       
    Returns:
        graph_list: List of NetworkX graphs.
    """
    import networkx as nx
    from collections import defaultdict

    unique_types = list(type_encoding.keys())

    graph_list = []

    for graph_item in graph_data:
        if "Graph" in graph_item:
            graph_info = graph_item["Graph"]
            G = nx.Graph()
            apartment_width = graph_info.get("Attributes", {}).get("Width", 7)
            apartment_length = graph_info.get("Attributes", {}).get("Length", 10)
            apartment_area = apartment_width * apartment_length
            temp_counts = defaultdict(int)

            # process the graph information
            if "Nodes" in graph_info:
                for node in graph_info["Nodes"]:
                    if "Properties" in node:
                        node_type = node["Properties"].get("Type")
                        if node_type is not None:
                            temp_counts[node_type] += 1
            program_vector = {room_type: temp_counts.get(room_type, 0) for room_type in unique_types}
            G.graph['width'] = apartment_width
            G.graph['length'] = apartment_length
            G.graph['area'] = apartment_area
            G.graph['program_vector'] = program_vector

            # process the nodes
            if "Nodes" in graph_info:
                for node in graph_info["Nodes"]:
                    if "Properties" in node:
                        node_props = node["Properties"]
                        node_id = node_props.get("Id")
                        node_type = node_props.get("Type")
                        if node_id is not None and node_type is not None:
                            G.add_node(
                                node_id,
                                name=node_type,  # Add the name separately
                                type=type_encoding[node_type],  # Only the idx
                                x=node_props.get("Point", {}).get("X"),  # 1st feature
                                y=node_props.get("Point", {}).get("Y"),  # 2nd feature
                                apartment_width=apartment_width,  # 3rd feature
                                apartment_length=apartment_length,  # 4th feature
                                apartment_area=apartment_area,  # 5th feature
                                program_vector=program_vector  # 6th feature
                            )
                            
            # process the edges
            if "Edges" in graph_info:
                for edge in graph_info["Edges"]:
                    if "Properties" in edge:
                        edge_props = edge["Properties"]
                        source_id = edge_props.get("SourceId")
                        target_id = edge_props.get("TargetId")
                        edge_length = edge_props.get("Length") # edge feature
                        if edge_length is None and source_id is not None and target_id is not None:
                            try:
                                src = G.nodes[source_id]
                                tgt = G.nodes[target_id]
                                edge_length = ((src['x'] - tgt['x']) ** 2 + (src['y'] - tgt['y']) ** 2) ** 0.5
                            except Exception:
                                edge_length = 0.0
                        if source_id is not None and target_id is not None:
                            G.add_edge(source_id, target_id, length=edge_length)

            # add graph to the list
            if len(G.nodes) > 0:
                graph_list.append(G)
    return graph_list

def convert_nx_to_pyg(nx_graph, type_encodings):
    """
    Converts a NetworkX graph into a PyTorch Geometric Data object.
    """
    try:
        unique_types = list(type_encodings.keys())
        num_types = len(type_encodings)
        x = []
        for node, data in nx_graph.nodes(data=True):
            # Only skip if 'type' is missing
            if 'type' not in data:
                continue
            type_idx = data['type']  # Now this is just an int
            node_x = data['x']
            node_y = data['y']
            width = nx_graph.graph['width']
            length = nx_graph.graph['length']
            area = nx_graph.graph['area']
            program_vector = [nx_graph.graph['program_vector'].get(room_type, 0) for room_type in unique_types]

            # One-hot encode the node type
            one_hot = F.one_hot(torch.tensor(type_idx), num_classes=num_types).tolist()

            # Concatenate all features for the node
            features = one_hot + [node_x, node_y, width, length, area] + program_vector
            x.append(features)
        if not x:
            return None
        x_tensor = torch.tensor(x, dtype=torch.float)
        
        edge_index = []
        edge_attr = []
        node_idx_map = {node: idx for idx, node in enumerate(nx_graph.nodes())}  # Map node IDs to indices for PyG
        for source, target, edge_data in nx_graph.edges(data=True):
            # Only add edges if both nodes exist in the mapping
            if source in node_idx_map and target in node_idx_map:
                idx_u = node_idx_map[source]
                idx_v = node_idx_map[target]
                # Add both directions for undirected graph
                edge_index.append([idx_u, idx_v])
                edge_index.append([idx_v, idx_u])
                length_val = edge_data.get('length', 0.0)  # Use edge length if available, else 0.0
                edge_attr.append([length_val])
                edge_attr.append([length_val])
        if len(edge_index) > 0:
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t()  # Shape: [2, num_edges]
            edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float)       # Shape: [num_edges, 1]
            data_obj = Data(
                x=x_tensor,  # Node features
                edge_index=edge_index_tensor,  # Edge indices
                edge_attr=edge_attr_tensor,    # Edge attributes (length)
                area=torch.tensor([nx_graph.graph['area']], dtype=torch.float),  # Graph-level area
                apartment_width=torch.tensor([nx_graph.graph['width']], dtype=torch.float),  # Graph-level width
                apartment_length=torch.tensor([nx_graph.graph['length']], dtype=torch.float),  # Graph-level length
                program_vector=torch.tensor([program_vector], dtype=torch.float)  # Graph-level program vector
            )
            return data_obj
    except Exception as e:
        # Print error and return None if graph conversion fails
        print(f"Failed to process graph: {str(e)}")
        return None

def convert_nx_to_pyg_2(nx_graph, type_encodings):
    """
    Converts a NetworkX graph into a PyTorch Geometric Data object.
    Only includes node type (one-hot), area, and program_vector as node features.
    Edge attributes are omitted.
    """
    try:
        unique_types = list(type_encodings.keys())
        num_types = len(type_encodings)
        x = []
        area = nx_graph.graph['area']
        program_vector = [nx_graph.graph['program_vector'].get(room_type, 0) for room_type in unique_types]
        for node, data in nx_graph.nodes(data=True):
            if 'type' not in data:
                continue
            type_idx = data['type']
            # One-hot encode the node type
            one_hot = F.one_hot(torch.tensor(type_idx), num_classes=num_types).tolist()
            # Only keep type, area, and program_vector
            features = one_hot + [area] + program_vector
            x.append(features)
        if not x:
            return None
        x_tensor = torch.tensor(x, dtype=torch.float)

        edge_index = []
        node_idx_map = {node: idx for idx, node in enumerate(nx_graph.nodes())}
        for source, target in nx_graph.edges():
            if source in node_idx_map and target in node_idx_map:
                idx_u = node_idx_map[source]
                idx_v = node_idx_map[target]
                # Add both directions for undirected graph
                edge_index.append([idx_u, idx_v])
                edge_index.append([idx_v, idx_u])
        if len(edge_index) > 0:
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t()
            data_obj = Data(
                x=x_tensor,
                edge_index=edge_index_tensor,
                area=torch.tensor([area], dtype=torch.float),
                program_vector=torch.tensor([program_vector], dtype=torch.float)
            )
            return data_obj
    except Exception as e:
        print(f"Failed to process graph: {str(e)}")
        return None

def get_node_type(features, type_encodings, reverse_encodings):
    """
    Determines the type of a node based on its features and encoding mappings.

    Returns:
        str: The name of the node type if it can be determined from the reverse_encodings dictionary.
             Returns 'unknown' if the type code is not found in the dictionary.
    """
    one_hot = features[:len(type_encodings)]
    type_code = int(np.argmax(one_hot))
    return reverse_encodings.get(type_code, 'unknown')

def generate_new_fully_connected_graph(required_nodes, apartment_width, apartment_length, num_additional_nodes, type_encodings, type_reverse_encodings):
    """
    Generates a new NetworkX graph based on the required nodes and apartment parameters.
    The node and graph attributes will match those created in build_graph_list for compatibility.

    Args:
        required_nodes: Dict of node type names and their required counts.
        apartment_width: Width of the apartment.
        apartment_length: Length of the apartment.
        num_additional_nodes: Number of extra nodes to add.
        type_encodings: Dict mapping node type names to indices.
        type_reverse_encodings: Dict mapping indices to node type names.

    Returns:
        G: A generated NetworkX graph.
    """
    import networkx as nx
    from collections import defaultdict

    num_fixed_nodes = sum(required_nodes.values())
    total_nodes = num_fixed_nodes + num_additional_nodes
    unique_types = list(type_encodings.keys())
    program_vector = {room_type: required_nodes.get(room_type, 0) for room_type in unique_types}

    G = nx.Graph()
    G.graph['width'] = apartment_width
    G.graph['length'] = apartment_length
    G.graph['area'] = apartment_width * apartment_length
    G.graph['program_vector'] = program_vector

    # Add required nodes with their types and random positions
    node_idx = 0
    for node_type, count in required_nodes.items():
        for _ in range(count):
            x_coord = float(random.random())
            y_coord = float(random.random())
            G.add_node(
                node_idx,
                name=node_type,
                type=type_encodings[node_type],
                x=x_coord,
                y=y_coord,
                apartment_width=apartment_width,
                apartment_length=apartment_length,
                apartment_area=apartment_width * apartment_length,
                program_vector=program_vector,
                fixed=True
            )
            node_idx += 1

    # Add additional random nodes
    available_types = list(type_encodings.keys())
    for _ in range(num_additional_nodes):
        selected_type = random.choice(available_types)
        x_coord = float(random.random())
        y_coord = float(random.random())
        G.add_node(
            node_idx,
            name=selected_type,
            type=type_encodings[selected_type],
            x=x_coord,
            y=y_coord,
            apartment_width=apartment_width,
            apartment_length=apartment_length,
            apartment_area=apartment_width * apartment_length,
            program_vector=program_vector,
            fixed=False
        )
        node_idx += 1

    # Fully connect the graph (add an edge between every pair of nodes)
    for i in range(total_nodes):
        for j in range(i + 1, total_nodes):
            src = G.nodes[i]
            tgt = G.nodes[j]
            edge_length = ((src['x'] - tgt['x']) ** 2 + (src['y'] - tgt['y']) ** 2) ** 0.5
            G.add_edge(i, j, length=edge_length)

    return G