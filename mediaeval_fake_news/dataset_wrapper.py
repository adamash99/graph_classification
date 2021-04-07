from torch.utils.data import Dataset
import torch as th
import pandas as pd
import numpy as np


class GraphDataset(Dataset):
    def __init__(self, df, embeddings, node_df, transform=None, target_transform=None):
        self.df = df
        self.embeddings = embeddings
        self.node_df = node_df
        self.transform = transform
        self.target_transform = target_transform
        
        self.num_points = df.shape[0]
        self.labels = df['label']
        self.label_count = df['label'].nunique()

    def __len__(self):
        return self.num_points
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx, :]
        if self.transform != None:
            feats = self.transform(row, self.embeddings, self.node_df)
        else:
            raise Exception('no x transform provided')
        
        target = row['label']
        if self.target_transform != None:
            target = self.target_transform(target, self.label_count)
            
        return (feats, target)

    def num_classes(self):
        return len(self.labels.unique())

def x_transform(row, embeddings, node_df):
    """
    row:    row of the original df corresponding to the graph being transformed
    """
    nodes = row['nodeID_list']
    node_embeddings = embeddings[[(node in nodes) for node in embeddings.index]]

    graph_id = row['graph_id']
    node_info = node_df[(node_df['graph_id'] == graph_id) & (node_df['node_id'].isin(nodes))]
    all_features = pd.merge(right=node_embeddings, right_index=True, left=node_info, left_on='node_id')
    all_features = all_features.drop(columns=['graph_id', 'node_id'])
    as_np = all_features.to_numpy()
    return th.from_numpy(as_np).type(th.float32)

def y_transform(label, label_count):
    """
    label:  label of the datapoint
    label_count:    number of different labels in dataset
    """
    label_tensor = th.zeros((label_count))
    label_tensor[label-1] = 1
    return label-1