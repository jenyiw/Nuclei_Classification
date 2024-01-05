"""

Main function to run the GNNs
"""

import h5py
import torch
from torch_geometric.data import Data
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler

def construct_graph(pos_hd_file,
                               neg_hd_file,
                               dataset:str,):

    """
    Construct graph from an array as a Data object and append to list

    Parameters:
    pos_hd_file: str, name of hdf5 file with positive graph data
    neg_hd_file: str, name of hdf5 file with negative graph data
    dataset: str, whether to return 'train', 'val', or 'test' data

    Returns:
    graph_list: List, list of graph Data objects
    
    """
    graph_list = []

    with h5py.File(pos_hd_file, 'r') as g:
        with h5py.File(neg_hd_file, 'r') as h:
            data_arr = np.concatenate([g['data'][...], h['data'][...]])
            label_arr = [1]*len(g['data'][...]) + [0]*len(g['data'][...])
            dataset_list = g.attrs['dataset'] + h.attrs['dataset']
            valid_idx = [x for x in range(len(dataset_list)) if dataset_list[x] == dataset]

    dims = data.shape
    
    #scale data
    temp_d = data.reshape(-1, 16)
    scaler = MinMaxScaler()
    data = scaler.fit_transform(temp_d)
    data = data.reshape(dims)

    for i in valid_idx:
        x = torch.tensor(data[i, ...], dtype=torch.float)

        #get labels
        y = torch.tensor(label_arr[i], dtype=torch.float)

        #get edges
        edge_index = [[], []]
        edges = list(combinations(range(6), 2))
        for n in edges:
            edge_index[0].append(n[0])
            edge_index[1].append(n[1])
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        #construct graph
        graph = Data(x=x, edge_index=edge_index, y=y)
        graph_list.append(graph)

    return graph_list
