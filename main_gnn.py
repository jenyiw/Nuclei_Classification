"""

Main function for GNN
"""
import torch
from torch_geometric.data import Batch
from torch_geometric.data import DataLoader
import h5py
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from dataset_gnn import construct_graph
from train_test_gnn import train_model_gnn, run_model_gnn
from gnn import GIN

def main(num_features,
		 pos_hd_name,
		 neg_hd_name,
         num_epochs:int=20,
         train_props:float=0.8,
		 load_existing_model:bool=True,
		 ):
    
    """
    Run main function to train and test a GNN.

    Parameters:
    num_features: int, number of node features
    pos_hd_name: str, file name for hdf5 file containing positive samples
    neg_hd_name: str, file name for hdf5 file containing negative samples
    num_epochs:int, number of epochs to train the model for
    train_props: float, proportion of data to use as training data
    load_existing model: bool, whether to use a pre-trained model

    Returns:
    None
    
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
    if load_existing_model:
		
        #load pre-trained model
        print('-'*60)
        print('Loading existing model...')
        model = GIN(64, num_features)

        model.load_state_dict(torch.load('./model_GNN'))
        model.to(device)
        model.eval()

    else:
		
        print('Constructing train data.')
		
        #get number of samples for each dataset
        with h5py.File(pos_hd_name, 'a') as g:
            total_num = g.attrs['num_samples']
            train_num = int(total_num*train_props)
            test_num = total_num - train_num
            val_num = int(train_num*0.2)
            train_num = train_num - val_num

            g.attrs['dataset'] = ['train', ]*train_num + ['val']*val_num + ['test',]*test_num
    
        #get number of samples for each dataset
        with h5py.File(neg_hd_name, 'a') as g:
            total_num = g.attrs['num_samples']
            train_num = int(total_num*train_props)
            test_num = total_num - train_num
            val_num = int(train_num*0.2)
            train_num = train_num - val_num

            g.attrs['dataset'] = ['train', ]*train_num + ['val']*val_num + ['test',]*test_num    
		
        #get training and validation datasets
        train_list = construct_graph(pos_hd_name, neg_hd_name, dataset='train')
        train_data = Batch.from_data_list(train_list)
        train_data = DataLoader(train_data, batch_size=256, shuffle=True)

        #get validation data
        print('Constructing validation data.')

        val_list = construct_graph(pos_hd_name, neg_hd_name, dataset='val')
        val_data = Batch.from_data_list(val_list)
        val_data = DataLoader(val_list, batch_size=10, shuffle=False)

        #train model
        model = GIN(64, num_features)

        print('Training model...')
        model = train_model_gnn(model,
                        train_data,
                            val_data,
                            device,
                            epochs=num_epochs)

    print('Running test image.')


    label_list = []

    print('Constructing test data.')
    
    #get test data
    test_list = construct_graph(pos_hd_name, neg_hd_name, dataset='val')
    test_data = Batch.from_data_list(test_list)
    test_data = DataLoader(test_list, batch_size=10, shuffle=False)

    for graph in test_list:
        label_list.append(graph.y.numpy())

    #test model
    loss, acc, label, prediction = run_model_gnn(model, test_data, device, test=True)

    #calculate metrics
    acc = accuracy_score(label, prediction)
    f1 = f1_score(label, prediction)
    prec = precision_score(label, prediction)
    recall = recall_score(label, prediction)

    print(f'Testing loss: {loss:.2f}')
    print(f'Acc: {acc:.2f} '
                            f'| F1: {f1:>5.2f}% '
                            f'| Precision: {prec:>5.2f} '
                            f'| Recall: {recall:>5.2f}% ')

	# df = pd.DataFrame({'prediction': predict})
	# df.to_csv(os.path.join(test_save_dir, f'prediction_{graph_type}.csv'),index=False, header=False)


#-------------------
if __name__ == "__main__":

    num_features = 16
    main(num_features,)