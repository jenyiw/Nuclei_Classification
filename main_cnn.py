"""
Main function to run CNN
"""


import h5py
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from cnn import ResNet34
from dataset import CustomDataset
from train_test import train_model, run_model

def main(num_epochs,
         pos_hd_name:str,
         neg_hd_name:str,
         train_props:float,
         load_existing_model:bool=False):
    
    """
    Run main function to train and test a CNN.

    Parameters:
    num_features: int, number of node features
    pos_hd_name: str, file name for hdf5 file containing positive samples
    neg_hd_name: str, file name for hdf5 file containing negative samples
    train_props: float, proportion of data to use as training data
    load_existing model: bool, whether to use a pre-trained model

    Returns:
    None
    
    """
    train_num = len()

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

    if not load_existing_model:

        #get datasets
        train_dataset = CustomDataset('/content', pos_hd_name, neg_hd_name, 'train')
        trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)

        val_dataset = CustomDataset('/content', pos_hd_name, neg_hd_name, 'val')
        valloader = DataLoader(val_dataset, batch_size=10, shuffle=False, num_workers=2)

        #train model
        model = ResNet34()
        model.to(device)
        model = train_model(model, trainloader, valloader, device, num_epochs)

    test_dataset = CustomDataset('/content', pos_hd_name, neg_hd_name, 'test')
    testloader = DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=2)

    model = ResNet34()
    model.to(device)
    model.load_state_dict(torch.load('./model_resnet34'))

    model.eval()
    with torch.no_grad():
        loss, label, prediction, _ = run_model(model, testloader, device, test=True, save_attn=False)

    # Calculate metrics
    acc = accuracy_score(label, prediction)
    f1 = f1_score(label, prediction)
    prec = precision_score(label, prediction)
    recall = recall_score(label, prediction)

    # print(label, prediction)
    print(f'Acc: {acc:.2f} '
                            f'| F1: {f1:>5.2f}% '
                            f'| Precision: {prec:>5.2f} '
                            f'| Recall: {recall:>5.2f}% ')

#-------------------
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(20, load_model=True)