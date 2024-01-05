"""
Main code to run GNNs
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May 12 15:38:12 2023

@author: Jen
"""

import numpy as np
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

#packages for GNN
from torch_geometric.data import Data


def run_model_gnn(model,
			   loader,
			   device,
                loss_fn,
                optimizer,
			   test:bool=False):

    loss = 0
    acc = 0
    predict = []
    label_list = []

    for i, data in enumerate(loader):

            predicted = model(data.x.to(device), data.edge_index.to(device), batch=data.batch.to(device))
            predicted = torch.reshape(predicted, (-1,))

            current_loss = loss_fn(predicted, data.y.float().to(device))
            loss += current_loss

            if not test:
                optimizer.zero_grad()
                current_loss.backward()
                optimizer.step()

            predicted = predicted.detach().cpu().numpy()
            predict.append(predicted)
            label_list.append(data.y.numpy())

    label_list = np.concatenate(label_list)
    predicted_arr = np.concatenate(predict)
    predicted_arr[predicted_arr < 0.5] = 0
    predicted_arr[predicted_arr != 0] = 1
    # current_acc = accuracy(predicted.argmax(dim=1), data.y)
    acc = accuracy_score(predicted_arr, label_list)

    return loss/len(loader), acc, label_list, predicted_arr

def train_model_gnn(model,
				loader,
					val_loader,
					device,
					epochs=20,):

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.00001,
                                 weight_decay=0.01
		 									  )


    metrics_df = pd.DataFrame(columns={'loss':[],
										'acc':[],
										'val_loss':[],
										'val_acc':[]})

    model.to(device)
    for epoch in range(epochs):

        # Train on batches
        model.train()

        total_loss, total_acc, _, predict = run_model_gnn(model, loader, device)

        # Validation
        with torch.no_grad():
            model.eval()
            val_loss, val_acc, _, predict = run_model_gnn(model, val_loader, device, test=True)

        # Print metrics every 10 epochs
        if(epoch % 1 == 0):
	              print(f'Epoch {epoch+1:>3} | Train Loss: {total_loss:.2f} '
	              f'| Train Acc: {total_acc*100:>5.2f}% '
 	              f'| Val Loss: {val_loss:.2f} '
 	              f'| Val Acc: {val_acc*100:.2f}%')

    torch.save(model.state_dict(), './model_GNN')

    return model

