"""

Code for training and testing functions
"""
import h5py
import numpy as np

import time
import torch
import torch.nn as nn
import torch.nn.functional as F


def run_model(model,
              dloader,
              device,
              optimizer,
              loss_fn,
              test:bool=False,
              save_attn:bool=False):

    predicted_arr = []
    label_arr = []

    run_loss = 0

    if save_attn:
          with h5py.File('/content/attn_map.hdf5', 'w') as f:
            f.create_dataset('attn_1', data=np.zeros((len(dloader)*10, 1, 62, 62)))
            f.create_dataset('attn_2', data=np.zeros((len(dloader)*10, 1, 30, 30)))

    for i, data in enumerate(dloader):

        inputs, labels = data

        #Run model
        inputs = inputs.float().to(device)
        predicted, attn_map_1, attn_map_2 = model(inputs)

        if save_attn:
          with h5py.File('/content/attn_map.hdf5', 'a') as f:
            f['attn_1'][i*10:i*10+10, ...] = attn_map_1.cpu().detach().numpy()
            f['attn_2'][i*10:i*10+10, ...] = attn_map_2.cpu().detach().numpy()

        # print(predicted.shape)
        predicted = torch.reshape(predicted, (-1,)).float()
        # print(predicted.shape)
        loss = loss_fn(predicted, labels.float().to(device))

        if not test:

            #update optimizer and calculate loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #Records loss
        run_loss += loss.item()/ len(dloader)

        #Records prediction
        predicted_arr.append(predicted.cpu().detach().numpy())
        label_arr.append(labels.detach().numpy())

    predicted_arr = np.concatenate(predicted_arr)
    label_arr = np.concatenate(label_arr).astype(np.uint8)

    #Threshold predictions
    predicted_arr[predicted_arr < 0.5] = 0
    predicted_arr[predicted_arr != 0] = 1
    predicted_arr = predicted_arr.astype(np.uint8)

    if save_attn:
          with h5py.File('/content/attn_map.hdf5', 'a') as f:
            f.attrs['predicted'] = predicted_arr
            f.attrs['label'] = label_arr

    return run_loss, predicted_arr, label_arr, model


def train_model(model,
         trainloader,
         testloader,
        device,
         num_epochs,):

    time_curr = time.time()

    model.to(device)

    #Metrics list: training loss, training accuracy, test loss, test accuracy
    metrics = np.zeros((num_epochs, 4))
    best_loss = 100
    counter = -1
    best_epoch = 0

    #initialize loss function and optimizer
    loss_fn=nn.BCELoss()
    optimizer=torch.torch.optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):

        if counter == 0:
          break

        time_epoch = time.time()

        #Train model
        model.train()
        total_loss, predicted_arr, label_arr, model = run_model(model, 
                                                         trainloader, 
                                                         device,
                                                         optimizer,
                                                         loss_fn)
        
        total_acc = np.sum(predicted_arr == label_arr) / len(label_arr)

        #use test data for validation
        model.eval()
        with torch.no_grad():
            test_loss, predicted_arr, label_arr, _ = run_model(model, testloader, device, test=True)
            test_acc = np.sum(predicted_arr == label_arr) / len(label_arr)

            # print('Test sum:', np.sum(predicted))

        if test_loss < best_loss:
            torch.save(model.state_dict(), './model_resnet34_attn_temp')
            best_loss = test_loss
            counter = 5
            best_epoch = epoch

        else:
          counter -= 1

        torch.save(model.state_dict(), './model_resnet34_attn_temp')

        #Record loss and accuracy
        metrics[epoch, 0] = total_loss
        metrics[epoch, 1] = total_acc
        metrics[epoch, 2] = test_loss
        metrics[epoch, 3] = test_acc

        if(epoch % 1 == 0):
            print(f'Epoch {epoch:>3} | Train Loss: {total_loss:.2f} '
                            f'| Train Acc: {total_acc*100:>5.2f}% '
                            f'| Test Loss: {test_loss:>5.2f} '
                            f'| Test Acc: {test_acc*100:>5.2f}% '
                            f'| Time: {int(time.time() - time_epoch)}')


    np.save('./log_resnet.npy', metrics)
    print('Best epoch:', best_epoch)


    return model