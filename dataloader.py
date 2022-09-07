from torch_geometric.data import Data
from torch_geometric.data import DataLoader
import numpy as np


def process_Data(x_data, edge_index, edge_attr ,y_data):
    data_list = []
    for i in range (np.size(x_data,0)):
        x= np.reshape(x_data[i,:,:],(np.size(x_data,1),np.size(x_data,2)))
        label = y_data[i].astype(np.ndarray)
        data=Data(x=x, edge_index=edge_index.contiguous(),edge_attr=edge_attr,y=label)
        data_list.append(data)
    return data_list

def data_generator(x_train, y_train, x_test, y_test, edge_index, edge_attr,  batch_size):
    train_dataset = process_Data(x_train, edge_index, edge_attr, y_train)
    test_dataset = process_Data(x_test, edge_index, edge_attr, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
