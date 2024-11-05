import numpy as np
import wandb
from scipy.io import loadmat
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from scipy import sparse
from GCN import Net
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.utils.convert import from_scipy_sparse_matrix
import matplotlib.pyplot as plt
import os
from t_sne import tsne, plot_dendrogram
from sklearn.cluster import AgglomerativeClustering
import random
import time
import torch
import argparse


def process_Data(x_data, edge_index, edge_attr ,y_data):
    data_list = []

    for i in range (np.size(x_data,0)):
        x= np.reshape(x_data[i,:,:],(np.size(x_data,1),np.size(x_data,2)))
        label = y_data[i].astype(np.ndarray)
        data=Data(x=x, edge_index=edge_index.contiguous(),edge_attr=edge_attr,y=label)
        data_list.append(data)
    return data_list


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Learning-Spatiotemporal-Graph-Representations-for-Visual-Perception-using-EEG-Signals')
    parser.add_argument("--seed", default=1657, help="Seed number")
    parser.add_argument("--folds", default=10, help="folds")
    parser.add_argument("--batch_size", default=64, help="Batch_size")
    parser.add_argument("--lr", default=0.0001, help="learning rate")
    parser.add_argument("--num_epoch", default=100, help="Number of epochs")
    parser.add_argument("--model_path", default='models/', help="Model saving path")
    parser.add_argument("--device", default='cuda', help="device")
    parser.add_argument("--dataset", default='MPI', help="device")

    args = parser.parse_args()

    seed = args.seed
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    cv = StratifiedKFold(n_splits=args.folds, shuffle = True, random_state=seed)
    if args.dataset == 'MPI':
        file_data = 'data/MPI'
        sub = os.listdir(file_path)
        # sub = ['Sub06','Sub07','Sub08','Sub10','Sub11','Sub12','Sub13','Sub15', 'Sub16','Sub17','Sub18','Sub19','Sub20','Sub21','Sub22','Sub23','Sub24','Sub25','Sub26','Sub27'  ]
    else : 
        file_data = 'data/SU'
        sub = os.listdir(file_path)
        #sub = ['S1','S2','S3','S4','S5','S6','S7','S8','S9', 'S10']

    # Preparing dataset
    for id in sub:
        wandb.login()
        wandb.init(project=f'gcnn_{id}')

        if args.dataset == 'MPI':
            # Read data
            root = 'data/MPI'
            file = os.path.join(root,id + '.mat')
            data = loadmat(file)

            X=np.array(data['X'])
            X = X[0:60, :, :]
            X=X.transpose(2, 0, 1)
            Y_1 =np.array(data['Y'])

            # Arrange labels
            for i in range(np.size(Y_1,1)):
                if Y_1[0,i]<5 :
                    label =0
                if Y_1[0,i]>4 :
                    label = 1
                Y_1[0,i]=label

            # Read adjacency matrix
            dist_root = 'posMPI.mat'
            dist_var = loadmat(dist_root)
            d = dist_var['pos']

            # Define network
            net = Net(num_node_feat=501, num_classes=2, time_points = 501, time_kernel = 20 , channels = 64 ,pool_kernel = 5,
                      hidden_channels = 128, features = 744 , device=args.device)

        else:
            # Read data
            root = 'data/SU'
            file = os.path.join(root, id + '.mat')
            X = np.array(data['X_3D'])
            X = X.transpose(2, 0, 1)
            Y_1 = np.array(data['categoryLabels'])
            Y_1 = Y_1 - 1

            # Read adjacency matrix
            dist_root = 'posSU.mat'
            data = loadmat(file)
            dist_var = loadmat(dist_root)
            d = dist_var['xyz']

            # Define network
            net = Net(num_node_feat=32, num_classes=6, time_points=32, time_kernel=5, channels=128, pool_kernel=2,
                      hidden_channels=128, features=256, device=args.device)

        # sparce matrix from adjacency matrix
        A = sparse.csc_matrix(d)
        edge_index, edge_attr = from_scipy_sparse_matrix(A)
        Y_1 = np.reshape(Y_1, -1)

        print(f'Training model for {id}')


        score_DL=[]
        conf_acc=[]
        score_cm=[]

        start=time.time()

        for train, test in cv.split(X, Y_1):
            X_train = X[train]
            X_test = X[test]

            x_train=torch.tensor(X_train)
            x_test= torch.tensor(X_test)

            y_train= Y_1[train]
            y_test= Y_1[test]

            # Preprocess data for input  into gcnn
            train_dataset = process_Data(x_train, edge_index, edge_attr, y_train)
            test_dataset = process_Data(x_test, edge_index, edge_attr, y_test)

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

            optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
            criterion = torch.nn.CrossEntropyLoss()

            #Train network
            model, acc, pred, label, train_features, test_features=train(train_loader, test_loader, net, args.num_epochs, criterion, optimizer)

            # Plot tsne
            tsne(test_features.detach().cpu().numpy(), label.cpu().numpy())

            model_a = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
            model_a = model_a.fit(test_features.detach().cpu().numpy())

            plt.title("Hierarchical Clustering Dendrogram")
            plot_dendrogram(model_a, truncate_mode="level", p=3)
            plt.xlabel("Number of points in node (or index of point if no parenthesis).")
            plt.show()

            # Calculate confusion matrix for each fold
            score_DL.append(acc)
            cm=confusion_matrix(label.cpu(), pred.cpu())
            score_cm.append(cm)

        end = time.time()
        mean_auc_dl = np.mean(score_DL)
        std_auc_dl = np.std(score_DL)
        mean_cm = np.sum(np.array(score_cm), axis=0)

        print(f'Training time: {end-start}, Acc: {mean_auc_dl}')
        print("Confusion matrix")
        print(mean_cm)
