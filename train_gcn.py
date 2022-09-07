import  torch
from torch.utils.tensorboard import SummaryWriter
import wandb
import numpy as np

np.random.seed(seed=5827)
writer = SummaryWriter()

def prediction(test_loader,net,device):
    model=net
    all_features = torch.tensor([]).to(device)
    all_predictions = torch.tensor([]).to(device)
    all_target = torch.tensor([]).to(device, dtype=torch.long)

    model.eval()
    device = 'cuda'

    correct = 0
    for data in test_loader:  # Iterate in batches over the training/test dataset.
        data = data.to(device)
        out, features = model(data.x, data.edge_index, data.edge_attr,data.batch, data.ptr)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        all_features = torch.cat((all_features,features.to(device)))
        all_predictions = torch.cat((all_predictions, pred.to(device)), dim=0)
        all_target = torch.cat((all_target, data.y.to(device)), dim=0)

    return correct / len(test_loader.dataset), all_predictions, all_target, all_features  # Derive ratio of correct predictions.


def train(train_loader,test_loader, net , num_epochs, criterion, optimizer, device):
    model=net
    wandb.watch(model, criterion, log = 'all', log_freq=10)
    for epoch in range(num_epochs):
        model.train()

        for data in train_loader:
            data=data.to(device)
            # forward propagation
            out, features = model(data.x, data.edge_index, data.edge_attr,data.batch, data.ptr)
            # calculate loss
            loss = criterion(out, data.y)
            # back propagation
            loss.backward()
            # weight update
            optimizer.step()
            # Clear gradients
            optimizer.zero_grad()


        train_acc, train_pred, train_label, train_features = prediction(train_loader, model)
        pred_acc, val_predictions, val_label, test_features = prediction(test_loader, model)

        wandb.log({"Loss": loss, "train acc": train_acc, "test acc": pred_acc})
        writer.add_scalar('Train acc', train_acc, epoch)
        writer.add_scalar('Vall acc', pred_acc, epoch)
        print("Epoch %03d, Train: %.4f, Val:  %.4f " % (epoch + 1, train_acc,  pred_acc))

    writer.close()
    return model, pred_acc, val_predictions, val_label, train_features, test_features

