import torch
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

def eval_metrics(predictions, target):
    with torch.no_grad():
        f1 = f1_score(target.data.cpu().numpy(), predictions.cpu().numpy(), average='macro')
        precision = precision_score(target.data.cpu().numpy(), predictions.cpu().numpy(), average='macro')
        recall = recall_score(target.data.cpu().numpy(), predictions.cpu().numpy(), average='macro')
        cm = confusion_matrix(target.data.cpu().numpy(),predictions.cpu().numpy())
    return f1, precision, recall, cm


