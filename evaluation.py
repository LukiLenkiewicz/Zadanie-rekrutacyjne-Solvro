import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, f1_score


def plot_loss(train_losses, val_losses):
    plt.plot(train_losses, label="train error")
    plt.plot(val_losses, label="validation error")
    plt.title("change in loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()


def evaluate_model(model, X_val, y_val):
    _, y_val_c = torch.max(y_val, 1)
    y_pred = model(X_val)
    _, y_pred = torch.max(y_pred, 1)
    
    y_val_c = y_val_c.numpy()
    y_pred = y_pred.numpy()
    
    acc = accuracy_score(y_val_c, y_pred)
    f1 = f1_score(y_val_c, y_pred, average="macro")    
    return f"accuracy: {acc:.4f}\nf1-score: {f1:.4f}"