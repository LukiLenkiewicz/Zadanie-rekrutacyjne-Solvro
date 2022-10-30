import numpy as np
import torch


def rotate_data(X, thetas=[np.pi/2, np.pi, 3*np.pi/2]):
    X_rotated = []
    for theta in thetas:
        R = np.array([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)]
        ])
        X_rotated.append(X@R)

    return X_rotated


def to_torch(X_train, X_val, y_train, y_val):
    return torch.from_numpy(X_train).to(torch.float32), torch.from_numpy(X_val).to(torch.float32), \
           torch.from_numpy(y_train).to(torch.float32), torch.from_numpy(y_val).to(torch.float32)


def prepare_data(X_train, X_val, y_train, y_val, batch_size=128, cnn=False):
    X_train, X_val, y_train, y_val = to_torch(X_train, X_val, y_train, y_val)

    X_train = X_train.split(batch_size)
    y_train = y_train.split(batch_size)

    if cnn:
        X_train = X_train.transpose(1, 2)
        X_val = X_val.transpose(1, 2)
    
    return X_train, X_val, y_train, y_val