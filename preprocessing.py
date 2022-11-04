import numpy as np
import torch


def rotate_data(X, thetas=[np.pi/2, np.pi, 3*np.pi/2]):
    X_rotated = []
    X_rotated.append(X)
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


def normalize(X):
    for i in range(len(X)):
        max_vals = np.max(X[i], axis=0)
        min_vals = np.min(X[i], axis=0)
        X[i] = (X[i] - min_vals)/(max_vals - min_vals)
    
    return X


def prepare_data(X_train, X_val, y_train, y_val, batch_size=128, cnn=False):

    X_train = normalize(X_train)
    X_val = normalize(X_val)

    X_train, X_val, y_train, y_val = to_torch(X_train, X_val, y_train, y_val)

    if torch.cuda.is_available():
        X_train = X_train.cuda()
        X_val = X_val.cuda()
        y_train = y_train.cuda()
        y_val = y_val.cuda()

    if cnn:
        X_train = X_train.transpose(1, 2)
        X_val = X_val.transpose(1, 2)

    X_train = X_train.split(batch_size)
    y_train = y_train.split(batch_size)
    
    return X_train, X_val, y_train, y_val
