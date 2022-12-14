import numpy as np
import torch


def add_rotated_samples(X, y, thetas):
    X_rotated = []
    X_rotated.append(X)
    for theta in thetas:
        ROTATION_MATRIX = np.array([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)]
        ])
        X_rotated.append(X@ROTATION_MATRIX)

    return np.concatenate(X_rotated), np.concatenate([y for _ in range(len(X_rotated))])


def to_torch(X_train, X_val, y_train, y_val):
    return torch.from_numpy(X_train).to(torch.float32), torch.from_numpy(X_val).to(torch.float32), \
           torch.from_numpy(y_train).to(torch.float32), torch.from_numpy(y_val).to(torch.float32)


def normalize(X):
    min_vals = np.min(X, axis=1).reshape(X.shape[0], 1, X.shape[2])
    max_vals = np.max(X, axis=1).reshape(X.shape[0], 1, X.shape[2])
    return (X - min_vals)/(max_vals - min_vals)


def prepare_data(X_train, X_val, y_train, y_val, angles=[np.pi], batch_size=128, cnn=False):

    X_train, y_train = add_rotated_samples(X_train, y_train, angles)

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
