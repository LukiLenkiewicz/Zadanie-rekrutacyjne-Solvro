import torch


def train(X_train, y_train, X_val, y_val, model, error, optimizer, num_epochs=50):
    train_losses = []
    val_losses = []
    _, y_val_c = torch.max(y_val, 1)

    for epoch in range(num_epochs):
        for i, (x, y) in enumerate(zip(X_train, y_train)):
            model.train()
            optimizer.zero_grad()
            y_pred = model(x)
            loss = error(y_pred, y)

            loss.backward()
            optimizer.step()
        
        train_losses.append(loss.item())
        model.eval()
        y_pred = model(X_val)

        _, pred = torch.max(y_pred, 1)
        acc = (y_val_c == pred).sum()/pred.shape[0]
            
        loss = error(y_pred, y_val)
        val_losses.append(loss.item())
        print(f'Epoch no.: {epoch+1}, train loss = {train_losses[epoch]:.4f}, val loss = {val_losses[epoch]:.4f}, accuracy = {acc}')

    return train_losses, val_losses