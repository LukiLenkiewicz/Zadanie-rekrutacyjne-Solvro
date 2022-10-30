
def train(num_epochs, X_train, y_train, model, error, optimizer):
    for epoch in range(num_epochs):
        for i, (x, y) in enumerate(zip(X_train, y_train)):

            optimizer.zero_grad()
            y_pred, _ = model(x)
            loss = error(y_pred, y)

            loss.backward()
            optimizer.step()
                    
            if (i+1) % 100 == 0:
                print(f'iter no.: {i+1}, loss = {loss.item():.4f}')