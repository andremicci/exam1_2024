import torch

def train(model, dataloaders, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("lr", 1e-3))
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    for epoch in range(config.get("epochs", 10)):
        for x, y in dataloaders["train"]:
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
    return model, None
