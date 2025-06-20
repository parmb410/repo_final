def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            x, y = batch[:2]  # ignore domain label
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0
