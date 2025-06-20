import random
import numpy as np
import torch
import os

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_algorithm_class(name):
    """Import and return the Algorithm class by name."""
    # This assumes diversify/algorithm/ contains a file named after each algorithm, e.g., diversify.py
    import importlib
    module = importlib.import_module(f'algorithm.{name}')
    # The class should be named with the first letter capitalized, e.g., Diversify for diversify.py
    cls_name = name.capitalize()
    return getattr(module, cls_name)

def evaluate(model, data_loader, device):
    """Evaluate classification accuracy."""
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for batch in data_loader:
            x = batch[0].to(device)
            y = batch[1].to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    acc = correct / total if total else 0.0
    return acc

def print_args(args, print_list):
    """Nicely print command line arguments (all, or subset in print_list)."""
    s = "==========================================\n"
    for arg, content in vars(args).items():
        if not print_list or arg in print_list:
            s += f"{arg}: {content}\n"
    print(s)
    return s
