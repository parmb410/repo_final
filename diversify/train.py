# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

# Import everything as local modules
from utils import set_seed, get_algorithm_class, evaluate, print_args
from datautil.util import get_dataset, get_input_shape, Nmax
from latent_split import estimate_optimal_k, assign_domains
from datautil.getcurriculumloader import get_curriculum_loader

# ---- Curriculum Learning Utilities ----
def compute_domain_difficulty(domain_losses):
    # Sort domains by validation loss (ascending: easy→hard)
    return sorted(domain_losses, key=domain_losses.get)

def pretrain_encoder(model, train_loader, device, pretrain_epochs=2):
    # Optional: Warm-up encoder before K estimation
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(pretrain_epochs):
        for batch in train_loader:
            x = batch[0].to(device)
            y = batch[1].to(device)
            optimizer.zero_grad()
            loss = model.loss(x, y)
            loss.backward()
            optimizer.step()
    print("Encoder pre-training done.")

# ---- Main ----
def main(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(print_args(args, []))

    # ==== 1. Load Dataset ====
    train_set, val_set, test_set = get_dataset(args)
    input_shape = get_input_shape(train_set)
    num_classes = Nmax(args, 0)  # or use actual class count

    # ==== 2. Initialize Model ====
    AlgorithmClass = get_algorithm_class(args.algorithm)
    model = AlgorithmClass(input_shape, num_classes, args).to(device)
    if not hasattr(model, 'optimizer'):
        model.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ==== 3. Optional Encoder Warm-up ====
    if args.warmup:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        pretrain_encoder(model, train_loader, device, pretrain_epochs=2)

    # ==== 4. Automated K Estimation ====
    print("Estimating optimal K and assigning domain labels...")
    # This function must return: domain_labels (array), K (int)
    domain_labels, K = estimate_optimal_k(train_set, model, device, args)

    # Assign these labels into dataset for use in curriculum (and everywhere)
    assign_domains(train_set, domain_labels)
    assign_domains(val_set, domain_labels, allow_missing=True)
    assign_domains(test_set, domain_labels, allow_missing=True)
    print(f"Estimated K = {K}")

    # ==== 5. Curriculum Learning Loader ====
    if args.curriculum:
        print("Preparing curriculum learning loader...")
        # Compute per-domain validation loss (or use another difficulty metric)
        domain_losses = {}
        for k in range(K):
            domain_subset = [i for i, d in enumerate(domain_labels) if d == k]
            if not domain_subset: continue
            # Use custom dataset output structure
            xs = torch.stack([train_set[i][0] for i in domain_subset])
            ys = torch.tensor([train_set[i][1] for i in domain_subset])
            with torch.no_grad():
                pred = model(xs.to(device))
                domain_losses[k] = torch.nn.functional.cross_entropy(pred.cpu(), ys).item()
        curriculum_order = compute_domain_difficulty(domain_losses)
        train_loader = get_curriculum_loader(train_set, domain_labels, curriculum_order, args)
    else:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    # ==== 6. Validation/Test Loaders ====
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # ==== 7. Training Loop ====
    best_val_acc = 0
    for epoch in range(args.max_epoch):
        model.train()
        for batch in train_loader:
            # Unpack according to your dataset structure
            x = batch[0].to(device)
            y = batch[1].to(device)
            model.optimizer.zero_grad()
            # Call model.loss(x, y), if your Algorithm class provides it, otherwise use model(x)
            loss = model.loss(x, y) if hasattr(model, 'loss') else torch.nn.functional.cross_entropy(model(x), y)
            loss.backward()
            model.optimizer.step()
        # ==== 8. Validation ====
        val_acc = evaluate(model, val_loader, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Optionally save best model
        print(f"Epoch {epoch}: Val Acc={val_acc:.4f}")

    # ==== 9. Final Evaluation ====
    test_acc = evaluate(model, test_loader, device)
    print(f"Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DIVERSIFY with Curriculum Learning & Automated K Estimation")
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--dataset', type=str, default='emg')
    parser.add_argument('--task', type=str, default='cross_people')
    parser.add_argument('--algorithm', type=str, default='diversify')
    parser.add_argument('--alpha1', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--lam', type=float, default=0.0)
    parser.add_argument('--local_epoch', type=int, default=3)
    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--warmup', action='store_true', help="Warm-up encoder before K estimation")
    parser.add_argument('--curriculum', action='store_true', help="Enable curriculum learning")
    args = parser.parse_args()
    main(args)
