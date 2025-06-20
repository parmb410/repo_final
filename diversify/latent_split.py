import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans
import torch

def get_features(model, dataset, device, batch_size=256):
    # Extract features (embeddings) for all samples in dataset
    model.eval()
    X = []
    with torch.no_grad():
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for batch in loader:
            if isinstance(batch, (tuple, list)):
                x = batch[0]
            else:
                x = batch
            x = x.to(device)
            features = model.get_feature(x)
            X.append(features.cpu().numpy())
    X = np.concatenate(X, axis=0)
    return X

def estimate_optimal_k(dataset, model, device, args, k_min=2, k_max=6):
    # Use features from encoder for clustering
    print("Extracting features for K estimation...")
    X = get_features(model, dataset, device)
    best_k = k_min
    best_score = -1
    scores = {}
    for k in range(k_min, k_max+1):
        kmeans = KMeans(n_clusters=k, random_state=args.seed).fit(X)
        labels = kmeans.labels_
        if len(np.unique(labels)) < 2:
            continue
        sil_score = silhouette_score(X, labels)
        db_score = -davies_bouldin_score(X, labels)  # Negative so higher is better
        score = sil_score + db_score
        scores[k] = score
        if score > best_score:
            best_score = score
            best_k = k
    # Final clustering
    final_kmeans = KMeans(n_clusters=best_k, random_state=args.seed).fit(X)
    domain_labels = final_kmeans.labels_
    print(f"Auto-estimated K={best_k} (scores: {scores})")
    return domain_labels, best_k

def assign_domains(dataset, domain_labels, allow_missing=False):
    # Adds domain_label to each dataset sample
    # Assumes dataset is indexable, and that len(domain_labels) == len(dataset)
    for i, idx in enumerate(range(len(dataset))):
        if hasattr(dataset[idx], 'domain_label'):
            dataset[idx].domain_label = int(domain_labels[i])
        elif isinstance(dataset, list):
            dataset[idx] = list(dataset[idx]) + [int(domain_labels[i])]
        else:
            # Try fallback: attach attribute or store in a separate list
            if hasattr(dataset, 'domain_labels'):
                dataset.domain_labels[idx] = int(domain_labels[i])
            else:
                if not allow_missing:
                    raise Exception("Cannot assign domain labels: unsupported dataset class")
    return dataset
