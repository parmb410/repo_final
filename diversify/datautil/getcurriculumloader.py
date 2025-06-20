from torch.utils.data import DataLoader, Subset

def get_curriculum_loader(dataset, domain_labels, curriculum_order, args):
    """
    Returns a DataLoader yielding batches from easy-to-hard domains (progressive exposure).
    - dataset: PyTorch dataset
    - domain_labels: list/array of domain label per sample
    - curriculum_order: list of domain indices, from easiest to hardest
    - args: includes batch_size and any other params
    """
    all_indices = []
    for d in curriculum_order:
        indices = [i for i, dom in enumerate(domain_labels) if dom == d]
        all_indices += indices
    # (Optionally: progressively add harder domains per epoch—advanced curriculum)
    curriculum_subset = Subset(dataset, all_indices)
    loader = DataLoader(curriculum_subset, batch_size=args.batch_size, shuffle=True)
    return loader
