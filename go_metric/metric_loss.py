import torch

def get_all_triplets(sim_score, sim_margin=3.0):
    matches = sim_score > sim_margin
    diffs = sim_score <= sim_margin
    triplets = matches.unsqueeze(2) * diffs.unsqueeze(1)
    return torch.where(triplets)

def multilabel_triplet_loss(embeddings, labels, label_weights=None, sim_margin=1.0, tmargin=1.5):
    if(label_weights is None):
        label_weights = torch.ones((1, labels.shape[1]), device=labels.device)
    emb_dist = torch.cdist(embeddings, embeddings)
    sim_score = labels.multiply(label_weights.to(labels.device)) @ labels.T
    a, p, n = get_all_triplets(sim_score, sim_margin=sim_margin)
    # print(a.shape)
    pos_pairs = emb_dist[a, p]
    neg_pairs = emb_dist[a, n]
    triplet_margin = neg_pairs - pos_pairs
    return torch.mean(torch.relu(tmargin - triplet_margin))