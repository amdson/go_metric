from torch_cluster import knn
from scipy.sparse import csr_matrix
import torch

def get_embeddings(model, dataloader, device):
    prot_ids = []
    embed_list = []
    with torch.no_grad():
        for d in dataloader:
            prot_id_l = d["prot_ids"]
            X = d["seq"]
            prot_ids.extend(prot_id_l)
            X = X.to(device)
            embed = model.embedding(X)
            embed_list.append(embed)
    embeddings = torch.cat(embed_list, dim=0)
    return prot_ids, embeddings

def embedding_knn(db_emb, q_emb, db_labels, k=10):
    matches = knn(db_emb, q_emb, k).cpu().numpy() # 2 x N*K
    q_labels = csr_matrix((q_emb.shape[0], db_labels.shape[1]), dtype=float)
    for i in range(q_emb.shape[0]):
        # if(i % 10000 == 0):
        #     print(f"{i/q_emb.shape[0]*100} %")
        db_res = matches[1, i*k:(i+1)*k]
        db_avg = csr_matrix(db_labels[db_res].sum(axis=0) / k)
        q_labels[i] = db_avg
    return q_labels

def embedding_wknn(db_emb, q_emb, db_labels, k=10):
    matches = knn(db_emb, q_emb, k).cpu().numpy() # 2 x N*K
    print("made matches", matches.shape)
    q_labels = csr_matrix((q_emb.shape[0], db_labels.shape[1]), dtype=float)
    for i in range(q_emb.shape[0]):
        if(i % 1000 == 0):
            print(f"{i/q_emb.shape[0]*100} %")
        val_res = matches[0, i*k:(i+1)*k]
        db_res = matches[1, i*k:(i+1)*k]
        dist = torch.square(db_emb[db_res] - q_emb[val_res]).sum(dim=1, keepdim=True)
        dist = torch.exp(-dist).cpu().numpy()
        label_res = db_labels[db_res]
        label_res = label_res.multiply(dist)
        db_avg = csr_matrix(label_res.sum(axis=0) / k)
        q_labels[i] = db_avg
    return q_labels