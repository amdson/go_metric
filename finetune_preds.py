import numpy as np
import json, torch
from torch.utils.data import DataLoader
from go_metric.data_utils import *
from scipy.sparse import csr_matrix, csc_matrix, dok_matrix, vstack, hstack

train_path = "/home/andrew/go_metric/data/go_bench"

train_dataset = BertSeqDataset.from_pickle(f"{train_path}/train.pkl")
val_dataset = BertSeqDataset.from_pickle(f"{train_path}/val.pkl")
test_dataset = BertSeqDataset.from_pickle(f"{train_path}/test.pkl")

collate_seqs = get_bert_seq_collator(max_length=1024, add_special_tokens=True)
dataloader_params = {"shuffle": True, "batch_size": 12, "collate_fn":collate_seqs}
val_dataloader_params = {"shuffle": False, "batch_size": 64, "collate_fn":collate_seqs}

train_loader = DataLoader(train_dataset, **dataloader_params, num_workers=6)
val_loader = DataLoader(val_dataset, **val_dataloader_params)
test_loader = DataLoader(test_dataset, **val_dataloader_params)

from go_metric.models.bert_emb import ProtBertBFDClassifier
import pickle 
with open("checkpoints/bert_emb_hparams.pkl", "rb") as f:
    hparams = pickle.load(f)
    hparams.num_classes = 865
model = ProtBertBFDClassifier.load_from_checkpoint("checkpoints/bert_emb.ckpt", hparams=hparams)
model.eval()
device = torch.device('cuda:2')
model.to(device)

def get_sparse_probs_bert(model, dataloader, threshold=0.02):
    prot_ids = []
    probs_list = []
    with torch.no_grad():
        for i, d in enumerate(dataloader):
            prot_id_l = d["prot_id"]
            inputs, mask, y = d['seq'].to(device), d['mask'].to(device), d['labels'].to(device)
            prot_ids.extend(prot_id_l)
            m_probs = model.forward(inputs, None, mask)
            torch.sigmoid(m_probs, out=m_probs)
            m_probs = m_probs.cpu().numpy()
            m_probs = np.where(m_probs > threshold, m_probs, 0) #Threshold unlikely predictions to keep output sparse. 
            new_probs = csr_matrix(m_probs, dtype=np.float32)
            probs_list.append(new_probs)
            if(i % 10 == 0):
                print(100 * i / len(dataloader))
    probs = vstack(probs_list)
    return prot_ids, probs

test_ids, test_probs = get_sparse_probs_bert(model, test_loader)
# val_ids, val_probs = get_sparse_probs_bert(model, val_loader)
with open("paper_result_predictions/bert_finetune.pkl", "wb") as f:
    pickle.dump({"prot_ids": test_ids, "probs": test_probs}, f)