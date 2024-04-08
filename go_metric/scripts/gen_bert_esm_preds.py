import numpy as np
import json, torch
from torch.utils.data import DataLoader
from go_metric.data_utils import *
from scipy.sparse import csr_matrix, csc_matrix, dok_matrix, vstack, hstack
import transformers
from go_metric.models.bert_esm_emb import ESMBERTClassifier

train_path = "/home/andrew/go_metric/data/go_bench"
train_dataset = BertSeqDataset.from_pickle(f"{train_path}/train.pkl")
val_dataset = BertSeqDataset.from_pickle(f"{train_path}/val.pkl")
test_dataset = BertSeqDataset.from_pickle(f"{train_path}/test.pkl")

import pickle 
# with open("/home/andrew/go_metric/checkpoints/esm_emb_hparams.pkl", "rb") as f:
#     hparams = pickle.load(f)
#     hparams.num_classes = 865

model = ESMBERTClassifier.load_from_checkpoint("/home/andrew/go_metric/checkpoints/esm_emb-v1.ckpt")
hparams = model.h
device = torch.device('cuda:1')
model.to(device)
model.eval()

tokenizer = transformers.AutoTokenizer.from_pretrained(hparams.model_name)
collate_seqs = get_custom_seq_collator(tokenizer, max_length=hparams.max_length, add_special_tokens=True)
val_dataloader_params = {"shuffle": False, "batch_size": 24, "collate_fn":collate_seqs}

train_loader = DataLoader(train_dataset, **val_dataloader_params)
val_loader = DataLoader(val_dataset, **val_dataloader_params)
test_loader = DataLoader(test_dataset, **val_dataloader_params)

def get_emb_preds(model, dataloader, threshold=0.02):
    prot_ids = []
    probs_list = []
    embs_list = []
    with torch.no_grad():
        for i, d in enumerate(dataloader):
            prot_id_l = d["prot_id"]
            inputs, mask, y = d['seq'].to(device), d['mask'].to(device), d['labels'].to(device)
            prot_ids.extend(prot_id_l)
            m_probs, m_emb = model.forward_emb(inputs, None, mask)
            torch.sigmoid(m_probs, out=m_probs)
            m_probs = m_probs.cpu().numpy()
            m_emb = m_emb.cpu().numpy()
            m_probs = np.where(m_probs > threshold, m_probs, 0) #Threshold unlikely predictions to keep output sparse. 
            new_probs = csr_matrix(m_probs, dtype=np.float32)
            probs_list.append(new_probs)
            embs_list.append(m_emb)
            if(i % 10 == 0):
                print(100 * i / len(dataloader))
    probs = vstack(probs_list)
    embs = np.concatenate(embs_list, axis=0)
    return prot_ids, probs, embs

train_ids, train_probs, train_embs = get_emb_preds(model, train_loader)
base_path = '/home/andrew/go_metric'
with open(f"{base_path}/paper_result_predictions/esm_train_finetune.pkl", "wb") as f:
    pickle.dump({"prot_ids": train_ids, "probs": train_probs}, f)
emb_dict = {"prot_id": train_ids, "embedding": train_embs}
with open(f"{base_path}/emb/esm_finetune_train_emb.pkl", 'wb') as f:
    pickle.dump(emb_dict, f)

val_ids, val_probs, val_embs = get_emb_preds(model, val_loader)
base_path = '/home/andrew/go_metric'
with open(f"{base_path}/paper_result_predictions/esm_val_finetune.pkl", "wb") as f:
    pickle.dump({"prot_ids": val_ids, "probs": val_probs}, f)
emb_dict = {"prot_id": val_ids, "embedding": val_embs}
with open(f"{base_path}/emb/esm_finetune_val_emb.pkl", 'wb') as f:
    pickle.dump(emb_dict, f)

test_ids, test_probs, test_embs = get_emb_preds(model, test_loader)
base_path = '/home/andrew/go_metric'
with open(f"{base_path}/paper_result_predictions/esm_finetune.pkl", "wb") as f:
    pickle.dump({"prot_ids": test_ids, "probs": test_probs}, f)
emb_dict = {"prot_id": test_ids, "embedding": test_embs}
with open(f"{base_path}/emb/esm_finetune_test_emb.pkl", 'wb') as f:
    pickle.dump(emb_dict, f)