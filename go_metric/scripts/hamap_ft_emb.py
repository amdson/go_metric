from Bio import AlignIO
import os

hamas_families = []
hamas_sequences = []
for i, prot_fam in enumerate(os.listdir("data/hamap_alignments/")):
    family_seq = list(AlignIO.read(f"data/hamap_alignments/{prot_fam}", "fasta"))[:512]
    hamas_sequences.extend(family_seq)
    hamas_families.extend([prot_fam]*len(family_seq))
    if(i > 500):
        break

# Gen hamap embeddings
import torch
from go_metric.data_utils import *
from torch.utils.data import DataLoader

class BERTHamasData(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, prot_names, prot_fam, sequences):
        self.prot_names = prot_names
        self.sequences = sequences #A list of strings representing proteins
        self.prot_fam = prot_fam
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, index):
        X = self.sequences[index]
        prot_id = self.prot_names[index]
        prot_fam = self.prot_fam[index]
        return {"seq": X, "prot_id": prot_id, "prot_fam": prot_fam}

hamap_dataset = BERTHamasData([seq.id for seq in hamas_sequences], hamas_families,
                              [" ".join(str(seq.seq).replace('.', '').replace('-', '')) for seq in hamas_sequences])

collate_seqs = get_bert_seq_collator(max_length=1024, add_special_tokens=False)
val_dataloader_params = {"shuffle": False, "batch_size": 256, "collate_fn":collate_seqs}
hamap_loader = DataLoader(hamap_dataset, **val_dataloader_params, num_workers=6)

from go_metric.models.bert_emb import ProtBertBFDClassifier
import pickle 
with open("checkpoints/bert_emb_hparams.pkl", "rb") as f:
    hparams = pickle.load(f)
    hparams.num_classes = 865
model = ProtBertBFDClassifier.load_from_checkpoint("checkpoints/bert_emb.ckpt", hparams=hparams)
model.eval()
device = torch.device('cuda:2')
model.to(device)
print("BERT Model Loaded")

def get_finetune_embeddings(model, dataset, device):
    collate_seqs = get_bert_seq_collator(max_length=1024, add_special_tokens=True)
    dataloader = DataLoader(dataset, collate_fn=collate_seqs, batch_size=128, shuffle=False)
    prot_ids, emb_l = [], []
    with torch.no_grad():
        for inputs in dataloader:
            prot_ids.extend(inputs['prot_id'])
            tokenized_sequences = inputs["seq"].to(device)
            attention_mask = inputs["mask"].to(device)

            word_embeddings = model.ProtBertBFD(tokenized_sequences,
                                           attention_mask)[0]
            embedding = model.pool_strategy({"token_embeddings": word_embeddings,
                                      "cls_token_embeddings": word_embeddings[:, 0],
                                      "attention_mask": attention_mask,
                                      }, pool_max=False, pool_mean_sqrt=False)
            emb_l.append(embedding.cpu())
            if(len(prot_ids) % 1024 == 0):
                print(f"{len(prot_ids)*100 / len(dataset)}%")
    embeddings = torch.cat(emb_l, dim=0)
    return prot_ids, embeddings

hamap_ids, hamap_embeddings = get_finetune_embeddings(model, hamap_dataset, device)
emb_dict = {"prot_id": hamap_ids, "embedding": hamap_embeddings}
with open("emb/finetune_hamap_emb.pkl", "wb") as f:
    pickle.dump(emb_dict, f)