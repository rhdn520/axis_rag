import os
os.environ["HF_HOME"] = "/home/seungwoochoi/data/huggingface/cache"
from tqdm import tqdm
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from FlagEmbedding import BGEM3FlagModel
import numpy as np
import wandb
import pandas as pd
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class queryDataset(Dataset):
    def __init__(self, queries):
        super().__init__()
        self.queries = queries

    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, index):

        return self.queries[index]

embedding_model = BGEM3FlagModel('BAAI/bge-m3')

#Load Query Dataset
dataset = load_dataset("microsoft/ms_marco", "v1.1", split ="train", streaming=True)
queries = [x['query'] for x in dataset]  # type: ignore

query_len = 20
query_embeddings = embedding_model.encode(queries)['dense_vecs'][:query_len+1]
query_embedding = torch.from_numpy(query_embeddings)
query_embedding = query_embedding.to("cuda", dtype=torch.float32)

passage_len = 50
passage_embedding = np.load("/home/seungwoochoi/data/axis_rag/data/ms_marco_embedding.npy")
passage_embedding = torch.from_numpy(passage_embedding)[:passage_len+1]
passage_embedding = passage_embedding.to("cuda")
passage_embedding = passage_embedding.to(torch.float32)
# print(passage_embedding)

#Full Embedding으로 계산했을 때 문서 유사도 베스트 순 정렬
full_sim = torch.matmul(passage_embedding, query_embedding.T)
# print(full_sim)

def spearman_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    # Get ranks
    def rankdata(t):
        _, indices = torch.sort(t)
        ranks = torch.zeros_like(t, dtype=torch.float32)
        ranks = ranks.to(device)
        # print(ranks)
        ranks[indices] = torch.arange(1, len(t)+1).to(device, dtype=torch.float32)
        return ranks

    x_rank = rankdata(x)
    y_rank = rankdata(y)
    # print(x_rank)
    # print(y_rank)
    # Compute Pearson correlation on the ranks
    x_mean = x_rank.mean()
    y_mean = y_rank.mean()

    cov = ((x_rank - x_mean) * (y_rank - y_mean)).mean()
    std_x = x_rank.std(unbiased=False)
    std_y = y_rank.std(unbiased=False)
    # print(cov / (std_x * std_y))
    return cov / (std_x * std_y) #type:ignore

passage_embedding = passage_embedding.to(device)
full_sim = full_sim.to(device)

list_sorted_axes = []
list_sorted_corr = []

for i, q_em in tqdm(enumerate(query_embeddings)):
    q_em_tensor = torch.from_numpy(q_em).float().to(device)
    sorted_axes = []
    partial_corr = []
    axes = list(range(1024))

    while len(axes) > 0:
        # print(len(axes))
        tmp_axes = sorted_axes[:]
        
        # [B] Vectorised candidate axis evaluation
        candidate_axes = torch.tensor([tmp_axes + [a] for a in axes], device=device)  # shape [num_axes, len_axes] #type:ignore
        # print(candidate_axes)
        
        # Create batch of passage projections: [num_axes, num_passages, len_axes]
        passage_batch = passage_embedding[:, candidate_axes.T].permute(2, 0, 1)  # [num_axes, N, len_axes]
        # print(passage_batch)

        # Create batch of query projections: [num_axes, len_axes]
        q_batch = q_em_tensor[candidate_axes]  # [num_axes, len_axes]
        # print(q_batch)

        # Compute scores: [num_axes, N]
        scores = torch.einsum('ank,ak->an', passage_batch, q_batch)
        # print(scores)

        # Compute Spearman for each axis candidate
        target = full_sim[:, i]
        # print(target)

        spearman_scores = torch.stack([
            spearman_corr(scores[j], target) for j in range(len(axes))
        ]) #type:ignore

        # Pick best axis
        # print(spearman_scores)
        best_idx = torch.argmax(spearman_scores).item()
        best_axis = axes[best_idx] #type:ignore
        best_score = torch.max(spearman_scores).item()
        # print(best_score)
        partial_corr.append(best_score)

        sorted_axes.append(best_axis)
        axes.remove(best_axis)

    list_sorted_axes.append(sorted_axes)
    list_sorted_corr.append(partial_corr) 
    



# list_sorted_axes is a list of lists, each inner list is axis order for one query
axis_df = pd.DataFrame(list_sorted_axes)
corr_df = pd.DataFrame(list_sorted_corr)
# Save to CSV
axis_df.to_csv("/home/seungwoochoi/data/axis_rag/sorted_axes.csv", index=False, header=False)
corr_df.to_csv("/home/seungwoochoi/data/axis_rag/corr.csv", index=False, header=False)