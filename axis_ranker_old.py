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

class AxisRanker(nn.Module):
    def __init__(self, embedding_size = 1024):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.GELU(),
            nn.Linear(embedding_size, embedding_size),
            nn.GELU(),
            nn.Linear(embedding_size, embedding_size)
        )
    
    def forward(self, embeddings): #embeddings 차원 : 1 * 임베딩차원 (1024)
        output = self.sequential(embeddings)
        return output

class queryDataset(Dataset):
    def __init__(self, queries):
        super().__init__()
        self.queries = queries

    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, index):

        return self.queries[index]

def main():

    topk_k = 10
    select_axis_n = 10
    device = "cuda"
    passage_embedding = np.load("data/ms_marco_embedding.npy")
    passage_embedding = torch.tensor(passage_embedding).to(device, dtype=torch.float16)
    print(passage_embedding.shape)

    embedding_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False) 
    
    axis_ranker = AxisRanker()
    axis_ranker = axis_ranker.to(dtype=torch.float16, device=device)

    axis_ranker_path = "checkpoints/axis_ranker_final.pt"
    if(os.path.exists(axis_ranker_path)):
        axis_ranker.load_state_dict(torch.load(axis_ranker_path, weights_only=True))


    optimizer = torch.optim.AdamW(axis_ranker.parameters(), lr=5e-5)

    wandb.init(project="axis_ranker_project", name="axis_ranker_run")
    wandb.watch(axis_ranker, log="all")

    #Load Query Dataset
    dataset = load_dataset("microsoft/ms_marco", "v1.1", split ="train", streaming=True)
    queries = [x['query'] for x in dataset]
    query_dataset = queryDataset(queries)
    query_dataloader = DataLoader(query_dataset, shuffle=False, batch_size=1)

    num_epochs = 10  # You can adjust the number of epochs
    for epoch in range(num_epochs):
        tqdm.write(f"🔁 Starting epoch {epoch + 1}/{num_epochs}")
        for step, data in enumerate(tqdm(query_dataloader, desc="Training")):
            # print(data)
            embeddings = embedding_model.encode(data)['dense_vecs']
            embeddings = embeddings.T
            embeddings = torch.from_numpy(embeddings)
            embeddings = embeddings.to(device, dtype=torch.float16)
            # print(embeddings.shape)
            
            full_query_norm = torch.norm(embeddings, dim=0, keepdim=True)
            full_passage_norm = torch.norm(passage_embedding, dim=1, keepdim=True)
            full_ranking_scores = torch.matmul(passage_embedding, embeddings) / (full_passage_norm * full_query_norm + 1e-8)
            full_topk_values, full_topk_indices = torch.topk(full_ranking_scores, topk_k, dim=0)
            # print(full_topk_indices)

            output = axis_ranker(embeddings.transpose(1,0))
            _ , chosen_axis_indices = torch.topk(output, select_axis_n)
            # print(chosen_axis_indices)
            
            reduced_passage_embeddings = passage_embedding[:, chosen_axis_indices[0]]
            reduced_query_embedding = embeddings[chosen_axis_indices[0], :]
            # print(reduced_passage_embeddings.shape)
            reduced_query_norm = torch.norm(reduced_query_embedding, dim=0, keepdim=True)
            reduced_passage_norm = torch.norm(reduced_passage_embeddings, dim=1, keepdim=True)
            reduced_ranking_scores = torch.matmul(reduced_passage_embeddings, reduced_query_embedding) / (reduced_passage_norm * reduced_query_norm + 1e-8)
            reduced_topk_values, reduced_topk_indices = torch.topk(reduced_ranking_scores, topk_k, dim=0)

            full_sorted_scores, full_sorted_indices = torch.sort(full_ranking_scores.view(-1), descending=True)
            reduced_sorted_scores = reduced_ranking_scores.view(-1)[full_sorted_indices]

            
            margin = 0.1
            alpha = 1.0
            beta = 1.0

            pairwise_loss = torch.tensor(0.0, device=device, dtype=torch.float16, requires_grad=True)
            top_pair_num = 20

            for i in range(top_pair_num):
                for j in range(i+1, top_pair_num):
                    s_full_i = full_sorted_scores[i]
                    s_full_j = full_sorted_scores[j]
                    if s_full_i > s_full_j:
                        s_red_i = reduced_sorted_scores[i]
                        s_red_j = reduced_sorted_scores[j]
                        pair_loss = torch.clamp(- (s_red_i - s_red_j) + margin, min=0)
                        pairwise_loss = pairwise_loss + pair_loss

            num_pairs = top_pair_num * (top_pair_num - 1) / 2
            pairwise_loss = pairwise_loss / num_pairs

            # Similarity regression loss over all documents
            sim_loss = torch.mean((full_sorted_scores - reduced_sorted_scores) ** 2)

            # Total hybrid loss
            loss = alpha * pairwise_loss + beta * sim_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            tqdm.write(f"Step {step}: Pairwise Loss = {pairwise_loss.item():.4f}, Sim Loss = {sim_loss.item():.4f}, Total Loss = {loss.item():.4f}")

            wandb.log({
                "pairwise_loss": pairwise_loss.item(),
                "sim_loss": sim_loss.item(),
                "total_loss": loss.item(),
                "epoch": epoch + 1,
                "step": step
            })

        torch.save(axis_ranker.state_dict(), f"checkpoints/axis_ranker_epoch{epoch+1}.pt")
        tqdm.write(f"💾 Saved model after epoch {epoch + 1}")

    torch.save(axis_ranker.state_dict(), "checkpoints/axis_ranker_final.pt")
    print("🎉 Training complete! Saved final model as axis_ranker_final.pt")

    wandb.finish()

    return

if __name__=="__main__":
    main()
