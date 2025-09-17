import os

os.environ["HF_HOME"] = "/home/seungwoochoi/data/huggingface/cache"
import torch
import torch.nn as nn
from datasets import load_dataset
import numpy as np
from FlagEmbedding import BGEM3FlagModel
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from datasets import load_from_disk
from axis_ranker import AxisRanker
from argument_parser import train_args_parser
from tqdm import tqdm


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.1):
        super().__init__()
        self.margin = margin

    def forward(self, sim_scores, labels):
        # Find Answer Index
        pos_idx = labels.nonzero(as_tuple=True)[0]

        # Answer Index의 Score에서 나머지 Index의 Score를 뺌
        pos_scores = sim_scores[pos_idx].unsqueeze(1)  # (num_pos, 1)
        neg_scores = sim_scores.unsqueeze(0)  # (1, num_total)
        
        loss = pos_scores - neg_scores
        loss = torch.clamp(self.margin - loss, min=0.0)

        #정답 라벨은 0으로 마진 계산에서 제외
        loss[:, pos_idx] = 0.0
        loss = loss.sum() / len(pos_idx)

        return loss
def train():
    device = "cuda"
    args = train_args_parser()

    # 1. Load preprocessed data
    print("Loading preprocessed data...")
    train_dataset = load_from_disk("data/ms_marco_embedded_train")
    train_dataset.set_format(
        type="torch", columns=["query_embedding", "passage_embeddings", "is_selected"]
    )

    # 2. Create DataLoader (FIX: batch_size=1 and no collate_fn)
    # Each item from the dataloader will now be a single query-passages group
    train_dataloader = DataLoader(
        train_dataset, batch_size=1, shuffle=True
    )

    # 3. Initialize model
    model = AxisRanker(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        hidden_layer_number=args.hidden_layer_number,
        add_sigmoid=args.add_sigmoid,
    ).to(device)

    # 4. Training loop
    print("Starting training!")
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    l1_lambda = args.l1_lambda

    contrastive_loss_fn = ContrastiveLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = LambdaLR(optimizer, lambda epoch: 0.95 ** epoch)

    for epoch in range(num_epochs):
        total_epoch_loss = 0
        for batch in tqdm(train_dataloader):
            # Skip samples with no positive passages
            if batch["is_selected"].sum() == 0:
                continue

            optimizer.zero_grad()

            # FIX: Adjust data handling for batch_size=1
            # Remove the extra dimension added by the dataloader
            query_emb = batch["query_embedding"].squeeze(0).to(device)
            passage_emb = batch["passage_embeddings"].squeeze(0).to(device)
            labels = batch["is_selected"].squeeze(0).to(device).float()

            # Repeat the single query embedding to match the number of passages
            num_passages = passage_emb.shape[0]
            query_emb = query_emb.repeat(num_passages, 1)

            q_dim_weight = model(query_emb)
            weighted_query_emb = query_emb * q_dim_weight
            sim_scores = torch.sum(weighted_query_emb * passage_emb, dim=1)

            # Calculate loss (this is now correct as scores are from one query)
            cts_loss = contrastive_loss_fn(sim_scores, labels)
            l1_penalty = l1_lambda * torch.abs(q_dim_weight).sum()
            loss = cts_loss + l1_penalty

            loss.backward()
            optimizer.step() # Optimizer step is per-batch

            total_epoch_loss += loss.item()

        # FIX: Move scheduler.step() to the end of the epoch
        scheduler.step()

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {total_epoch_loss / len(train_dataloader)}"
        )
        
    # 저장할 경로와 파일명 지정
    MODEL_PATH = f"models/axis_ranker_model_{args.hidden_layer_number}_{args.hidden_dim}_{args.add_sigmoid}_{l1_lambda}_{learning_rate}.pth"

    # 디렉토리가 없으면 생성
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # 모델의 state_dict 저장
    torch.save(model.state_dict(), MODEL_PATH)

    print(f"모델이 {MODEL_PATH}에 저장되었습니다.")
    print("학습 완료!")


if __name__ == "__main__":
    train()
