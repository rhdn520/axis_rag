import os
os.environ["HF_HOME"] = "/home/seungwoochoi/data/huggingface/cache"
import torch
from datasets import load_from_disk
from tqdm import tqdm
import numpy as np
from axis_ranker import AxisRanker
from argument_parser import train_args_parser

print("평가 스크립트 시작...")

# --- 모델 및 데이터 로드 ---
device = "cuda" if torch.cuda.is_available() else "cpu"
args = train_args_parser()

model = AxisRanker(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        hidden_layer_number=args.hidden_layer_number,
        add_sigmoid=args.add_sigmoid,
    ).to(device)

CHECKPOINT_PATH = f"models/axis_ranker_model_{args.hidden_layer_number}_{args.hidden_dim}_{args.add_sigmoid}_{args.l1_lambda}_{args.learning_rate}.pth"
print(f"'{CHECKPOINT_PATH}'에서 모델을 불러옵니다...")
model.load_state_dict(torch.load(CHECKPOINT_PATH))
model.eval()

print("전처리된 데이터 로딩 중...")
validation_dataset = load_from_disk("data/ms_marco_embedded_validation")
validation_dataset.set_format(type='torch', columns=['query_embedding', 'passage_embeddings', 'is_selected'])

train_dataset = load_from_disk("data/ms_marco_embedded_train")
train_dataset.set_format(type='torch', columns=['query_embedding', 'passage_embeddings', 'is_selected'])

train_q_mean = torch.mean(train_dataset['query_embedding'], dim=0).to(device)
train_q_std = torch.std(train_dataset['query_embedding'], dim=0).to(device)

# --- 평가 설정 ---
K_VALUES = [10, 20, 30, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
# 평가에 사용할 지표들 정의 (P@1, P@10 제외)
METRICS = ["MRR", "Success@1", "Success@3"]

# 결과를 저장할 딕셔너리 초기화 (중첩 구조로 변경)
results = {
    "Baseline": {metric: [] for metric in METRICS},
    "Weighted (Full)": {metric: [] for metric in METRICS},
}
for k in K_VALUES:
    results[f"Top-{k} (Sparse)"] = {metric: [] for metric in METRICS}
    results[f"Top-{k} (Weighted Sparse)"] = {metric: [] for metric in METRICS}
    results[f"Top-{k} (Variance)"] = {metric: [] for metric in METRICS}
    results[f"Top-{k} (Variance x Weight)"] = {metric: [] for metric in METRICS}
    results[f"Top-{k} (Random Sparse)"] = {metric: [] for metric in METRICS}

# --- MRR 및 기타 지표 계산을 위한 헬퍼 함수 ---

def get_sorted_labels(scores, labels):
    """점수를 내림차순으로 정렬하고, 해당 순서에 맞는 레이블을 반환합니다."""
    if not isinstance(scores, torch.Tensor):
        scores = torch.tensor(scores, device=labels.device)
    # 점수가 GPU에 있다면 레이블도 동일한 장치로 이동
    if scores.device != labels.device:
        labels = labels.to(scores.device)
        
    sorted_indices = torch.argsort(scores, descending=True)
    return labels[sorted_indices]

def calculate_rr(sorted_labels):
    """정렬된 레이블을 기반으로 Reciprocal Rank를 계산합니다."""
    first_relevant_rank_tensor = torch.where(sorted_labels == 1)[0]
    if len(first_relevant_rank_tensor) > 0:
        rank = first_relevant_rank_tensor[0].item() + 1
        return 1.0 / rank
    return 0.0

# [헬퍼 함수 유지] Precision@k 함수는 코드에 남겨둡니다.
def calculate_precision_at_k(sorted_labels, k):
    """상위 K개 결과의 정확도(Precision@K)를 계산합니다."""
    if len(sorted_labels) < k:
        k = len(sorted_labels)
    if k == 0:
        return 0.0
    
    top_k_labels = sorted_labels[:k]
    return (top_k_labels.sum() / k).item()

def calculate_success_at_k(sorted_labels, k):
    """상위 K개 결과에 정답이 포함되었는지(Success@K)를 계산합니다."""
    if len(sorted_labels) < k:
        k = len(sorted_labels)
    if k == 0:
        return 0.0
        
    return 1.0 if sorted_labels[:k].sum() > 0 else 0.0

def calculate_recall_at_k(sorted_labels, k, total_relevant):
    """상위 K개 결과의 재현율(Recall@K)을 계산합니다."""
    if total_relevant == 0:
        return 1.0 # 정답이 없는 경우, 재현율은 정의에 따라 1 또는 0으로 처리 가능 (여기서는 1로 가정)
        
    if len(sorted_labels) < k:
        k = len(sorted_labels)
    if k == 0:
        return 0.0

    relevant_in_top_k = sorted_labels[:k].sum()
    return (relevant_in_top_k / total_relevant).item()

def visualize_embeddings(query_emb, q_dim_weight, query_emb_normalized, fig_index):
    """쿼리 임베딩과 차원 가중치를 시각화합니다."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    query_emb = query_emb.squeeze().cpu().numpy()
    q_dim_weight = q_dim_weight.squeeze().cpu().numpy()
    query_emb_normalized = query_emb_normalized.squeeze().cpu().numpy()

    plt.figure(figsize=(16, 9))

    plt.subplot(3, 1, 1)
    sns.barplot(x=np.arange(len(query_emb)), y=query_emb, palette="viridis")
    plt.title("Query Embedding Values")
    # plt.xlabel("Dimension")  # x label 제거
    plt.ylabel("Value")
    plt.gca().set_xticklabels([])

    plt.subplot(3, 1, 2)
    sns.barplot(x=np.arange(len(query_emb_normalized)), y=query_emb_normalized, palette="viridis")
    plt.title("Normalized Query Embedding Values")
    # plt.xlabel("Dimension")  # x label 제거
    plt.ylabel("Value")
    plt.gca().set_xticklabels([])

    plt.subplot(3, 1, 3)
    sns.barplot(x=np.arange(len(q_dim_weight)), y=q_dim_weight, palette="viridis")
    plt.title("Dimension Weights from AxisRanker")
    print(np.where(q_dim_weight <= 0.0)[0].shape[0])
    # plt.xlabel("Dimension")  # x label 제거
    plt.ylabel("Weight")
    plt.gca().set_xticklabels([])

    plt.tight_layout()
    plt.savefig(f"figures/query_embedding_and_weights_{fig_index}.png")

# --- 종합 평가 시작 ---
print("="*50)
print("종합 평가를 시작합니다...")

count = 0 

q_dim_weight_list = []
query_emb_normalized_list = []

with torch.no_grad():
    for data in tqdm(validation_dataset, desc="Evaluating"):
        count += 1
        query_emb = data['query_embedding'].unsqueeze(0).to(device)
        passage_embs = data['passage_embeddings'].to(device)
        labels = data['is_selected']
        total_relevant_docs = labels.sum()

        # 각 점수에 대해 모든 지표를 계산하는 함수
        def evaluate_all_metrics(scores, labels, method_name):
            sorted_labels = get_sorted_labels(scores, labels)
            
            results[method_name]["MRR"].append(calculate_rr(sorted_labels))
            # P@k 계산은 제외
            # results[method_name]["P@1"].append(calculate_precision_at_k(sorted_labels, 1))
            # results[method_name]["P@10"].append(calculate_precision_at_k(sorted_labels, 10))
            results[method_name]["Success@1"].append(calculate_success_at_k(sorted_labels, 1))
            results[method_name]["Success@3"].append(calculate_success_at_k(sorted_labels, 3))
            # results[method_name]["Recall@10"].append(calculate_recall_at_k(sorted_labels, 10, total_relevant_docs))

        # 1. Baseline (순수 내적)
        baseline_scores = torch.matmul(query_emb, passage_embs.T).squeeze()
        evaluate_all_metrics(baseline_scores, labels, "Baseline")

        # AxisRanker 모델로부터 가중치 계산 (나머지 방식에 공통 사용)
        q_dim_weight = model(query_emb).squeeze()
        q_dim_weight_list.append(q_dim_weight.cpu().numpy())

        query_emb_normalized = (query_emb - train_q_mean) / train_q_std
        query_emb_normalized_abs = torch.abs(query_emb_normalized)
        query_emb_normalized_list.append(query_emb_normalized.cpu().numpy())

        # 3. Weighted (Full) (가중 내적)
        weighted_query_emb = query_emb.squeeze() * q_dim_weight
        weighted_scores = torch.matmul(weighted_query_emb, passage_embs.T)
        evaluate_all_metrics(weighted_scores, labels, "Weighted (Full)")

        
        # K별 평가 루프
        for k in K_VALUES:
            top_k_vals, top_k_indices = torch.topk(torch.abs(q_dim_weight), k)
            _, var_top_k_indices = torch.topk(query_emb_normalized_abs, k)
            var_top_k_indices = var_top_k_indices.squeeze()
            
            query_emb_topk = query_emb.squeeze()[top_k_indices]
            passage_embs_topk = passage_embs[:, top_k_indices]
            
            # 2. Top-K (Sparse) (선택된 차원으로 단순 내적)
            top_k_scores_sparse = torch.matmul(query_emb_topk, passage_embs_topk.T)
            evaluate_all_metrics(top_k_scores_sparse, labels, f"Top-{k} (Sparse)")

            # 4. Top-K (Weighted Sparse) (선택된 차원에 가중치를 곱하여 내적)
            weighted_query_emb_topk = query_emb_topk * top_k_vals
            top_k_scores_weighted = torch.matmul(weighted_query_emb_topk, passage_embs_topk.T)
            evaluate_all_metrics(top_k_scores_weighted, labels, f"Top-{k} (Weighted Sparse)")

            # 5. Top-K (Variance) (임베딩 평균에서 떨어진 거리 / std의 절댓값)
            # print(query_emb.squeeze()[var_top_k_indices].shape)
            # print(passage_embs[:, var_top_k_indices].shape)
            var_topk = torch.matmul(
                query_emb.squeeze()[var_top_k_indices], 
                passage_embs[:, var_top_k_indices].T
                )
            evaluate_all_metrics(var_topk, labels, f"Top-{k} (Variance)")

            # 6. Top-K (Variance x Weight)
            var_topk_weighted = torch.matmul(
                query_emb.squeeze()[var_top_k_indices] * q_dim_weight[var_top_k_indices], 
                passage_embs[:, var_top_k_indices].T
            )
            evaluate_all_metrics(var_topk_weighted, labels, f"Top-{k} (Variance x Weight)")

            # 7. Random-K (Sparse) - 추가적인 비교를 위해 무작위로 K개 차원 선택
            rand_indices = torch.randperm(q_dim_weight.size(0))[:k]
            query_emb_randk = query_emb.squeeze()[rand_indices]
            passage_embs_randk = passage_embs[:, rand_indices]
            rand_k_scores_sparse = torch.matmul(query_emb_randk, passage_embs_randk.T)
            evaluate_all_metrics(rand_k_scores_sparse, labels, f"Top-{k} (Random Sparse)")

        
        if count <= 10:
            # query_emb, q_dim_weight dimension별 시각화
            visualize_embeddings(query_emb, q_dim_weight, query_emb_normalized, count)


# --- 최종 결과 출력 ---
print("\n" + "="*50)
print("종합 평가 결과")
print("="*50)

for method, metric_dict in results.items():
    if len(next(iter(metric_dict.values()))) > 0:
        # 각 지표의 평균 점수를 문자열로 조합
        result_str = " | ".join([
            f"{metric} = {np.mean(scores):.4f}" 
            for metric, scores in metric_dict.items()
        ])
        print(f"{method:<28}: {result_str}")
    else:
        print(f"{method:<28}: 데이터 없음")
print("="*50)

#save q_dim_weight_list
q_dim_weight_array = np.array(q_dim_weight_list)
np.save(f"results/q_dim_weight_list_{args.hidden_layer_number}_{args.hidden_dim}_{args.add_sigmoid}_{args.l1_lambda}_{args.learning_rate}.npy", q_dim_weight_array)

#save query_emb_normalized_list
query_emb_normalized_array = np.array(query_emb_normalized_list)
np.save(f"results/query_emb_normalized_list_{args.hidden_layer_number}_{args.hidden_dim}_{args.add_sigmoid}_{args.l1_lambda}_{args.learning_rate}.npy", query_emb_normalized_array)