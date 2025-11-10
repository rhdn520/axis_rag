
import os
os.environ["HF_HOME"] = "/home/seungwoochoi/data/huggingface/cache"
import torch
from datasets import load_dataset, Dataset
from FlagEmbedding import BGEM3FlagModel
import numpy as np

def main():
    split = "validation"
    device = "cuda"
    embedding_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, device=device) # FP16 사용 권장
    
    # streaming=False로 전체 데이터를 불러옵니다. (메모리가 부족하면 나눠서 처리)
    dataset = load_dataset("microsoft/ms_marco", "v1.1", split=split)

    def process_data(data):
        embeddings = embedding_model.encode(data['passages']['passage_text'] + [data['query']])["dense_vecs"]
        
        processed_data = {
            'passage_embeddings':embeddings[:-1],
            'query_embedding':embeddings[-1],
            'is_selected':data['passages']['is_selected']
        }

        return processed_data


    # num_proc으로 멀티프로세싱을 활용해 속도를 높일 수 있습니다.
    processed_dataset = dataset.map(
        process_data, 
        batched=False, 
        batch_size=32 # GPU 메모리에 맞는 배치 사이즈 조절
    )
    
    # 전처리된 데이터셋을 디스크에 저장!
    processed_dataset.save_to_disk(f"data/ms_marco_embedded_{split}")
    print("전처리 및 저장 완료!")

if __name__ == "__main__":
    main()