import os
from dotenv import load_dotenv
load_dotenv()
hf_home = os.getenv("HF_HOME")
if hf_home:
    os.environ["HF_HOME"] = hf_home

import numpy as np
from datasets import load_dataset
from FlagEmbedding import BGEM3FlagModel
marco_data = load_dataset("microsoft/ms_marco", "v1.1", split ="train", streaming=True)

passages = []
for data in marco_data:
    passages += data['passages']['passage_text']


model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True) 

passage_embeddings = model.encode(passages, batch_size=2, max_length=8192, return_dense=True)

np.save("/home/seungwoochoi/data/axis_rag/data/ms_marco_embedding.npy", passage_embeddings['dense_vecs'])