import os
os.environ["HF_HOME"] = "/home/seungwoochoi/data/huggingface/cache"
import torch
import torch.nn as nn


# class ContrasitiveLoss(nn.Module):
#     def __init__(self, temperature=0.05):
#         super().__init__()
#         self.temperature = temperature

#     def forward(self, sim_scores, labels):
#         # labels가 1인 경우의 sim_score가 가장 값이 커야 함.
        


#         return loss.mean()



class AxisRanker(nn.Module):
    def __init__(self, input_dim=1024, output_dim=1024, hidden_dim=30, hidden_layer_number=2, add_sigmoid=False):
        super().__init__()
        self.sequential = nn.Sequential()

        self.sequential.append(
            nn.Linear(input_dim, hidden_dim)
        )
        self.sequential.append(
            nn.GELU()
        )
        for i in range(hidden_layer_number-1):
            self.sequential.append(
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.sequential.append(
                nn.GELU()
            )
        self.sequential.append(
            nn.Linear(hidden_dim, output_dim)
        )

        if(add_sigmoid):
            self.sequential.append(
                nn.Sigmoid()
            )
    
    def forward(self, input_vector):
        output = self.sequential(input_vector)
        return output

