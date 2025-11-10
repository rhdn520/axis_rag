#!/bin/bash
#SBATCH --job-name=axis_rag
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=1-23:59:59
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=1

source /data3/seungwoochoi/.bashrc
source /data3/seungwoochoi/miniconda3/etc/profile.d/conda.sh
conda activate axis_rag

# Grid Training
# for hidden_dimension in 30; do
#     for number_hidden_layer in 2; do
#         for l1_lambda in 0; do
            
#             srun python train.py \
#             --hidden_dim $hidden_dimension \
#             --hidden_layer_number $number_hidden_layer \
#             --l1_lambda $l1_lambda \
#             --learning_rate 0.0003 \
#             --num_epochs 30
#             # --add_sigmoid

#         done    
#     done
# done


# Grid Evaluation
for hidden_dimension in 30; do
    for number_hidden_layer in 2; do
        for l1_lambda in 0; do
            
            srun python eval.py \
            --hidden_dim $hidden_dimension \
            --hidden_layer_number $number_hidden_layer \
            --l1_lambda $l1_lambda \
            --learning_rate 0.0003 \
            # --add_sigmoid

        done    
    done
done





# srun python preprocess.py

# srun python eval.py