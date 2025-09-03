# 指定使用的 GPU
###
 # @Author: Sicen Liu
 # @Date: 2025-06-02 18:59:35
 # @LastEditTime: 2025-06-18 20:06:40
 # @FilePath: /liusicen/shared_files/MOFE/run_FL_fusion_Disorder723.sh
 # @Description: 
 # 
 # Copyright (c) 2025 by ${liusicen_cs@outlook.com}, All Rights Reserved. 
### 
export CUDA_VISIBLE_DEVICES=0

# 设定循环次数
num_iterations=10

# 遍历每个参数，将当前参数设为1，其余设为0
for ((i=1; i<=num_iterations; i++)); do
    cmd="python main.py --model_name FL_DynamicFusion --pssm 1 --esm2 1 --batch_size 2 "

    echo "Running: $cmd"
    $cmd
done