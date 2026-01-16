#!/bin/bash
#SBATCH --job-name=FTSLLM-sp500
#SBATCH --partition=FinLLM
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:1
#SBATCH --mail-type=end
#SBATCH --mail-user=lzaahy@163.com
#SBATCH --output=/mnt/petrelfs/chengdawei/lustre/wavlet/attn.output
#SBATCH --error=/mnt/petrelfs/chengdawei/lustre/wavlet/attn.err

__conda_setup="$('/mnt/petrelfs/chengdawei/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/mnt/petrelfs/chengdawei/miniconda3/etc/profile.d/conda.sh" ]; then
     . "/mnt/petrelfs/chengdawei/miniconda3/etc/profile.d/conda.sh"
    else
     export PATH="/mnt/petrelfs/chengdawei/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup

conda activate allm4ts
conda list

export LD_LIBRARY_PATH=/mnt/petrelfs/share/gcc/mpfr-4.1.0/lib:/mnt/petrelfs/share/gcc/mpc-1.2.1/lib:/mnt/petrelfs/share/gcc/gcc-10.2.0/lib64:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export PATH=/mnt/petrelfs/share/gcc/gcc-10.2.0/bin:$PATH
export LD_LIBRARY_PATH=/mnt/petrelfs/share/cuda-11.8/lib64:$LD_LIBRARY_PATH
export PATH=/mnt/petrelfs/share/cuda-11.8/bin:$PATH
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

echo $LD_LIBRARY_PATH
echo $PATH

gcc --version

cd /mnt/petrelfs/chengdawei/lustre/wavlet/scripts/long-term-forecasting
                                                                                        
bash /mnt/petrelfs/chengdawei/lustre/wavlet/scripts/long-term-forecasting/sp500.sh
