# Load Gpu Modules

module load GpuModules
module load pytorch-py37-cuda11.2-gcc8/1.9.1
srun -p gpu --gres=gpu:1 -t 5:59:59 --ntasks=1 --cpus-per-task=32 --mem=16G --pty bash