# Multi-Teacher Distillation and Conflict-Aware Optimization for Parameter-Efficient Multi-Task Learning

## Introduction

This is the official implementation of the paper: [Multi-Teacher Distillation and Conflict-Aware Optimization for Parameter-Efficient Multi-Task Learning]


## How to Run


1. **Clone the repository**
    ```bash
    git clone https://github.com/SKY2717/TDCO
    cd TDCO
    ```

2. **Install the prerequisites**
    - Install `PyTorch>=1.12.0` and `torchvision>=0.13.0` with `CUDA>=11.6`
    - Install dependencies: `pip install -r requirements.txt`
      
3. **Dataset download link**
   We use the same data (PASCAL-Context and NYUD-v2) as ATRC/InvPT. You can download the preprocessed datasets from the following links:

- **PASCALContext.tar.gz**: [Google Drive](https://cs.stanford.edu/~roozbeh/pascal-context/) 
- **NYUDv2.tar.gz**: [Google Drive]([https://drive.google.com/file/d/1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p7/view?usp=sharing](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)) 


5. **Run the code**
    ```python
    python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py --cfg configs/mtlora/tiny_448/<config>.yaml --pascal <path to pascal database> --tasks semseg,normals,sal,human_parts --distill --teacher-paths configs/teachers_pascal.json --batch-size 32 --ckpt-freq=20 --epoch=300    --distill-temp 4.0  --distill-alpha 0.5  --output output/tdco_pascal  --tag tdco_full --resume-backbone <path to the weights of the chosen Swin variant>
    ```
    Swin variants and their weights can be found at the official [Swin Transformer repository](https://github.com/microsoft/Swin-Transformer). 
  
    The outputs will be saved in `output/` folder unless overridden by the argument `--output`.

6. **Eval**

    To evaluate, use `--eval` and `--resume <checkpoint>` as follows:
    ```python
   torchrun --nproc_per_node 1 --master_port 12345 main.py --cfg configs/mtlora/tiny_448/mtlora_tiny_448_r64_scale4_pertask.yaml --pascal <path to pascal database> --tasks semseg,normals,sal,human_parts --batch-size 32 --resume <.pth path> --eval
    ```
