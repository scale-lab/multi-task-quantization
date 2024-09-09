# E-MTL: Efficient Multi-task Learning Architecture using Hybrid Transformer and ConvNet blocks

## Introduction

This is the official implementation of the paper: **E-MTL: Efficient Multi-task Learning Architecture using Hybrid Transformer and ConvNet blocks**. 

This repository provides a Python-based implementation of the MTL architecture proposed in the paper. The repository is based upon [Swin-Transformer](https://github.com/microsoft/Swin-Transformer) and uses some modules from [Multi-Task-Learning-PyTorch](https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch).


## How to Run

Running E-MTL code is very similar to Swin's codebase:

1. **Clone the repository**
    ```bash
    git clone https://github.com/scale-lab/E-MTL.git
    cd E-MTL
    ```

2. **Install the prerequisites**
    - Install `PyTorch>=1.12.0` and `torchvision>=0.13.0` with `CUDA>=11.6`
    - Install dependencies: `pip install -r requirements.txt`

3. **Run the code**
        ```
        python main.py --cfg configs/swin/<swin variant>.yaml --pascal <path to pascal database> --tasks semseg,normals,sal,human_parts --batch-size <batch size> --ckpt-freq=20 --epoch=300 --resume-backbone <path to the weights of the chosen Swin variant>
        ```
  
## Authorship
Since the release commit is squashed, the GitHub contributors tab doesn't reflect the authors' contributions. The following authors contributed equally to this codebase:
- [Ahmed Agiza](https://github.com/ahmed-agiza)
- [Marina Neseem](https://github.com/marina-neseem)

## Citation
If you find E-MTL helpful in your research, please cite our paper:
```
@inproceedings{neseem2023emtl,
  title={E-MTL: Efficient Multi-task Learning Architecture using Hybrid Transformer and ConvNet blocks},
  author={Agiza, Ahmed and Neseem, Marina and Reda, Sherief},
  booktitle={},
  pages={},
  year={2023}
}
```

## License
MIT License. See [LICENSE](LICENSE) file
