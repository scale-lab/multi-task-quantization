# Multi-Task Learning Quantization: Evaluating Impact of Task-Specific Quantization on Overall Performance

## Introduction

This is a forked directory of the official implementation of the paper: **E-MTL: Efficient Multi-task Learning Architecture using Hybrid Transformer and ConvNet blocks**. This repository explores the effect of quantization on different modules of an Efficient Multi-Task Learning Model.

## How to Run

If you'd like to run the standalone E-MTL model, checkout [EMTL Repo](https://github.com/scale-lab/E-MTL/tree/main).

An example run of the project:

```
python -m torch.distributed.launch --nproc_per_node 1 --master_port $1 main.py --cfg configs/swin/swin_tiny_patch4_window7_224.yaml --pascal ../../data/shared/AdaMTL/data/PASCAL_MT --tasks $5 --batch-size 96 --ckpt-freq=100 --epoch=400 --eval-freq 100 --resume-backbone pretrained/swin_tiny_patch4_window7_224.pth --name $2/ --wieb $3 --widbpt $4 --output ../../data/shared/QEMTL/
```
Relevant Arguments: 

| Argument   | Example   | Description   |
|------------|------------|------------|
| --nproc_per_node | 1 | Row 1 Col3 |
| --master_port | 11111 | Row 2 Col3 |
| --cfg | configs/swin/swin_tiny_patch4_window7_224.yaml | swin backend configuration |
| --tasks | semseg,normals,sal,human_parts | tasks performed by the MTL |
| --resume-backbone | pretrained/swin_tiny_patch4_window7_224.pth | loading init weights |
| --wieb | 6-8 | weights, inputs encoder bits for quantization |
| --widbpt | 8-8,8-8,8-8,8-8 | decoder-specific in-order weights, inputs quantized bits |


  
## Authorship
Current Authors of the project:
- [Mahdi Boulila](https://github.com/MahdiBoulila)
- [Marina Neseem](https://github.com/marina-neseem)

## Citation
If you find E-MTL helpful in your research, please cite our paper:
```
@inproceedings{boulila2024qmtl,
  title={E-MTL: Efficient Multi-task Learning Architecture using Hybrid Transformer and ConvNet blocks},
  author={Boulila, Mahdi and Neseem, Marina and Reda, Sherief},
  booktitle={},
  pages={},
  year={2024}
}
```

## License
MIT License. See [LICENSE](LICENSE) file
