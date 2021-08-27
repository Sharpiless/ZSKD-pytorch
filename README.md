# Zero-Shot Knowledge Distillation in Deep Networks Pytorch

|                   |        |         |               |
|-------------------|--------|---------|---------------|
| Model             | Method | Dataset | top1-accuracy |
| LeNet5-LeNet5Half | Paper  | Mnist   | 98.77         |
| LeNet5-LeNet5Half | Ours   | Mnist   | 96.98         |

# Usage

```bash
python main.py --dataset=cifar10 --lr=0.01 --t_train=False --num_sample=24000 --batch_size=100
```