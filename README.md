# SimSiam-on-paired-datasets
Research project conducted at the Computer Vision Laboratory of Tohoku University

### Pre-training
```
export DATA_ROOT=/path/to/stl10
export CUDA_VISIBLE_DEVICES=gpu
python3 main.py  --optimizer=lars --learning-rate-weights=2.0 --pred-lr-rate=100.  --checkpoint-dir=/path/for/checkpoint $DATA_ROOT
```
### Evaluation: Linear classification
```
export DATA_ROOT=/path/to/stl10
export CUDA_VISIBLE_DEVICES=gpu
python evaluate.py $DATA_ROOT /path/to/checkpoint/resnet18.pth --lr-classifier 0.03
```
