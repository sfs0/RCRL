# RCRL
> The full source code is now available in this repository.
> This repository provides environment setup instructions, dataset organization guidelines, and commands for training and evaluation.
---

## Environment

- Python **3.8**
- PyTorch **1.8.0**
- torchvision **0.9.0**
- CUDA **11.3.1**
- cuDNN **11.3**
- opencv-python **4.5.3.56**
- tensorboardx **2.4**

---

## Datasets

- **LEVIR-CD** ：<https://justchenhao.github.io/LEVIR/>
  
- **WHU-CD** ：<http://gpcv.whu.edu.cn/data/building_dataset.html>
  
- **GoogleGZ-CD** ：<https://aistudio.baidu.com/datasetdetail/129387>

After downloading, please organize each dataset under a common root directory with the following structure:

~~~markdown
DATA_ROOT/
  LEVIR/              # or WHU / GoogleGZ
    train/
      A/              # images at Time A
      B/              # images at Time B
      label/          # binary change masks
    val/
      A/
      B/
      label/
    test/
      A/
      B/
      label/
~~~

Here, `A` and `B` store the bitemporal images at Time A and Time B, respectively, and `label` stores the ground-truth change maps.

------

## Running

### Training

Use the following commands to train RCRL on each dataset.

```bash
# LEVIR-CD, 5% labeled data
python train_RCRL.py \
  --epoch 100 \
  --batchsize 16 \
  --gpu_id '0' \
  --data_name 'LEVIR' \
  --train_ratio 0.05 \
  --model_name 'rcrl_LEVIR'

# WHU-CD, 20% labeled data
python train_RCRL.py \
  --epoch 100 \
  --batchsize 16 \
  --gpu_id '0' \
  --data_name 'WHU' \
  --train_ratio 0.2 \
  --model_name 'rcrl_WHU'

# GoogleGZ-CD, 20% labeled data
python train_RCRL.py \
  --epoch 100 \
  --batchsize 16 \
  --gpu_id '0' \
  --data_name 'GoogleGZ' \
  --train_ratio 0.2 \
  --model_name 'rcrl_GoogleGZ'
```

### Testing

After training, the corresponding checkpoints (named by `--model_name`) are loaded by the test script to evaluate the model:

```bash
# LEVIR-CD
python test_RCRL.py \
  --gpu_id '0' \
  --data_name 'LEVIR' \
  --model_name 'RCRL_LEVIR'

# WHU-CD
python test_RCRL.py \
  --gpu_id '0' \
  --data_name 'WHU' \
  --model_name 'RCRL_WHU'

# GoogleGZ-CD
python test_RCRL.py \
  --gpu_id '0' \
  --data_name 'GoogleGZ' \
  --model_name 'RCRL_GoogleGZ'
```

