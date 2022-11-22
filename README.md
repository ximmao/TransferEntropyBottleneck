# Transfer Entropy Bottleneck

### dependency

Code is implemented with `python3.7` and `pytorch 1.8.1`, it also require torchvision for its pretrained network. To run 1d experiment, torchdiffeq is required. Please see the list of library dependencies:
```
python=3.7
torch==1.8.1
torchvision==0.10.0
matplotlib
numpy
scipy
pyyaml
torchdiffeq
```

### dataset 

Dataset __rotatingMNIST__, __needle in the haystack__ and __ multi-component sinusoids__ can be generated via scripts under /dataset/creation_scripts. Upon successful generation, they need to be put in `dataset/data`. You should expect to see the following dir tree

```
dataset
|   __init__.py
|   dataset_base.py
|   bouncing_balls_p.py
|   rotating_digits.py
|   changing_waves_m.py
|
└───data
|   |
|   └───bball_data_noswitch_g1_3
|   |   |
|   |   └───train
|   |   └───valid
|   |   └───test
|   |
|   └───rdigit_data_noswitch
|   |   |
|   |   └───train
|   |   └───valid
|   |   └───test
|   |
|   └───fwaves_data_switch_0p5_multi
|   |   |
|   |   └───train
|   |   └───valid
|   |   └───test
| 
└───creation_scripts
|   create_ccballs_needle.py
|   create_vrdigits.py
|   create_fcwaves_needle.py

```

### training & testing

Use yaml config file `arguments_mnist.yaml`, `arguments_needle.yaml` and `arguments_sinemul.yaml` to run rotating mnist, needle in the haystack and 1d time-series experiments respectively

#### TEB

- training TEB model from scatch
```
python -u train_TE.py --device cuda --seed 0 --config_file TASK_YAML_FILE --log_dir YOUR_LOG_DIR --Y_only False --Y_first False --Y_continuetrain False --y_stopgradient False --TE_epochs TE_TOTAL_EPOCH --TE_checkpoint -1
```
- training TEB model from previous checkpoint from epoch 300 / flag in () is optional, used when the checkpoint is from another folder
```
python -u train_TE.py --device cuda --seed 0 --config_file TASK_YAML_FILE --log_dir YOUR_LOG_DIR --Y_only False --Y_first False --Y_continuetrain False --y_stopgradient False --TE_epochs TE_TOTAL_EPOCH --TE_checkpoint 300 (--TE_checkpoint_foldername OTHER_FOLDER_NAME)
```
- testing TEB model
```
python -u test_TE.py --device cuda --seed 0 --config_file TASK_YAML_FILE --log_dir YOUR_LOG_DIR --TE_checkpoint EPOCH_IDX (--TE_checkpoint_foldername OTHER_FOLDER_NAME)
```

#### TEB from context

- training Y model to learn context from scratch
```
python -u train_TE.py --device cuda  --seed 0 --config_file TASK_YAML_FILE --log_dir YOUR_LOG_DIR --Y_only True --Y_first False --Y_epochs Y_TOTAL_EPOCH --Y_checkpoint -1
```
- training TEB model from learned context from epoch 300 
```
python -u train_TE.py --device cuda --seed 0 --config_file TASK_YAML_FILE --log_dir YOUR_LOG_DIR --Y_only False --Y_first False --Y_continuetrain False --y_stopgradient True --TE_epochs TE_TOTAL_EPOCH --TE_checkpoint -1 --Y_checkpoint 300
```
- testing TEB model from context
```
python -u test_TE.py --device cuda --seed 0 --config_file TASK_YAML_FILE --log_dir YOUR_LOG_DIR --Y_checkpoint Y_EPOCH_IDX --TE_checkpoint TE_EPOCH_IDX
```
