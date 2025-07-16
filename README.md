# Pre-training Language Models to Express Uncertainty

## Setup
The repository was developed and tested with Python version 3.10.9. 

## Training
Training is done using the script `idk_bert.py` and the configuration found in `config.yaml`.
An example training execution of the model `google/multiberts-seed_0-step_1900k` using the config `config.yaml` looks as follows:
```shell
python idk_bert.py --config ./config.yaml train --model google/multiberts-seed_0-step_1900k
```
See `config.yaml` and `python idk_bert.py train -h` for more options and information.

## Evaluating
Evaluation is done using the script `idk_bert.py` and the configuration found in `config.yaml`.
An example evaluation execution of the model `models/idk_bert` on the `LAMA-SQuAD` dataset looks as follows:
```shell
python idk_bert.py eval models/idk-bert LAMA-squad
```
See `config.yaml` and `python idk_bert.py eval -h` for more options and information.

## Results
The basic results can be acquired by evaluating as described above. 
For a deeper comparison between models (and in order to generate the figures found in the paper) execute:
```shell
python compare_models.py
```
The script's results can be found under `comparison_new_tp`.
