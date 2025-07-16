# Pre-training Language Models to Express Uncertainty - Llama/Mistral

## Setup
Use the Docker container / Dockerfile to reproduce the exact environment. We can use `bash kd-scripts/run-chairserver.sh python train_llama_idk.py ...` to launch a training w/ the correct Dockerfile. The Dokcer image is also prebuilt and hosted on `konstantinjdobler/tv:v3-flash-rms` @ the DockerHub. Use `docker build --platform "linux/amd64"  --tag <tag> .` to re-build it. 

## Training
Training is done using the script `idk_bert.py` and the configuration found in `config.yaml`.
An example training execution looks as follows:
```shell
python train_idk_llama.py ---config_path idk_llama/cfgs/uncertainty-fsdp.yml --micro_batch_size 1 --data_dir <...> --num_devices -1 -n runname --tokenizer /path/to/tokenizer 
```
See `idk_llama/cfgs/uncertainty-fsdp.yml` and `python train_llama_idk.py --help` for more options and information.

## Evaluating
Todo

## Results
Todo

TODO:
- think about config values, how long does training makes sense?
- create idk tokenizer for Llama/Mistral
- rename thinking -> idk everywhere