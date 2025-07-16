#!/bin/bash

# Default values
image="konstantinjdobler/tv:v3-flash-rms"
command="bash"
gpus="none"

# Function to parse the command line arguments
parse_arguments() {
  local in_command=false

  while [[ $# -gt 0 ]]; do
    case "$1" in
      -g)
        shift
        gpus="$1"
        ;;
      -i)
        shift
        image="$1"
        ;;
      *)
        if [ "$in_command" = false ]; then
            command="$1"
        else
            command="${command} $1"

        fi
        in_command=true
        ;;
    esac
    shift
  done
}





# Call the function to parse arguments
parse_arguments "$@"

# Rest of your script
echo "image: $image"
echo "command: $command"
echo "gpus: $gpus"

# Look for WANDB_API_KEY
if [ -z "$WANDB_API_KEY" ]; then
  export WANDB_API_KEY=$(awk '/api.wandb.ai/{getline; getline; print $2}' ~/.netrc)
  if  [ -z "$WANDB_API_KEY" ]; then
    echo "WANDB_API_KEY not found"
  else
    echo "WANDB_API_KEY found in ~/.netrc"
  fi
else
  echo "WANDB_API_KEY found in environment"
fi

# Tested on chairserver w/ 4x A6000 - doesn't bring speedups
# # https://lightning.ai/docs/pytorch/stable/advanced/model_parallel.html#when-using-ddp-on-a-multi-node-cluster-set-nccl-parameters
# export NCCL_NSOCKS_PERTHREAD=4
# export NCCL_SOCKET_NTHREADS=2
#  --env NCCL_NSOCKS_PERTHREAD --env NCCL_SOCKET_NTHREADS \


# NOTE: --ipc=host for full RAM and CPU access or -m 300G --cpus 32 to control access to RAM and cpus
# -p 5678:5678 \
docker run --rm -it  -m 200G --cpus 32 --shm-size=200G \
 -v "$(pwd)":/workspace -v /scratch/:/scratch/ -v /home/kdobler/data/:/home/kdobler/data/ -v /home/kdobler/raw-data/:/home/kdobler/raw-data/ -w /workspace \
 --user $(id -u):$(id -g) \
 --env XDG_CACHE_HOME --env WANDB_DATA_DIR --env WANDB_API_KEY --env NCCL_DEBUG \
 --gpus=\"device=${gpus}\" $image $command
