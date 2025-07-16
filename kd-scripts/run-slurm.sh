#!/bin/bash

# Default values
image="none"
command="bash"
gpus="none"
arch="x86"
memory="32G"
time_limit="24:00:00"
cpus="8"
specific_node=""

HPI_SCRATCH="/hpi/fs00/scratch/konstantin.dobler"
default_image_x86="konstantinjdobler+tv+v3-flash-rms.sqsh"
default_image_ppc="NOT IMPLEMENTED"

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
      -a)
        shift
        arch="$1"
        ;;
      -m)
        shift
        memory="$1"
        ;;
      -t)
        shift
        time_limit="$1"
        ;;
      -c)
        shift
        cpus="$1"
        ;;
      -w)
        shift
        specific_node="$1"
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

if [ "$arch" = "x86" ]; then
  if [ "$image" = "none" ]; then
    slurm_constraint="GPU_SKU:A100&ARCH:X86"
    image="$default_image_x86"
  fi
elif [ "$arch" = "ppc" ]; then
  if [ "$image" = "none" ]; then
    slurm_constraint="GPU_SKU:V100&ARCH:PPC"
    image="$default_image_ppc"
  fi
else
  echo "Unknown architecture: $arch"
  exit 1
fi

# Call the function to parse arguments
parse_arguments "$@"

# Rest of your script
echo "image: $image"
echo "command: $command"
echo "gpus: $gpus"
echo "arch: $arch"
echo "memory: $memory"
echo "time_limit: $time_limit"
echo "cpus: $cpus"
echo "slurm_constraint: $slurm_constraint"

if [ -z  "$specific_node" ]; then
  specific_node=""
else
  echo "specific_node: $specific_node"
  specific_node="-w $specific_node"
fi

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


srun -A demelo -p sorcery -N 1 --pty -C $slurm_constraint \
 --container-image $HPI_SCRATCH/$image --container-workdir $HPI_SCRATCH/llama-transfer-dev \
 --container-mounts $HPI_SCRATCH,/hpi/fs00/home/konstantin.dobler/.config --container-env XDG_CACHE_HOME,WANDB_DATA_DIR,WANDB_API_KEY \
 --gpus $gpus --mem $memory -t $time_limit -c $cpus $specific_node \
 $command

