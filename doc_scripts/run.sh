#!/bin/bash
cd $(dirname $0)

MOUNT_DIR=/mnt/storage/datasets/HCP/
GPU=ALL
SHARA=""
#./run.sh -g 3 -s ~/vaegan/shara
# python3 main.py --model_name "cAAE" --z_dim "128"
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -m|--mount_dir)
    MOUNT_DIR="$2"
    shift # past argument
    shift # past value
    ;;
    -g|--set_gpu)
    GPU="$2"
    shift # past argument
    shift # past value
    ;;
    -s|--shara)
    SHARA="$2"
    shift # past argument
    shift # past value
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

echo "MOUNT_GIR  = ${MOUNT_DIR}"
echo "GPU        = ${GPU}"

if [ -n "$SHARA" ]
then
	echo "SHARA      = ${SHARA}"
fi

if [ "$GPU" == ALL ]
then
	GPU=all
else
	GPU="device=$GPU"
fi

echo Building Docker container...
docker build \
        -f ./get_dataset/Dockerfile \
        -t caae_image \
        .

#docker build \
#	-f Dockerfile \
#	-t caae_image \
#	$(for i in `cat var.env`; do out+="--build-arg $i " ; done; echo $out;out="") \
#	.

echo running Adversarial Autoencoder example
if [ -n "$SHARA" ]
then
	docker run \
        	--name caae_doc \
        	--gpus $GPU \
        	-it \
        	-v $MOUNT_DIR:/root/HCP/ \
        	-v $SHARA:/root/shara/ \
		--env-file ./get_dataset/var.env \
        	caae_image

else
        docker run \
	        --name caae_doc \
        	-it \
        	-v $MOUNT_DIR:/root/HCP/ \
        	caae_image
fi

cp -r ../model ./train/

echo Building Docker container...
docker build \
        -f ./train/Dockerfile \
        -t caae_image_nn \
        .

rm -rf ./train/model

echo running Adversarial Autoencoder example
if [ -n "$SHARA" ]
then
        docker run \
                --name caae_doc_nn \
                --gpus $GPU \
                -it \
                -v $MOUNT_DIR:/root/HCP/:ro \
                -v $SHARA:/root/shara/ \
                caae_image_nn
else
        docker run \
                --name caae_doc_nn \
                --gpus $GPU \
                -it \
                -v $MOUNT_DIR:/root/HCP/:ro \
		caae_image_nn
fi
#                --env-file var.env \
#                caae_image_nn
#fi


