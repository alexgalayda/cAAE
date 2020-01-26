#!/bin/bash
cd $(dirname $0)
if [ -n "$1" ]
then
	MOUNT_DIR=$1
else
	MOUNT_DIR=/mnt/storage/datasets/HCP/
fi

cp ../requirements.txt .
cp ../credentials .
cp ../copy_dataset.py .
echo Building Docker container...
docker build \
        -f Dockerfile \
        -t caae_image \
        .

#docker build \
#	-f Dockerfile \
#	-t caae_image \
#	$(for i in `cat var.env`; do out+="--build-arg $i " ; done; echo $out;out="") \
#	.

echo running Adversarial Autoencoder example
if [ -n "$2" ]
then
	docker run \
        	--name caae_doc \
        	--gpus '"device=0,1"' \
        	-it \
        	-v $MOUNT_DIR:/root/HCP/ \
        	-v $2:/root/shara/ \
		--env-file var.env \
        	caae_image

else
        docker run \
	        --name caae_doc \
        	--gpus '"device=0,1"' \
        	-it \
        	-v $MOUNT_DIR:/root/HCP/ \
        	--env-file var.env \
        	caae_image
fi
