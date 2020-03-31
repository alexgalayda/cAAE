#!/bin/bash
cd $(dirname $0)

if [ -z "$1" ]
then
    CONFIG_FILE=config/example.env
else
    CONFIG_FILE=config/$1.env
fi

echo We will use $CONFIG_FILE
source $CONFIG_FILE

if $TEST_FLG;
then
    echo -e "Test mode: \033[32m[ON]\033[0m"
else
    echo -e "Test mode: \033[31m[OFF]\033[0m"
fi


echo "GPU = $GPU"
if [ "$GPU" != all ]
then
	GPU="\"device=$GPU\""
fi

download () {
    if $TEST_FLG;
    then
        TEST=" bash"
        SHARA=$(mount $SHARA $SHARA_CONT false)
    else
        TEST=" "
        SHARA=" "
    fi

    echo Building Docker container...
    docker_build="docker build
    -f doc_scripts/get_dataset/Dockerfile
    -t ${NAME}_image
    --build-arg GET_DATASET=${GET_DATASET}
    ."
    echo $docker_build
    $docker_build

    mkdir $HCP
    echo running Adversarial Autoencoder example...
    docker_run="docker run
    --name ${NAME}_cont
    --gpus $GPU
    --shm-size=1g
    -it
    $(mount $HCP $HCP_CONT)
    ${SHARA}
    --rm
    ${NAME}_image
    ${TEST}
    "
    echo $docker_run
    $docker_run
}

#train () {
#}

main () {
    download
#    train
}

#def port(container_port=8888, server_port=6969):
#    return f' -p {server_port}:{container_port} ' if server_port else ' '

mount()
{
    #1 -- on cont, 2 -- on host, 3 -- read only
    # если ничего не ввести, то все равно будет read only
    # на явно ввести false, защита от стрельбы по коленям
    local mnt=" --mount type=bind,source=${1},destination=${2}"
    if $3; then mnt="${mnt},readonly "; fi
    echo "${mnt}"
}

main



#MOUNT_DIR=/mnt/storage/datasets/HCP/
#GPU=ALL
#SHARA=""
##./run.sh -g 3 -s ~/vaegan/shara
## python3 main.py --model_name "cAAE" --z_dim "128"
## cp -r ~/vaegan/shara/model ../model
#while [[ $# -gt 0 ]]
#do
#key="$1"
#
#case $key in
#    -m|--mount_dir)
#    MOUNT_DIR="$2"
#    shift # past argument
#    shift # past value
#    ;;
#    -g|--set_gpu)
#    GPU="$2"
#    shift # past argument
#    shift # past value
#    ;;
#    -s|--shara)
#    SHARA="$2"
#    shift # past argument
#    shift # past value
#    ;;
#    *)    # unknown option
#    POSITIONAL+=("$1") # save it in an array for later
#    shift # past argument
#    ;;
#esac
#done
#set -- "${POSITIONAL[@]}" # restore positional parameters
#
#echo "MOUNT_GIR  = ${MOUNT_DIR}"
#echo "GPU        = ${GPU}"
#
#if [ -n "$SHARA" ]
#then
#	echo "SHARA      = ${SHARA}"
#fi
#
#if [ "$GPU" == ALL ]
#then
#	GPU=all
#else
#	GPU="device=$GPU"
#fi
#
#echo Building Docker container...
#docker build \
#        -f ./get_dataset/Dockerfile \
#        -t caae_image \
#        .
#
##docker build \
##	-f Dockerfile \
##	-t caae_image \
##	$(for i in `cat var.env`; do out+="--build-arg $i " ; done; echo $out;out="") \
##	.
#
#echo running Adversarial Autoencoder example
#if [ -n "$SHARA" ]
#then
#	docker run \
#        	--name caae_doc \
#        	--gpus $GPU \
#        	-it \
#        	-v $MOUNT_DIR:/root/HCP/ \
#        	-v $SHARA:/root/shara/ \
#		--env-file ./get_dataset/var.env \
#        	caae_image
#
#else
#        docker run \
#	        --name caae_doc \
#        	-it \
#        	-v $MOUNT_DIR:/root/HCP/ \
#        	caae_image
#fi
#
#cp -r ../model ./train/
#
#echo Building Docker container...
#docker build \
#        -f ./train/Dockerfile \
#        -t caae_image_nn \
#        .
#
#rm -rf ./train/model
#
#echo running Adversarial Autoencoder example
#if [ -n "$SHARA" ]
#then
#        docker run \
#                --name caae_doc_nn \
#                --gpus $GPU \
#                -it \
#                -v $MOUNT_DIR:/root/HCP/:ro \
#                -v $SHARA:/root/shara/ \
#                caae_image_nn
#else
#        docker run \
#                --name caae_doc_nn \
#                --gpus $GPU \
#                -it \
#                -v $MOUNT_DIR:/root/HCP/:ro \
#		caae_image_nn
#fi
##                --env-file var.env \
##                caae_image_nn
##fi
#
#
