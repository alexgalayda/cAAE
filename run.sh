#!/bin/bash

port(){
    #1 -- server port, 2 -- container port
    if [ -z "$2" ]
    then
        local container_port=8000
    else
        local container_port=$2
    fi
    echo " -p ${1}:${container_port} "
}

mount_dir(){
    #1 -- on cont, 2 -- on host, 3 -- read only
    # если ничего не ввести, то все равно будет read only
    # надо явно ввести false, защита от стрельбы по коленям
    local mnt=" --mount type=bind,source=${1},destination=${2}"
    if $3; then mnt="${mnt},readonly "; fi
    echo "${mnt}"
}

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

if $TEST_FLG;
then
    TEST=" bash"
    SHARA=$(mount_dir $SHARA $SHARA_CONT false)
else
    TEST=" "
    SHARA=" "
fi

download () {
    echo Building Docker container...
    docker_build="docker build
    -f ${GET_DATASET}/Dockerfile
    -t ${NAME}_download_image
    --build-arg GET_DATASET=${GET_DATASET}
    ."
    echo $docker_build
    $docker_build

    mkdir $HCP
    echo running download...
    docker_run="docker run
    --name ${NAME}_download_cont
    --gpus $GPU
    --shm-size=1g
    -it
    $(mount_dir $HCP $HCP_CONT false)
    ${SHARA}
    --rm
    ${NAME}_image
    ${TEST}
    "
    echo $docker_run
    $docker_run
}

train () {
    echo Building Docker container...
    echo $SHARA
    docker_build="docker build
    -f ${TRAIN}/Dockerfile
    -t ${NAME}_train_image
    --build-arg TRAIN=${TRAIN}
    ."
    echo $docker_build
    $docker_build

    mkdir $RESULT
    echo running Adversarial Autoencoder example...
    docker_run="docker run
    --name ${NAME}_train_cont
    --gpus $GPU
    --shm-size=1g
    -it
    $(mount_dir $HCP $HCP_CONT true)
    $(mount_dir $RESULT $RESULT_CONT)
    $(port $PORT)
    ${SHARA}
    --rm
    ${NAME}_train_image
    ${TEST}
    "
    echo $docker_run
    $docker_run
}

main () {
    if [ "$(ls -A $HCP)" ]
        then
        read -p "There is already some kind of dataset in the $HCP folder. Are you sure? " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]
        then
            download
        fi
    fi
    train
}

main