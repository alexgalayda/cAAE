#!/bin/bash

port(){
    #1 -- server port, 2 -- container port
    if [ -z "$2" ]
    then
        local container_port=8888
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
    TESTING=" bash"
    SHARA=$(mount_dir $SHARA $SHARA_CONT false)
else
    TESTING=" "
    SHARA=" "
fi

download_HCP () {
    echo Building Docker container...
    docker_build="docker build
    -f ${GET_DATASET}/Dockerfile_HCP
    -t ${NAME}_download_hcp_image
    --build-arg GET_DATASET=${GET_DATASET}
    --build-arg HCP_CONT=${HCP_CONT}
    ."
    echo $docker_build
    $docker_build

    echo running download...
    docker_run="docker run
    --name ${NAME}_download_hcp_cont
    --gpus $GPU
    --shm-size=1g
    -it
    $(mount_dir $HCP $HCP_CONT false)
    ${SHARA}
    --rm
    ${NAME}_download_hcp_image
    ${TESTING}
    "
    echo $docker_run
    $docker_run
}

download_BRATS () {
    echo Building Docker container...
    docker_build="docker build
    -f ${GET_DATASET}/Dockerfile_BRATS
    -t ${NAME}_download_brats_image
    --build-arg GET_DATASET=${GET_DATASET}
    --build-arg BRATS_CONT=${BRATS_CONT}
    --build-arg BRATS_TAR=${BRATS_TAR}
    ."
    echo $docker_build
    $docker_build

    echo running download...
    docker_run="docker run
    --name ${NAME}_download_brats_cont
    --gpus $GPU
    --shm-size=1g
    -it
    $(mount_dir $BRATS $BRATS_CONT false)
    ${SHARA}
    --rm
    ${NAME}_download_brats_image
    ${TESTING}
    "
    echo $docker_run
    $docker_run
}

training () {
    # прокинуть веса и прочую херь
    echo Building Docker container...
    if [ -n "$SHARA" ] && ! [ "$SHARA" == " " ]
    then
        echo "Shared directory = $SHARA"
    fi
    docker_build="docker build
    -f ${TRAIN}/Dockerfile
    -t ${NAME}_train_image
    --build-arg TRAIN=${TRAIN}
    --build-arg CONFIG_NAME=${CONFIG_NAME}
    ."
    echo $docker_build
    $docker_build

    mkdir -777  $RESULT
    echo running Adversarial Autoencoder example...
    docker_run="docker run
    --name ${NAME}_train_cont
    --gpus $GPU
    --shm-size=1g
    -it
    $(mount_dir $HCP $HCP_CONT true)
    $(mount_dir $WEIGHTS $WEIGHTS_CONT false)
    $(mount_dir $RESULT $RESULT_CONT false)
    $(port $PORT)
    ${SHARA}
    --rm
    ${NAME}_train_image
    ${TESTING}
    "
    echo $docker_run
    $docker_run
}

testing () {
    echo Building Docker container...
    if [ -n "$SHARA" ]
    then
        echo "Shared directory = $SHARA"
    fi
    docker_build="docker build
    -f ${TEST}/Dockerfile
    -t ${NAME}_test_image
    --build-arg TEST=${TEST}
    --build-arg CONFIG_NAME=${CONFIG_NAME}
    ."
    echo $docker_build
    $docker_build

    mkdir -m 777 $RESULT
    echo running Adversarial Autoencoder example...
    docker_run="docker run
    --name ${NAME}_test_cont
    --gpus $GPU
    --shm-size=1g
    -it
    $(mount_dir $BRATS $BRATS_CONT true)
    $(mount_dir $WEIGHTS $WEIGHTS_CONT true)
    $(mount_dir $RESULT $RESULT_CONT false)
    $(port $PORT)
    ${SHARA}
    --rm
    ${NAME}_test_image
    ${TESTING}
    "
    echo $docker_run
    $docker_run
}

jupyter () {
    mkdir -m 777 $HCP
    if [ "$(ls -A $HCP)" ]
    then
        read -p "There is already some kind of dataset in the $HCP folder. Are you sure? " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]
        then
            download_HCP
        fi
    else
        download_HCP
    fi
    mkdir -m 777 $BRATS
    if [ "$(ls -A $BRATS)" ]
        then
        read -p "There is already some kind of dataset in the $BRATS folder. Are you sure? " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]
        then
            download_BRATS
        fi
    else
        download_BRATS
    fi
    echo Building Docker container...
    if [ -n "$SHARA" ]
    then
        echo "Shared directory = $SHARA"
    fi
    docker_build="docker build
    -f ${TEST}/Dockerfile_jup
    -t ${NAME}_jupyter_image
    --build-arg TEST=${TEST}
    --build-arg CONFIG_NAME=${CONFIG_NAME}
    ."
    echo $docker_build
    $docker_build

    mkdir -m 777 $RESULT
    echo running Adversarial Autoencoder example...
    docker_run="docker run
    --name ${NAME}_jupyter_cont
    --gpus $GPU
    --shm-size=1g
    -it
    $(mount_dir $HCP $HCP_CONT true)
    $(mount_dir $BRATS $BRATS_CONT true)
    $(mount_dir $WEIGHTS $WEIGHTS_CONT false)
    $(mount_dir $RESULT $RESULT_CONT false)
    $(port $PORT)
    ${SHARA}
    --rm
    ${NAME}_jupyter_image
    "
    echo $docker_run
    $docker_run
}

test() {
    mkdir -m 777 $HCP
    if [ "$(ls -A $HCP)" ]
    then
        read -p "There is already some kind of dataset in the $HCP folder. Are you sure? " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]
        then
            download_HCP
        fi
    else
        download_HCP
    fi
    mkdir -m 777 $WEIGHTS
    if [ "$(ls -A $WEIGHTS)" ]
        then
        read -p "There is already some kind of weights in the $WEIGHTS folder. Are you sure? " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]
        then
            training
        fi
    else
        training
    fi
    mkdir -m 777 $BRATS
    if [ "$(ls -A $BRATS)" ]
        then
        read -p "There is already some kind of dataset in the $BRATS folder. Are you sure? " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]
        then
            download_BRATS
        fi
    else
        download_BRATS
    fi
    testing
}

main () {
    if [ "$1" == "jupyter" ]
    then
        jupyter
    else
        test
    fi
}

main $1
