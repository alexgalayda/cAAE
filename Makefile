CONFIG=config/config.env
include ${CONFIG}

all: train

download: downloadHCP downloadBRATS

downloadHCP:
	mkdir -p ${HCP}
	docker-compose -f ${COMPOSE_HCP} --env-file ${CONFIG} up --build --detach
	docker attach ${NAME}_hcp_container
downloadBRATS:
	mkdir -p ${BRATS}
	docker-compose -f ${COMPOSE_BRATS} --env-file ${CONFIG} up --build --detach
	docker attach ${NAME}_brats_container
train:
	mkdir -p ${WEIGHTS}
	mkdir -p ${RESULT}
	docker-compose -f ${COMPOSE_TRAIN} --env-file ${CONFIG} up --build --detach
	docker attach ${NAME}_train_container
test:
	mkdir -p ${RESULT}
	docker-compose -f ${COMPOSE_TEST} --env-file ${CONFIG} up --build --detach
	docker attach ${NAME}_test_container
#attach:
#	docker attach ${NAME}_hcp_container
#rm:
#	docker rm ${NAME}_hcp_container
