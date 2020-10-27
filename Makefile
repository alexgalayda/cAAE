CONFIG=config/config.env
include ${CONFIG}

all: downloadHCP

download: downloadHCP downloadBRATS

downloadHCP:
	docker-compose -f ${COMPOSE_HCP} --env-file ${CONFIG} up --build --detach

downloadBRATS:
	docker-compose -f ${COMPOSE_BRATS} --env-file ${CONFIG} up --build --detach

#train:

#test:
