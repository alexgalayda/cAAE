CONFIG=config/config.env
include ${CONFIG}

all: downloadHCP

download: downloadHCP downloadBRATS

downloadHCP:
	mkdir -p ${STORAGE}
	docker-compose -f ${COMPOSE_HCP} --env-file ${CONFIG} up --build --detach
	#docker attach ${NAME}_hcp_container
rm:
	docker rm ${NAME}_hcp_container
downloadBRATS:
	docker-compose -f ${COMPOSE_BRATS} --env-file ${CONFIG} up --build --detach

attach:
	docker attach ${NAME}_hcp_container
#train:

#test: