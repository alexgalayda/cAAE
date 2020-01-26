docker ps -a --filter "name=caae" --format "table {{.ID}}\t{{.Image}}\t{{.Names}}"
printf "\n"
docker images --filter=reference='*caae*' --format "table {{.Repository}}\t{{.Tag}}\t{{.ID}}"

