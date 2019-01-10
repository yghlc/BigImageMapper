#!/bin/bash

#build docker image
docker build -t  ubuntu1604_for_itsc .


# create and run container
#docker run -it -v --name isce_test isce_container

#docker run -it isce_container

docker run -v /home/hlc/programs/docker_isce_v2.2:/home/hlc/programs/isce_v2.2 -it isce_container


### launch a new terminal to the container, e9ef58868d14 is the container by "nvidia-docker ps" or "nvidia-docker ps -a"
#nvidia-docker exec -it e9ef58868d14 bash

### start the container at the background
#4cc63f4a50d1 is got by "nvidia-docker ps -q -l"
#nvidia-docker start e9ef58868d14

### attach to the container
#nvidia-docker attach e9ef58868d14

### install isce after entering the docker container
/home/hlc/programs/isce_v2.2/temp/isce-2.2.0/setup
#./install.sh -p /home/hlc/programs/isce_v2.2
./install.sh  -v -c /home/hlc/programs/isce_v2.2/SConfigISCE



