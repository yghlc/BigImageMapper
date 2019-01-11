#!/bin/bash

# it turns out that docker have permission issues (different user inside and outside)

#build docker image
docker build -t  ubuntu1604_for_itsc .


# create and run container

#docker run -it isce_container
# --rm : remove the container on exit
# by mount home folder, the container can load environment settings in .bashrc
docker run --rm -v $HOME/:/home/hlc/ -it ubuntu1604_for_itsc
#cryo06
docker run --rm -v $HOME/:/home/hlc/ -v /docker:/docker -v /DATA3:/DATA3 -v /DATA4:/DATA4 -it ubuntu1604_for_itsc
nvidia-docker run --rm -v $HOME/:/home/hlc/ -v /docker:/docker -v /DATA3:/DATA3 -v /DATA4:/DATA4 -it ubuntu1604_for_itsc
#cryo03
nvidia-docker run --rm -v $HOME/:/home/hlc/ -v /500G:/500G -v /DATA1:/DATA1 -it ubuntu1604_for_itsc

# tag and push to docker hub
docker tag ubuntu1604_for_itsc ubuntu1604_for_itsc:v1
docker push ubuntu1604_for_itsc:v1


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



