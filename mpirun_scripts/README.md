## start docker in each node
```
docker run --gpus all -it --rm -v /path/in/host:/path/in/vm --device=/dev/infiniband --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --net host  nvcr.io/ea-bignlp/ga-participants/nemofw-training:24.05 bash 
```

## set up ssh
### create keys and no password ssh
```
ssh-keygen -t rsa
```
enter to generate id_rsa, id_rsa.pub  in .ssh directory

enter .ssh directory
```
cat id_rsa.pub > authorized_keys
```

and generate a config, the content in config is like bellow,  the hostname is the node you use, the port is the ssh port you use in container. 
```
Host h20n4
    Hostname h20n4
    Port 12345
    StrictHostKeyChecking no

Host h20n5
    Hostname h20n5
    Port 12345
    StrictHostKeyChecking no
```
in the .ssh directory, you have "authorized_keys  config  id_rsa  id_rsa.pub"

you can back up this .ssh dir, and copy it to other node's container

start a sshd , use the port in config
```
/usr/sbin/sshd -p 12345
```

### test the mpi environment

test the bellow command in container, make sure the mpirun is execuated correctly in container, not in host, you can replace 'hostname' to other command to verify this.
```
mpirun -np 8 -H h20n4:8,h20n5:8 hostname
```

### run the nemo test.

MODEL=llama2_70b_20L TP=8 CP=1 PP=1 MBS=2 GBS=16 TPOVERLAP=true bash  ./mpirun.sh

