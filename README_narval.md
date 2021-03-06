# vanilla-bert-vs-huggingface



module load  ipython-kernel/3.7
virtualenv --no-download ~/hface_test
source ~/hface_test/bin/activate
pip install --no-index --upgrade pip



https://stackoverflow.com/questions/9232568/identifying-the-dependency-relationship-for-python-packages-installed-with-pip


## Setup 



#### Huggingface


```
module load  ipython-kernel/3.7
virtualenv --no-download ~/hface_gpu
source ~/hface_gpu/bin/activate
pip install --no-index --upgrade pip

```



```
cd hugging
pip install --no-index -r requirements.txt
./setup.sh

```
https://stackoverflow.com/questions/63231021/tensorflow-gpu-issue-cuda-runtime-error-device-kernel-image-is-invalid

I need to downgrade to Tensorflow 2.3



I need at least tensorflow 2.3 to use transformers





salloc --time=00:59:59  --mem-per-cpu=16G --gres=gpu:1 --account=rrg-mageed --job-name=arthur-ds srun $VIRTUAL_ENV/bin/notebook.sh



## Exec

```
cd projects/def-mageed/msarthur/cp6-highlights
module load  ipython-kernel/3.7
source ~/hface_gpu/bin/activate
salloc --time=00:59:59  --mem-per-cpu=16G --gres=gpu:1 --account=def-mageed --job-name=arthur-ds srun $VIRTUAL_ENV/bin/notebook.sh


-- faster allocation
salloc --time=01:59:59  --mem-per-cpu=16G --gres=gpu:1 --account=def-mageed --job-name=arthur-ds srun $VIRTUAL_ENV/bin/notebook.sh

```


Make sure to change the python kernel to `Arthur hugging`


## Util


Clone bert uncased to local scrath

```
git clone https://huggingface.co/bert-base-uncased
```


```
squeue -u $USER

scancel -u $USER

sshuttle --dns -Nr msarthur@graham.computecanada.ca

 sshuttle --dns -Nr msarthur@graham.computecanada.ca -x 199.241.166.2 
```



### Debug CUDA issue:
```
salloc --time=00:59:59  --mem-per-cpu=16G --gres=gpu:1 --account=def-mageed --job-name=arthur-ds


import tensorflow as tf
from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

get_available_gpus()

```


If the output is....:


```

```


192.168.0.1