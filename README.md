# vanilla-bert-vs-huggingface



## Setup 








#### Huggingface


```
module load  ipython-kernel/3.7
virtualenv --no-download ~/hface
source ~/hface/bin/activate
pip install --no-index --upgrade pip

```



```
cd hugging
pip install --no-index -r requirements.txt
./setup.sh
```





salloc --time=00:59:59  --mem-per-cpu=12G --gres=gpu:1 --account=def-mageed --job-name=arthur-cp6 srun $VIRTUAL_ENV/bin/notebook.sh



## Exec

```
cd projects/def-mageed/msarthur/vanilla-bert-vs-huggingface/
```


#### Vanilla



```
cd projects/def-mageed/msarthur/vanilla-bert-vs-huggingface/
module load StdEnv/2018 python/3
source ~/vanilla/bin/activate
salloc --time=02:59:59  --mem-per-cpu=32G --gres=gpu:1 --account=def-mageed --job-name=arthur-vanilla srun $VIRTUAL_ENV/bin/notebook.sh
```


Make sure to change the python kernel to `Arthur vanilla`







#### Hugging



```
cd projects/def-mageed/msarthur/vanilla-bert-vs-huggingface/
module load  ipython-kernel/3.7
source ~/hface/bin/activate
salloc --time=00:59:59  --mem-per-cpu=16G --gres=gpu:1 --account=def-mageed --job-name=arthur-ds srun $VIRTUAL_ENV/bin/notebook.sh

```


Make sure to change the python kernel to `Arthur hugging`




/home/msarthur/.local/share/jupyter/runtime/notebook_cookie_secret