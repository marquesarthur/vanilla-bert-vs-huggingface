# vanilla-bert-vs-huggingface



## Setup 



#### Vanilla


```
module load  ipython-kernel/3.7
virtualenv --no-download ~/vanilla
source ~/vanilla/bin/activate
pip install --no-index --upgrade pip
```




libs: note that not all of them are available using `--no-index`

```
cd vanilla
pip install --no-index -r requirements.txt
./setup.sh
```







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






## Exec



#### Vanilla



```
module load  ipython-kernel/3.7
source ~/vanilla/bin/activate
salloc --time=02:59:59  --mem-per-cpu=32G --gres=gpu:1 --account=def-mageed --job-name=arthur-vanilla srun $VIRTUAL_ENV/bin/notebook.sh
```


Make sure to change the python kernel to `Arthur vanilla`


