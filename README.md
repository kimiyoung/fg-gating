# Words or Characters? Fine-grained Gating for Reading Comprehension

## Intro

This is an implementation of the paper
```
Words or Characters? Fine-grained Gating for Reading Comprehension
Zhilin Yang, Bhuwan Dhingra, Ye Yuan, Junjie Hu, William W. Cohen, Ruslan Salakhutdinov
ICLR 2017
```

SOTA results on Children's Book Test (CBT) and Who-Did-What (WDW)

## Get data

Download and extract the preprocessed data.

```
wget http://kimi.ml.cmu.edu/fg_data/cbtcn.tar
tar -xvf cbtcn.tar
wget http://kimi.ml.cmu.edu/fg_data/cbtne.tar
tar -xvf cbtne.tar
wget http://kimi.ml.cmu.edu/fg_data/wdw.tar.gz
tar -xvzf wdw.tar.gz
mkdir wdw_relaxed
cd wdw_relaxed
wget http://kimi.ml.cmu.edu/fg_data/wdw_relaxed/data.tgz
tar -xvzf data.tgz
```

## Requirements

Lasagne + Theano. Python 2.7.

Install Lasagne and Theano with the instructions here: https://github.com/Lasagne/Lasagne#installation

## Run the Models

### CBTCN
```
python run.py --dropout 0.4 --dataset cbtcn --seed 1
```

### CBTNE
```
python run.py --dropout 0.4 --dataset cbtne --seed 31
```

### WDW
```
python run.py --dropout 0.3 --dataset wdw --seed 11
```

### WDW Relaxed
```
python run.py --dropout 0.3 --dataset wdw_relaxed --seed 51
```
