# Multi-Document Summarization Domain Transfer

## Setup

```bash
conda env create --file=environment.yml
```

and the project relies on the environment variable `MDS` in the `~/.bashrc`

```bash
export MDS=/path/to/repo
```

### Dependencies

All dependencies are in `$MDS/dependencies`

**UniEval**

Repo: https://github.com/maszhongming/UniEval/tree/main

```bash
cd $MDS/dependencies
git clone https://github.com/maszhongming/UniEval.git
cd UniEval
pip install -r requirements.txt
```


**Multi-MDS**

Repo: https://github.com/ariecattan/multi_mds/tree/main

```bash
cd $MDS/dependencies
git clone git@github.com:ariecattan/multi_mds.git
cd multi_mds

conda create --name multi_mds python=3.8 
conda activate multi_mds 
pip install -r requirements.txt 
```

**MoverScore**

Repo: https://github.com/AIPHES/emnlp19-moverscore

```bash
cd $MDS/dependencies

git clone git@github.com:AIPHES/emnlp19-moverscore.git
pip install pyemd pytorch_pretrained_bert
```

**FactCCX**

```bash
cd $MDS/dependencies

git clone git@github.com:yuhui-zh15/FactCCX.git
cd FactCCX
wget https://storage.googleapis.com/sfr-factcc-data-research/factccx-checkpoint.tar.gz
tar -xzf factccx-checkpoint.tar.gz
```

## Results

The results are stored in `${MDS}/results` and use the following paths:

```bash
${MDS}/results/${model name}/${dataset name}/${dataset split}
```

Each result folder has the following files:

* `generated.jsonl`
* `evaluation.json`
* `annotate.csv`
