# Multi-Document Summarization Domain Transfer

Overleaf: https://www.overleaf.com/project/6620385b9cde38a2ca0639a9

ArXiv: https://arxiv.org/abs/2503.15768

GitHub: https://github.com/AADeLucia/Multi-Document-Summarization-Analysis


## Setup

### Environment

```bash
conda env create --file=environment.yaml
```

Configure a `.env` file in the root of the repository with the following content:

```bash
MDS=/path/to/repo
```

### Dependencies

All dependencies should be installed in `$MDS/dependencies`. Most of the original model repositories were modified. Be sure to use the specified forks, if applicable.

**Create the dependencies folder:**

```bash
source .env
mkdir "${MDS}/dependencies"
```

**Clone the modified model training and inference repositories:**

```bash
source .env
cd ${MDS}/dependencies

# Clone PRIMER
git clone git@github.com:AADeLucia/PRIMER.git
cd PRIMER
conda create --name primer python=3.7
conda activate primer
pip install --upgrade pip setuptools wheel
conda install cudatoolkit=10.0
pip install git+https://github.com/allenai/longformer.git
pip install -r requirements.txt
pip install "torchmetrics==0.3.2" "transformers>=4.2.0,<=4.29.2" 
conda activate mds

# ConvoSumm
git clone git@github.com:AADeLucia/ConvoSumm.git

# SummN
git clone git@github.com:AADeLucia/Summ-N.git
```

**Download the evaluation metrics:**

```bash
source .env
cd ${MDS}/dependencies

# UniEval
git clone https://github.com/maszhongming/UniEval.git

# MoverScore
git clone git@github.com:AIPHES/emnlp19-moverscore.git

cd $MDS
```


### Download Datasets
To download and process the datasets to `${MDS}/data`, run the following script:

```bash
bash scripts/download_datasets.sh
```


## Train Models

To replicate the model training, run the following scripts. Make sure to change the `DATASET` and `MODEL_NAME` variables as needed.

- `scripts/train_convosumm.sh`
- `scripts/train_primera.sh`
- `scripts/train_summn.sh`

`DATASET` options:
- `convosumm/reddit`
- `multixscience`
- `multinews_plus`


## Results

The results are stored in `${MDS}/results` and use the following paths:

```bash
${MDS}/results/${model name}/${dataset name}/${dataset split}
```

Each result folder has the following files:

* `generated.jsonl`
* `evaluation.json`
* `annotate.csv`
