# Ngram language model
A super-simple ngram language model, to be used for teaching.

## Data
We use the data from [WikiText-2](https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/), a modern benchmark dataset for language modeling.

WikiText-2 is over 2 times larger than the Penn Treebank (PTB). WikiText-103 can also be used, but it is over 100 larger than the PTB, and comes in at about 180 MB.

## Usage
The goal was to make this self-sufficient.

To obtain the data, run:
```bash
cd data
./get-data.sh
```

To obtain counts, run:
```bash
./get-counts.py
```

## Results
With unsmoothed MLE estimates we can get the following perplexities:
```TODO```
