# Ngram language model
An ngram word-level language model with backoff and Witten-Bell smoothing. To be used for teaching ngram models.

## Data
We use the data from [WikiText-2](https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/), a modern benchmark dataset for language modeling.

WikiText-2 is over 2 times larger than the Penn Treebank (PTB). WikiText-103 can also be used, but it is over 100 larger than the PTB, and comes in at about 180 MB.

## Setup
To obtain the data, run:
```bash
mkdir data
./get-data.sh
```

## Usage
To run a quick test, type:
```bash
./main.py --order 3 --interpolate --max-lines 1000
```
To train on full dataset, omit `--max-lines`.

You can choose to write the model out in [arpa format](https://cmusphinx.github.io/wiki/arpaformat/) (also used by [kenlm](https://kheafield.com/code/kenlm/)):
```bash
mkdir arpa
./main.py --order 3 --interpolate --save-arpa --name wiki-interpolate
```

## Results
We can get the following results:
```bash
./main.py --order 2 --interpolate           # Test perplexity 340.33
./main.py --order 3 --interpolate           # Test perplexity 311.53
./main.py --order 4 --interpolate           # Test perplexity 332.66
./main.py --order 5 --interpolate           # Test perplexity 337.44

./main.py --order 2 --interpolate --lower   # Test perplexity 310.93
./main.py --order 3 --interpolate --lower   # Test perplexity 282.86
./main.py --order 4 --interpolate --lower   # Test perplexity 302.15
./main.py --order 5 --interpolate --lower   # Test perplexity 307.24
```

## Comparison
We can compare with [kenlm](https://kheafield.com/code/kenlm/), which implements modified Kneser-Kney smoothing:
```bash
mkdir arpa/kenlm
lmplz -o 2 < data/wikitext-2-raw/wiki.train.raw > arpa/kenlm/wiki.2gram.arpa
lmplz -o 3 < data/wikitext-2-raw/wiki.train.raw > arpa/kenlm/wiki.3gram.arpa
lmplz -o 5 < data/wikitext-2-raw/wiki.train.raw > arpa/kenlm/wiki.5gram.arpa
```

```
In progress
```


## TODO
- [ ] Load model from arpa file.
- [ ] Backoff is broken...
- [ ] Figure out if arpa is also for interpolated model.
