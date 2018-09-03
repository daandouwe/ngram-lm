#!/usr/bin/env python
from data import Corpus
from ngram import Ngram

path = '/Users/daan/data/wikitext/wikitext-2'
print(f'Loading corpus from `{path}`...')
corpus = Corpus(path)
model = Ngram(order=3, data_dir='data')

print('\nPredicting test set NLL...')
nll = model(corpus.test, prepend=False)
print(f'Test NLL: {nll}')

alpha = 0.8
print(f'\nSmoothing language model with alpha {alpha}')
model.smooth(alpha=0.8)

print('\nPredicting test set NLL...')
nll = model(corpus.test, prepend=False)
print(f'Test NLL: {nll}')
print()

print('Samples:')
for i in range(1, 8):
    print(f'{i}.')
    print(model.sample(50))
    print()
