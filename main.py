#!/usr/bin/env python
from data import Corpus
from ngram import Ngram

path = '/Users/daan/data/wikitext/wikitext-2'
print(f'Loading corpus from `{path}`...')
corpus = Corpus(path)
model = Ngram(order=2, data_dir='data')

print(f'Ngram vocab size: {len(model.vocab):,}')
print(f'Corpus vocab size: {len(corpus.vocab):,}')

# print('\nPredicting test set NLL...')
# nll = model(corpus.test, prepend=False)
# print(f'Test NLL: {nll}')

alpha = 0.8
print(f'\nSmoothed language model with alpha {alpha}')
nll = model(corpus.test, prepend=False, smoothed=True, alpha=alpha)
print(f'Test NLL: {nll}')
print()

model.check_smoothing(alpha)
print()

print('Samples:')
for i in range(1, 8):
    print(f'{i}.')
    print(model.sample(50))
    print()
