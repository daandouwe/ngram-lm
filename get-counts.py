#!/usr/bin/env python
import sys
import json
from collections import Counter

from data import Corpus


def get_unigrams(words):
    assert isinstance(words, list)
    assert all(isinstance(word, str) for word in words)
    counts = Counter(words).most_common()  # sorted by value
    return dict(counts)


def get_ngrams(words, history):
    assert isinstance(words, list)
    assert all(isinstance(word, str) for word in words)
    ngrams = []
    for i in range(history, len(words)):
        ngram = ' '.join(words[i-history:i+1])
        ngrams.append(ngram)
    counts = Counter(ngrams).most_common()  # sorted by value
    return dict(counts)


def main(data_dir):
    print(f'Reading and processing data from `{data_dir}`...')
    corpus = Corpus(data_dir)

    print(f'Collecting ngram counts...')
    print('Unigram...')
    unigrams = get_unigrams(corpus.train)
    print('Bigram...')
    bigrams = get_ngrams(corpus.train, history=1)
    print('Trigram...')
    trigrams = get_ngrams(corpus.train, history=2)
    print('Fourgram...')
    fourgrams = get_ngrams(corpus.train, history=3)

    for i, gram in enumerate((unigrams, bigrams, trigrams, fourgrams), 1):
        with open(f'data/wikitext.{i}gram.json', 'w') as f:
            json.dump(gram, f, indent=4)

    print('Done.')

if __name__ == '__main__':
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = 'data/wikitext-2'
    main(data_dir)
