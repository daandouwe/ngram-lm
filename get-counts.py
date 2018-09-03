#!/usr/bin/env python
import sys

from data import Corpus
from collections import Counter


def get_unigrams(words):
    assert isinstance(words, list)
    assert all(isinstance(word, str) for word in words)
    counts = Counter(words)
    total = sum(counts.values())
    unigrams = dict(sorted((word, count/total) for word, count in counts.items()))
    return unigrams


def get_ngrams(words, n):
    assert isinstance(words, list)
    assert all(isinstance(word, str) for word in words)
    ngrams = []
    for i in range(n, len(words)):
        ngram = ' '.join(words[i-n:i+1])
        ngrams.append(ngram)
    counts = Counter(ngrams)
    total = sum(counts.values())
    ngrams = dict(sorted((pair, count/total) for pair, count in counts.items()))
    return ngrams


def make_conditionals(unigrams, bigrams):
    """Conditional distribution.

    Format:
        cond['left right'] = p(right|left) = p(left,right) / p(left)
    """
    cond = dict()
    for pair in bigrams:
        left, right = pair.split()
        cond[pair] = bigrams[pair] / unigrams[left]
    return dict((pair, prob) for pair, prob in sorted(cond.items(), key=lambda x: x[1]))


def main(path):
    print(f'Reading and processing data from `{path}`...')
    corpus = Corpus(path)

    print(f'Collecting ngram statistics...')
    unigrams = get_unigrams(corpus.train)
    bigrams = get_ngrams(corpus.train, n=1)
    trigrams = get_ngrams(corpus.train, n=2)
    fourgrams = get_ngrams(corpus.train, n=3)
    cond = make_conditionals(unigrams, bigrams)


    for i, gram in enumerate((unigrams, bigrams, trigrams, fourgrams), 1):
        with open(f'wikitext.{i}gram', 'w') as fout:
            print(
                '\n'.join((f'{word} {prob}' for word, prob in gram.items())),
                file=fout
            )
    with open('wikitext.cond', 'w') as fout:
        print(
            '\n'.join((f'{pair} {prob}' for pair, prob in cond.items())),
            file=fout
        )


if __name__ == '__main__':
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = 'data/wikitext-2/wiki.train.tokens'
    main(path)
