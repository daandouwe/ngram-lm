#!/usr/bin/env python
import os
import json
from copy import deepcopy
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from data import Corpus
from utils import START, END

EPS = 1e-45  # Fudge factor.

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


class Ngram:
    #TODO: extend smoothin...
    #TODO: Load counts from all orders up to n, for smoothing...

    def __init__(self, order, data_dir):
        self.order = order  # e.g. a trigram has order 3...
        self.history = order - 1  # ...and has a history of 2 words.

        self.ngram = self.read(data_dir, order)
        self.nmingram = self.read(data_dir, self.history)
        self.vocab = self.make_vocab(self.ngram)

        self.ngram_total = sum(self.ngram.values())
        self.nmingram_total = sum(self.nmingram.values())

        self.probs = self.make_probs(self.ngram, self.nmingram)

    def __call__(self, sentence, prepend=True, smoothed=False, alpha=None):
        if isinstance(sentence, str):
            sentence = [word.lower() for word in sentence.split()]
        if isinstance(sentence, list):
            sentence = [word.lower() for word in sentence]
        if prepend:
                sentence = self.prepend() + sentence
        zeros = 0; nll = 0.0
        for i in tqdm(range(self.history, len(sentence))):
            history, next = sentence[i-self.history:i], sentence[i]
            if smoothed:
                prob = self.smoothed_prob(next, history, alpha)
            else:
                prob = self.prob(next, history)
            if prob == EPS:
                zeros += 1
            nll += -1 * np.log(prob)
        nll /= len(sentence)
        print(f'Out of {len(sentence):,} probabilities there were {zeros:,} zeros (each zero replaced with eps {EPS:.1e}).')
        return nll

    def read(self, dir, n):
        path = os.path.join(dir, f'wikitext.{n}gram.json')
        assert os.path.exists(path)
        with open(path) as fin:
            ngram_counts = json.load(fin)
        return ngram_counts

    def make_vocab(self, ngram):
        vocab = set()
        for gram in ngram.keys():
            vocab.update(gram.split())
        return vocab

    def make_probs(self, ngram, nmingram):
        conditional = defaultdict(lambda: dict())
        for gram in ngram.keys():
            words = gram.split()
            history, next = ' '.join(words[:-1]), words[-1]
            p_gram = ngram[gram] / self.ngram_total
            p_history = nmingram[history] / self.nmingram_total
            conditional[history][next] = p_gram / p_history
        return dict(conditional)  # stop defaultdict behaviour

    def prepend(self):
        if self.history == 1:
            history = [START]
        if self.history == 2:
            history = [END, START]
        if self.history == 3:
            history = ['.', END, START]
        return history

    def check(self, history):
        if isinstance(history, list):
            order = len(history)
            history = ' '.join(history)
        if isinstance(history, str):
            order = len(history.split())
        assert order == self.order-1, f'History not the right size: `{order}`. Order is {self.order}.'
        return history

    def get_probs(self, history):
        return self.probs.get(history, None)

    def prob(self, next, history):
        assert isinstance(next, str)
        history = self.check(history)
        distribution = self.get_probs(history)
        if distribution is None:
            return EPS
        else:
            return distribution.get(next, EPS)

    def smoothed_prob(self, next, history, alpha):
        assert isinstance(next, str)
        assert isinstance(alpha, float)
        assert self.order == 2, 'smoothing only implemented for bigram models.'
        history = self.check(history)

        distribution = self.get_probs(history)
        p_bigram = distribution.get(next, 0)
        p_unigram = self.nmingram[next] / self.nmingram_total

        p_unigram = alpha * p_unigram + (1 - alpha) * (1 / len(self.vocab))
        p_bigram = alpha * p_bigram + (1 - alpha) * p_unigram

        return p_bigram

    def check_smoothing(self, alpha):
        for history in ('the', 'something', 'strange', 'word'):
            total = 0.0
            for next in self.vocab:
                total += self.smoothed_prob(next, history, alpha)
            print(total)  # should equal 1

    def witten_bell(self, history):
        unique = 1
        count = 1
        return 1 - (unique / (unique + count))

    def get_sample(self, distribution):
        words = np.array(list(distribution.keys()))
        probs = np.array(list(distribution.values()))
        probs = probs / probs.sum()  # Counter rouding errors in computation.
        return np.random.choice(words, p=probs)

    def sample(self, max_length=20):
        finished = ('.', '?', '!')
        assert self.order in (2, 3, 4)
        history = self.prepend()
        n = self.order - 1
        sentence = history
        while not history[-1] in finished and len(sentence) < max_length:
            distribution = self.get_probs(' '.join(history))
            next = self.get_sample(distribution)
            sentence.append(next)
            history = sentence[-n:]
        return ' '.join(sentence[n:])
