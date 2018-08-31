#!/usr/bin/env python
import os
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from data import Corpus
from utils import START, END

EPS = 1e-45  # Fudge factor


class Ngram:
    def __init__(self, order, data_dir):
        self.order = order
        ngram = self.read(data_dir, order)
        nmingram = self.read(data_dir, order-1)
        self.probs = self.make_probs(ngram, nmingram)
        del ngram, nmingram

    def __call__(self, sentence, prepend=True):
        if isinstance(sentence, str):
            sentence = [word.lower() for word in sentence.split()]
        if isinstance(sentence, list):
            sentence = [word.lower() for word in sentence]
        if prepend:
            if self.order == 2:
                sentence = [START] + sentence
            if self.order == 3:
                sentence = [END, START] + sentence
        zeros = 0
        nll = 0.0
        n = self.order - 1  # length of histories
        for i in tqdm(range(n, len(sentence))):
            history, next = sentence[i-n:i], sentence[i]
            prob = self.prob(next, history)
            if prob == EPS:
                zeros += 1
            nll += -1 * np.log(prob)
        nll /= len(sentence)
        print(f'Out of {len(sentence):,} probabilities there were {zeros:,} zeros (epsilons).')
        return nll

    def read(self, dir, n):
        path = os.path.join(dir, f'wikitext.{n}gram')
        assert os.path.exists(path)
        ngram = dict()
        with open(path) as fin:
            for line in fin:
                line = line.split()
                prob = float(line[-1])
                history = ' '.join(line[:-1])
                ngram[history] = prob
        return ngram

    def make_probs(self, ngram, nmingram):
        cond = defaultdict(lambda: dict())
        for gram in ngram:
            words = gram.split()
            history, next = ' '.join(words[:-1]), words[-1]
            cond[history][next] = ngram[gram] / nmingram[history]
        return dict(cond)

    def check(self, history):
        if isinstance(history, list):
            order = len(history)
            history = ' '.join(history)
        if isinstance(history, str):
            order = len(history.split())
        assert order == self.order-1, f'History not the right size: `{order}`. Order is {self.order}.'
        return history

    def prob(self, next, history):
        assert isinstance(next, str)
        history = self.check(history)
        distribution = self.get_probs(history)
        if distribution is None:
            return EPS
        else:
            return distribution.get(next, EPS)

    def get_probs(self, history):
        return self.probs.get(history, None)

    def get_sample(self, distribution):
        words = np.array(list(distribution.keys()))
        probs = np.array(list(distribution.values()))
        probs = probs / probs.sum()  # Counter rouding errors in computation.
        return np.random.choice(words, p=probs)

    def sample(self, max_length=20):
        finished = ('.', '?', '!')
        assert self.order in (2, 3, 4)
        if self.order == 2:
            history = [START]
        if self.order == 3:
            history = [END, START]
        if self.order == 4:
            history = ['.', END, START]
        n = self.order - 1
        sentence = history
        while not history[-1] in finished and len(sentence) < max_length:
            distribution = self.get_probs(' '.join(history))
            next = self.get_sample(distribution)
            sentence.append(next)
            history = sentence[-n:]
        return ' '.join(sentence[n:])


if __name__ == '__main__':
    # corpus = Corpus('/Users/daan/data/wikitext/wikitext-2')
    model = Ngram(order=4, data_dir='data')

    # nll = model(corpus.test, prepend=False)
    # print(f'NLL: {nll}')

    for i in range(1, 21):
        print(f'{i}.')
        print(model.sample(50))
        print()
