#!/usr/bin/env python
import sys
import os
import string
from collections import defaultdict, Counter
import itertools

import numpy as np
from tqdm import tqdm

from utils import SOS
from arpa import Arpa


class Ngram(dict):
    def __init__(self, order=3, vocab=set()):
        self.order = order
        self.vocab = vocab
        self.ngrams = set()
        self.vocab_size = len(self.vocab)
        self.is_unigram = (order == 1)
        self.k = 0
        self.sos = SOS
        self._add_k = False
        self._interpolate = False
        self._backoff = False

    def __call__(self, data):
        logprob = 0
        for history, word in self.get_ngrams(data):
            prob = self.prob(history, word)
            logprob += np.log(prob)
        return logprob

    def get_ngrams(self, data):
        for i in range(len(data)-self.order+1):
            history, word = self.get_ngram(data, i)
            yield history, word

    def get_ngram(self, data, i):
        history, word = data[i:i+self.order-1], data[i+self.order-1]
        history = ' '.join(history)
        return history, word

    def train(self, data, add_k=0, interpolate=False):
        def normalize(counter):
            total = float(sum(counter.values()))
            return dict((word, count/total) for word, count in counter.items())

        self._add_k = (add_k > 0)
        self.k = add_k
        self.data = data
        self.vocab.update(set(data))
        self.vocab_size = len(self.vocab)
        if self.is_unigram:
            self.ngrams = set(data)
            counts = Counter(data)
            lm = normalize(counts)
        else:
            counts = defaultdict(Counter)
            for history, word in self.get_ngrams(data):
                counts[history][word] += 1
                ngram = history + ' ' + word
                self.ngrams.add(ngram)
            lm = ((hist, normalize(words)) for hist, words in counts.items())
        self.counts = counts
        super(Ngram, self).__init__(lm)
        if interpolate:
            self.interpolate()

    def _prob(self, history, word):
        ngram = history + ' ' + word
        if ngram in self.ngrams:
            prob = self[history][word]
        else:
            prob = 0
        return prob

    def _smooth_add_k(self, history, word):
        assert self.k > 0, self.k
        try:
            self.counts[history]
            count = self.counts[history].get(word, 0)
            total = sum(self.counts[history].values())
        except KeyError:
            count = 0
            total = 0
        prob = (self.k + count) / (self.k*self.vocab_size + total)
        return prob

    def _smooth_interpolate(self, history, word):
        lmbda = self.witten_bell(history)
        if self.is_unigram:
            higher = self.get(word, 0)
            lower = 1.0 / self.vocab_size  # uniform model
        else:
            higher = self._prob(history, word)
            lower_history = ' '.join(history.split()[1:])
            lower = self._backoff_model.prob(lower_history, word)
        return lmbda * higher + (1 - lmbda) * lower

    def _smooth_backoff(self, history, word):
        raise NotImplementedError('no backoff yet.')

    def prob(self, history, word):
        ngram = history + ' ' + word
        if not all(word in self.vocab for word in set(ngram.split())):
            return 0
        elif self._add_k:
            prob = self._smooth_add_k(history, word)
        elif self._interpolate:
            prob = self._smooth_interpolate(history, word)
        elif self._backoff:
            prob = self._smooth_backoff(history, word)
        else:
            prob = self._prob(history, word)
        return prob

    def logprob(self, history, word):
        return np.log(self.prob(history, word))

    def interpolate(self):
        print(f'Building {self.order-1}-gram model...')
        self._interpolate = True
        if not self.is_unigram:
            self._backoff_model = Ngram(self.order - 1)
            self._backoff_model.train(self.data, interpolate=True)  # Recursive backoff.

    def witten_bell(self, history):
        if self.is_unigram:
            unique_follows = self.counts.get(history, 0)
            total = self.counts.get(history, 0)
        else:
            unique_follows = len(self.counts.get(history, []))
            total = sum(self.counts.get(history, dict()).values())
        # Avoid division by zero.
        if unique_follows == 0 and total == 0:
            frac = 1  # justified by limit? n/n -> 1 as n -> 0
        elif unique_follows == 0 and not total == 0:
            frac = 0
        else:
            frac = unique_follows / (unique_follows + total)
        return 1 - frac

    def perplexity(self, data, sos=False):
        if sos:
            data = self.order * [self.sos] + data
        nll = self(data) / len(data)
        return np.exp(-nll)

    def _generate_one(self, history):
        # Pad in case history is too short.
        history = (self.order-1) * [self.sos] + history
        # Select only what we need.
        history = history[-(self.order-1):]
        # Turn list into string.
        history = ' '.join(history)
        if self._add_k or self._interpolate or self._backoff:
            probs, words = zip(*[(self.prob(history, word), word) for word in self.vocab])
        else:
            probs, words = zip(*[(self.prob(history, word), word) for word in self[history]])
        return self._sample(probs, words)

    def _sample(self, probs, words):
        # Take care of the rounding errors that numpy does not like.
        probs = np.array(probs) / np.array(probs).sum()
        return np.random.choice(words, p=probs)

    def generate(self, num_words, history=[]):
        text = history
        for i in range(num_words):
            text.append(self._generate_one(text))
        return text


    def _arpa_ngrams(self, highest_order):
        arpa_data = []
        for ngram in sorted(self.ngrams):
            ngram = ngram.split()
            history, word = ' '.join(ngram[:-1]), ngram[-1]
            logprob = np.log10(self.prob(history, word))
            ngram = ' '.join(ngram)
            if self.order == highest_order:
                arpa_data.append((logprob, ngram))
            else:
                discount = np.log10(1 - self.witten_bell(history))
                arpa_data.append((logprob, ngram, discount))
        if self.is_unigram:
            return {self.order: arpa_data}
        else:
            higher = {self.order: arpa_data}
            lower = self._backoff_model._arpa_ngrams(highest_order)
            return {**higher, **lower}  # merge dictionaries

    def _ngram_counts(self):
        if self.is_unigram:
            return {1: len(self.ngrams)}
        else:
            higher = {self.order: len(self.ngrams)}
            lower = self._backoff_model._ngram_counts()
            return {**higher, **lower}  # merge dictionaries

    def save_arpa(self, path):
        assert self._interpolate, 'must be an interpolated model to write arpa file'
        arpa = Arpa(self.order)
        arpa_counts = self._ngram_counts()
        arpa_ngrams = self._arpa_ngrams(self.order)
        for order in range(1, self.order+1):
            arpa.add_ngrams(order, arpa_ngrams[order])
            arpa.add_count(order, arpa_counts[order])
        arpa.write(path)

    def _from_arpa(self, arpa, highest=False):
        pass

    def load_arpa(self, path):
        """Construct Ngram from arpa file."""
        arpa = parse_arpa(path)
        self._from_arpa(arpa, highest=True)


    def sum_to_one(self, eps=1e-8, random_sample=True, n=100):
        print('Checking if probabilities sum to one...')
        histories = self.all_histories
        if random_sample:
            print(f'Checking a random subset of size {n}.')
            idxs = np.arange(len(histories))
            np.random.shuffle(idxs)
            histories = [histories[i] for i in idxs[:n]]
        for history in tqdm(histories):
            total = 0
            for word in self.vocab:
                total += self.prob(history, word)
            if abs(1.0 - total) > eps:
                exit(f'p(word|`{history}`) sums to {total}!')
        return True

    @property
    def all_histories(self):
        return [' '.join(ngram.split()[:-1]) for ngram in self.ngrams]

    @property
    def is_smoothed(self):
        return (self._add_k or self._interpolate or self._backoff)
