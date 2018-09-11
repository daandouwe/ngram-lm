#!/usr/bin/env python
import sys
import os
import string
import itertools
from collections import defaultdict, Counter

import numpy as np
from tqdm import tqdm

from arpa import ArpaFile
from utils import SOS, EPS


class Ngram(dict):
    """Assign probabilites to sentences."""
    def __init__(self, order=3, vocab=set(), sos=SOS):
        self.order = order
        self.vocab = vocab
        self.sos = sos
        self.ngrams = set()
        self.vocab_size = len(self.vocab)
        self.is_unigram = (order == 1)
        self.k = 0
        self.lambdas = dict()
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

    def train(self, data, add_k=0, interpolate=False, backoff=False):
        def normalize(counter):
            total = float(sum(counter.values()))
            return dict((word, count/total) for word, count in counter.items())

        assert not (interpolate and backoff and (add_k > 0)), 'smoothing methods are mutually exclusive'
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
        if backoff:
            self.backoff()

    def interpolate(self):
        self._interpolate = True
        if not self.is_unigram:
            print(f'Building {self.order-1}-gram model...')
            self._backoff_model = Ngram(self.order-1)
            self._backoff_model.train(self.data, interpolate=True)  # Recursive backoff.

    def backoff(self):
        exit('Sorry, backoff is broken.')
        self._backoff = True
        if not self.is_unigram:
            print(f'Building {self.order-1}-gram model...')
            self._backoff_model = Ngram(self.order-1)
            self._backoff_model.train(self.data, backoff=True)  # Recursive backoff.
            print(f'Constructing backoff alphas for {self.order-1}-gram model...')
            self._construct_backoff_alphas()

    def prob(self, history, word, verbose=True):
        assert isinstance(word, str), word
        if isinstance(history, list):
            history = ' '.join(history)
        assert isinstance(history, str), history
        ngram = history + ' ' + word
        if not all(word in self.vocab for word in set(ngram.split())):
            return 0  # we work with a fixed vocabulary, also when smoothing
        elif self._add_k:
            prob = self._smooth_add_k(history, word)
        elif self._interpolate:
            prob = self._smooth_interpolate(history, word)
        elif self._backoff:
            prob = self._smooth_backoff(history, word, verbose=verbose)
        else:
            prob = self._prob(history, word)
        return prob

    def _prob(self, history, word):
        ngram = history + ' ' + word
        if ngram in self.ngrams:
            prob = self[history][word]
        else:
            prob = 0
        return prob

    def logprob(self, history, word):
        return np.log(self.prob(history, word))

    def perplexity(self, data, sos=False):
        data = (self.order-1) * [self.sos] + data if sos else data
        nll = self(data) / len(data)
        return np.exp(-nll)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%% #
    #      Smoothing methods      #
    # %%%%%%%%%%%%%%%%%%%%%%%%%%% #

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
        lmbda = self._witten_bell(history)
        if self.is_unigram:
            higher = self.get(word, 0)
            lower = 1.0 / self.vocab_size  # uniform model
        else:
            higher = self._prob(history, word)
            lower_history = ' '.join(history.split()[1:])
            lower = self._backoff_model.prob(lower_history, word)
        return lmbda * higher + (1 - lmbda) * lower

    def _witten_bell(self, history):
        assert isinstance(history, str), history
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

    def _smooth_backoff(self, history, word, verbose):
        if self.is_unigram:
            higher = self.get(word, 0)
        else:
            higher = self._prob(history, word)
        if higher > 0:
            if verbose: print(self.order, history, word, higher)
            return higher
        else:
            if self.is_unigram:
                lower = 1.0 / self.vocab_size  # uniform model
                alpha = self._backoff_alphas.get(word, 1)
            else:
                lower_history = ' '.join(history.split()[1:])
                lower = self._backoff_model._smooth_backoff(lower_history, word, True)
                alpha = self._backoff_alphas.get(history, 1)
            print('backoff', self.order, history, word, higher, lower, alpha, alpha*lower)
            return alpha * lower

    def _construct_backoff_alphas(self):
        self._backoff_alphas = dict()
        # For each condition (or context)
        for history in self.all_histories:
            backoff_history = ' '.join(history.split()[1:])
            backoff_total_pr = 0.0
            total_observed_pr = 0.0

            # this is the subset of words that we OBSERVED following
            # this context.
            # i.e. Count(word | context) > 0
            for word in self[history].keys():
                total_observed_pr += self.prob(history, word, verbose=False)
                # we also need the total (n-1)-gram probability of
                # words observed in this n-gram context
                backoff_total_pr += self._backoff_model.prob(backoff_history, word)

            # beta is the remaining probability weight after we factor out
            # the probability of observed words.
            # As a sanity check, both total_observed_pr and backoff_total_pr
            # must be GE 0, since probabilities are never negative
            total_observed_pr = min(total_observed_pr, 1)
            beta = 1.0 - total_observed_pr

            # backoff total has to be less than one, otherwise we get
            # an error when we try subtracting it from 1 in the denominator
            backoff_total_pr = min(backoff_total_pr, 1)
            alpha_history = beta / (1.0 - backoff_total_pr + EPS)

            self._backoff_alphas[history] = alpha_history

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    #      Methods for text generation    #
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    def generate(self, num_words, history=[]):
        text = history
        for i in range(num_words):
            text.append(self._generate_one(text))
        return text

    def _generate_one(self, history):
        # Pad in case history is too short.
        history = (self.order-1) * [self.sos] + history
        # Select only what we need.
        history = history[-(self.order-1):]
        # Turn list into string.
        history = ' '.join(history)
        # if self.is_smoothed:
            # Use entire vocabulary.
            # probs, words = zip(*[(self.prob(history, word), word) for word in self.vocab])
        # else:
            # Use only seen words.
            # probs, words = zip(*[(self.prob(history, word), word) for word in self[history]])
        probs, words = zip(*[(self.prob(history, word), word) for word in self[history]])
        return self._sample(probs, words)

    def _sample(self, probs, words):
        # Take care of the rounding errors which numpy does not like.
        probs = np.array(probs) / np.array(probs).sum()
        return np.random.choice(words, p=probs)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    #      Methods for handling arpa files      #
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    def save_arpa(self, path):
        assert (self._interpolate or self._backoff), 'must be interpolated model to write arpa file'
        arpa = ArpaFile(self.order)
        counts = self._ngram_counts()
        ngrams = self._arpa_ngrams(self.order)
        for order in range(1, self.order+1):
            arpa.add_ngrams(order, ngrams[order])
            arpa.add_count(order, counts[order])
        arpa.write(path)

    def load_arpa(self, path):
        arpa = parse_arpa(path)
        self._from_arpa(arpa, highest=True)

    def _arpa_ngrams(self, highest_order):
        data = []
        for ngram in sorted(self.ngrams):
            ngram = ngram.split()
            history, word = ' '.join(ngram[:-1]), ngram[-1]
            logprob = np.log10(self.prob(history, word))
            ngram = ' '.join(ngram)
            if self.order == highest_order:
                data.append((logprob, ngram))
            else:
                discount = np.log10(1 - self._witten_bell(history))
                data.append((logprob, ngram, discount))
        if self.is_unigram:
            return {self.order: data}
        else:
            higher = {self.order: data}
            lower = self._backoff_model._arpa_ngrams(highest_order)
            return {**higher, **lower}  # merge dictionaries

    def _ngram_counts(self):
        if self.is_unigram:
            return {1: len(self.ngrams)}
        else:
            higher = {self.order: len(self.ngrams)}
            lower = self._backoff_model._ngram_counts()
            return {**higher, **lower}  # merge dictionaries

    def _from_arpa(self, arpa, highest=False):
        raise NotImplementedError('no arpa loading yet')

    # %%%%%%%%%%%%% #
    #    Tests      #
    # %%%%%%%%%%%%% #

    def sum_to_one(self, eps=EPS, random_sample=True, n=100):
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

    # %%%%%%%%%%%%%%%% #
    #    Properties    #
    # %%%%%%%%%%%%%%%% #

    @property
    def all_histories(self):
        return [' '.join(ngram.split()[:-1]) for ngram in self.ngrams]

    @property
    def is_smoothed(self):
        return (self._add_k or self._interpolate or self._backoff)
