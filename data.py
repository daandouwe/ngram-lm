#!/usr/bin/env python
"""
Adapted from https://github.com/pytorch/examples/tree/master/word_language_model.
"""

import os
from tqdm import tqdm

from utils import SOS, EOS, UNK, process


class Corpus(object):
    def __init__(self, path, order, lower=False, max_lines=-1):
        self.order = order
        self.lower = lower
        self.max_lines = max_lines
        self.vocab = set()
        self.train = self.tokenize(os.path.join(path, 'wiki.train.tokens'), training_set=True)
        self.valid = self.tokenize(os.path.join(path, 'wiki.valid.tokens'))
        self.test = self.tokenize(os.path.join(path, 'wiki.test.tokens'))

    def tokenize(self, path, training_set=False):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        with open(path) as fin:
            num_lines = sum(1 for _ in fin.readlines())
        with open(path, 'r', encoding="utf8") as f:
            words = []
            for i, line in enumerate(tqdm(f, total=num_lines)):
                if self.max_lines > 0 and i > self.max_lines:
                    break
                line = line.strip()
                if not line:
                    continue  # Skip empty lines.
                elif line.startswith('='):
                    continue  # Skip headers.
                else:
                    sentence = (self.order - 1) * [SOS] + \
                        [process(word, self.lower) for word in line.split()] + [EOS]
                    if training_set:
                        words.extend(sentence)
                        self.vocab.update(sentence)
                    else:
                        sentence = [word if word in self.vocab else UNK for word in sentence]
                        words.extend(sentence)
        return words


if __name__ == '__main__':
    path = 'data/wikitext-2'
    corpus = Corpus(path, order=3)
    print(len(corpus.test))
    print(corpus.test[:100])
