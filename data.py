#!/usr/bin/env python
import os
from tqdm import tqdm

from utils import START, END, process


class Corpus(object):
    def __init__(self, path):
        self.train = self.tokenize(os.path.join(path, 'wiki.train.tokens'))
        self.valid = self.tokenize(os.path.join(path, 'wiki.valid.tokens'))
        self.test = self.tokenize(os.path.join(path, 'wiki.test.tokens'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        with open(path) as fin:
            num_lines = sum(1 for _ in fin.readlines())
        with open(path, 'r', encoding="utf8") as f:
            words = []
            for line in tqdm(f, total=num_lines):
                line = line.strip()
                if not line:
                    continue  # Skip empty lines.
                if line.startswith('='):
                    continue  # Skip headers.
                else:
                    sentence = [START] + [process(word) for word in line.split()] + [END]
                    words.extend(sentence)
        return words


if __name__ == '__main__':
    path = '/Users/daan/data/wikitext/wikitext-2'
    corpus = Corpus(path)
    print(len(corpus.train))
    print(corpus.train[:100])
