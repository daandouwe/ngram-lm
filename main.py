#!/usr/bin/env python
import argparse
from math import exp

from data import Corpus
from ngram import Ngram


def main(args):
    print(f'Loading corpus from `{args.data}`...')
    corpus = Corpus(args.data, order=args.order, lower=args.lower, max_lines=args.max_lines)
    model = Ngram(order=args.order)

    print('Example data:')
    print('Train:', corpus.train[:20])
    print('Valid:', corpus.valid[:20])

    print('Training model...')
    model.train(corpus.train, add_k=args.add_k, interpolate=args.interpolate)
    print(f'Ngram vocab size: {len(model.vocab):,}')
    print(f'Corpus vocab size: {len(corpus.vocab):,}')

    if args.save_arpa:
        name = args.name + '.' + str(args.order) + 'gram'
        model.save_arpa(name)
        quit()

    assert model.sum_to_one(n=10)

    text = model.generate(100)
    print('Sampled text:')
    print(' '.join(text))
    print()

    if model.is_smoothed:
        print('\nPredicting test set NLL...')
        logprob = model(corpus.test)
        nll = - logprob / len(corpus.test)
        print(f'Test NLL: {nll:.2f} | Perplexity {exp(nll):.2f}')
    else:
        exit('No evaluation with unsmoothed model: probability is probably 0 anyways.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='data/wikitext-2')
    parser.add_argument('--save-arpa', action='store_true')
    parser.add_argument('--name', default='wiki')
    parser.add_argument('--lower', action='store_true', help='lowercase data')
    parser.add_argument('--order', type=int, default=3)
    parser.add_argument('--add-k', type=int, default=0)
    parser.add_argument('--interpolate', action='store_true')
    parser.add_argument('--max-lines', type=int, default=-1)

    args = parser.parse_args()

    main(args)
