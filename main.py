#!/usr/bin/env python
import argparse
import os
from math import exp

from data import Corpus
from ngram import Ngram


def main(args):
    print(f'Loading corpus from `{args.data}`...')
    corpus = Corpus(args.data, order=args.order, lower=args.lower, max_lines=args.max_lines)
    model = Ngram(order=args.order)
    name = f'{args.name}.{args.order}gram'

    print('Example data:')
    print('Train:', corpus.train[:20])
    print('Valid:', corpus.valid[:20])

    print('Training model...')
    model.train(corpus.train,
        add_k=args.add_k, interpolate=args.interpolate, backoff=args.backoff)
    print(f'Vocab size: {len(model.vocab):,}')

    if args.save_arpa:
        print(f'Saving model to `{name}`...')
        model.save_arpa(name)

    assert model.sum_to_one(n=10)

    print('Generating text...')
    text = model.generate(100)
    text = ' '.join(text)
    path = os.path.join(args.out, f'generated.{name}.txt')
    print(text)
    with open(path, 'w') as f:
        print(text, file=f)

    if model.is_smoothed:
        print('\nPredicting test set NLL...')
        logprob = model(corpus.test)
        nll = -logprob / len(corpus.test)
        print(f'Test NLL: {nll:.2f} | Perplexity {exp(nll):.2f}')
        path = os.path.join(args.out, f'result.{name}.txt')
        with open(path, 'w') as f:
            print(f'Test NLL: {nll:.2f} | Perplexity {exp(nll):.2f}', file=f)
    else:
        exit('No evaluation with unsmoothed model: probability is probably 0 anyways.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Dir args
    parser.add_argument('--data', default='data/wikitext-2',
                        help='data directory')
    parser.add_argument('--out', default='out',
                        help='directory to write out to')
    parser.add_argument('--name', default='wiki',
                        help='model name')

    # Data args
    parser.add_argument('--max-lines', type=int, default=-1,
                        help='reading subset of data (useful for development)')
    parser.add_argument('--lower', action='store_true', help='lowercase data',
                        help='lowercase data')

    # Model args
    parser.add_argument('--order', type=int, default=3
                        help='order of language model')
    parser.add_argument('--add-k', type=int, default=0,
                        help='add k smoothing')
    parser.add_argument('--interpolate', action='store_true',
                        help='witten-bell interpolation')
    parser.add_argument('--backoff', action='store_true',
                        help='backoff smoothin')
    parser.add_argument('--save-arpa', action='store_true',
                        help='save model to an arpa file')

    args = parser.parse_args()

    main(args)
