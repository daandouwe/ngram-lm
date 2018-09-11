import os


class ArpaFile(dict):
    """Load and write arpa files.

    Format specification:
        https://cmusphinx.github.io/wiki/arpaformat/
    """
    def __init__(self, order, precision=7):
        self.order = order
        self.precision = precision
        for i in range(1, order+1):
            self[i] = []
        self['data'] = {i: 0 for i in range(1, order+1)}

    def __str__(self):
        line = []
        line.append('\\data\\')
        for i in range(1, self.order+1):
            count = self['data'][i]
            line.append(f'ngram {i}={count}')
        line.append('')
        for i in range(1, self.order):
            line.append(f'\\{i}-grams:')
            for logprob, token, discount in self[i]:
                logprob = round(logprob, self.precision)
                discount = round(discount, self.precision)
                line.append(f'{logprob}\t{token}\t{discount}')
            line.append('')
        line.append(f'\\{self.order}-grams:')
        for logprob, token in self[self.order]:
            logprob = round(logprob, self.precision)
            line.append(f'{logprob}\t{token}')
        line.append('')
        line.append('\\end\\')
        return '\n'.join(line)

    def add_ngrams(self, order, data):
        assert isinstance(order, int), order
        assert isinstance(data, list), type(data)
        self[order].extend(data)

    def add_count(self, order, count):
        assert isinstance(order, int), order
        assert isinstance(count, int), count
        self['data'][order] = count

    def write(self, path, dir='arpa'):
        path += '.arpa'
        path = os.path.join(dir, path)
        with open(path, 'w') as f:
            print(self, file=f)

    def read(self, path):
        self._parse_arpa(path)

    def _parse_arpa(self, path):
        pass


if __name__ == '__main__':
    arpa = Arpa(5)
    print(arpa)
