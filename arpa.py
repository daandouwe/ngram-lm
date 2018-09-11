def parse_arpa(path):
    pass


def write_arpa(dictionary, path):
    pass


class Arpa(dict):
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
            name = 'ngram ' + str(i)
            line.append(name + '=' + str(self['data'][i]))
        line.append('')
        for i in range(1, self.order):
            name = str(i) + '-grams'
            line.append('\\' + name + ':')
            for logprob, token, discount in self[i]:
                logprob, discount = round(logprob, self.precision), round(discount, self.precision)
                line.append('{}\t{}\t{}'.format(logprob, token, discount))
            line.append('')
        name = str(self.order) + '-grams'
        line.append('\\' + name + ':')
        for logprob, token in self[self.order]:
            logprob = round(logprob, self.precision)
            line.append('{}\t{}'.format(logprob, token))
        line.append('')
        line.append('\\end\\')
        return '\n'.join(line)

    def add_ngrams(self, order, data):
        self[order].extend(data)

    def add_count(self, order, count):
        self['data'][order] = count

    def write(self, path):
        path += '.arpa'
        with open(path, 'w') as f:
            print(self, file=f)



if __name__ == '__main__':
    arpa = Arpa(5)
    print(arpa)
