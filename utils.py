import unicodedata

SOS = '<s>'
EOS = '</s>'
UNK = '<unk>'
YEAR = '<year>'
NUM = '<num>'

EPS = 1e-8

def is_year(s):
    try:
        int(s)
        return int(s) in range(1400, 2100)
    except ValueError:
        return False


def is_number(s):
    s = s.replace(',', '')   # 10,000 -> 10000
    s = s.replace(':', '')   # 5:30 -> 530
    s = s.replace('-', '')   # 17-08 -> 1708
    s = s.replace('/', '')   # 17/08/1992 -> 17081992
    s = s.replace('th', '')  # 20th -> 20
    s = s.replace('rd', '')  # 93rd -> 20
    s = s.replace('nd', '')  # 22nd -> 20
    s = s.replace('m', '')   # 20m -> 20
    s = s.replace('s', '')   # 20s -> 20
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


def num(word):
    if is_number(word):
        return NUM
    else:
        return word


def year(word):
    if is_year(word):
        return YEAR
    else:
        return word


def process(word, lower=False):
    if lower:
        word = word.lower()
    word = year(word)  # Turns into <year> if applicable.
    word = num(word)   # Turns into <num> if applicable.
    return word
