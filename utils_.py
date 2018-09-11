import os
import codecs
import string


NAMES_DIR = 'names'


CITIES_DIR = 'cities'


PAD_CHAR = '~'


COUNTRIES = [
    'af',
    'cn',
    'de',
    'fi',
    'fr',
    'in',
    'ir',
    'pk',
    'za'
]


LANGUAGES = [
    'Arabic',     'English',    'Irish',      'Polish',
    'Chinese',    'French',     'Italian',    'Portuguese',
    'Czech',      'German',     'Japanese',   'Russian',
    'Dutch',      'Greek',      'Korean',     'Scottish',
    'Spanish',    'Vietnamese'
]


# All characters used in the validation and test sets.
def all_chars(name):
    all_lines = \
        [line.strip() for line in open(os.path.join('data', name, 'val', f'{name}_val.txt')).readlines()] + \
        [line.strip() for line in open(os.path.join('data', name, 'test', f'{name}_test.txt')).readlines()]
    return set('~'.join(all_lines))


def read_train(path, order):
    with open(path) as f:
        data = f.readlines()
    pad = PAD_CHAR * order
    data = ''.join((pad + line.strip() for line in data))
    return data


def read_test(path):
    with open(path) as f:
        return [line.strip() for line in f.readlines()]
