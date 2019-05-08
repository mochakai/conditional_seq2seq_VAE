import sys
import json
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

def show_result(y_dict, y_label=''):
    fig, ax = plt.subplots()

    ax.plot(y_dict)

    ax.set_xlabel("iter / 100")
    ax.set_ylabel(y_label)
    ax.grid()
    plt.show()


def main(args):
    source = {}
    with open(args.file_name, 'r') as f:
        source = json.load(f)
    # pprint(list(zip(source['y_dict'].keys(), [max(i) for i in source['y_dict'].values()])))
    show_result(source['loss'], 'loss')
    show_result(source['kl'], 'kl_loss')
    show_result(source['score'], 'BLEU-4')


def get_args():
    parser = ArgumentParser()
    parser.add_argument("file_name", help="your json file name", type=str, default='')
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())

    sys.stdout.flush()
    sys.exit()