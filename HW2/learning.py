import unittest, argparse
import unittest
import pickle, argparse, os, sys
from sklearn.metrics import accuracy_score
import csv
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.utils.data.dataloader as dataloader
import nltk
import os
from torch.autograd import Variable
lstm

class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="learning argparse")
    parser.add_argument("echo",
                        help="echo the string you use here")  ### add argument method specify command-line option
    parser.add_argument("square", help="display the square of a number", type=int)  ### convert the time to integer
    parser.add_argument('--verbosity', '-v', help='increase output verbosity')
    args = parser.parse_args()  ### return some data from the options field specified
    print(args.echo)
    print(args.square ** 2)
    if args.verbosity:
        print('verbosity turned on')

    # unittest.main()
