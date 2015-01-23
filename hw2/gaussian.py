#!/usr/bin/env python
from __future__ import division, print_function

from filters import gaussian_ker

def main():
    for n in [3, 7, 21]:
        print('N = {}, sigma = {}:\n'.format(n, n / 5))
        print(gaussian_ker(n))
        print('\n')

if __name__ == '__main__':
    main()