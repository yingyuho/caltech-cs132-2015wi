#!/usr/bin/env python
from __future__ import division, print_function

import argparse

import cv2
import numpy as np
from scipy.ndimage.filters import convolve
from scipy.stats import norm

def hdiff_ker():
    """Returns horizontal gradient kernel [-2, -1, 0, 1, 2]."""
    return np.array([[-2, -1, 0, 1, 2]], dtype=float)

def vdiff_ker():
    """Returns vertical gradient kernel [-2, -1, 0, 1, 2]."""
    return np.array([[-2, -1, 0, 1, 2]], dtype=float).T

def gaussian_ker(window_size, sigma=None):
    """Returns a 2D gaussian kernel."""
    if sigma is None:
        sigma = window_size / 5
    x = np.arange(window_size) - (window_size - 1) / 2
    ker1 = norm.pdf(x, scale=sigma)
    return np.outer(ker1, ker1)

def main():
    parser = argparse.ArgumentParser(
        description='''Various image filters''')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-x', action='store_true', 
        help='compute horizontal gradient')
    group.add_argument('-y', action='store_true', 
        help='compute vertical gradient')
    group.add_argument('-g', type=int, metavar='N', 
        help=('compute Gaussian blur with window size N and sigma N/5; '
            'N must be a positive odd integer'))
    parser.add_argument('-o', dest='outfile', metavar='OUTFILE', 
        help='output image filename; or display on screen if not provided')
    parser.add_argument('infile', help='input image filename')
    args = parser.parse_args()

    # Select appropriate kernel
    if args.x:
        ker = hdiff_ker()
    elif args.y:
        ker = vdiff_ker()
    elif args.g is not None:
        if args.g > 0 and args.g % 2:
            ker = gaussian_ker(args.g)
        else:
            print('N must be a positive odd integer')
            return
    else:
        print('unknown action')

    b = np.sum(np.fmax(0, ker))
    a = np.sum(np.fmin(0, ker))

    # Load image
    image = cv2.imread(args.infile, cv2.CV_LOAD_IMAGE_GRAYSCALE).astype(float)
    # Error when opening failed
    if image is None:
        print('cannot open image {}'.format(args.infile))
        return False

    # Convolve and normalize and cast to 8-bit int
    out = convolve(image, ker)
    out = np.round((out - 255 * a) / (b - a)).astype('uint8')

    # Save image or display on screen
    if args.outfile is not None:
        cv2.imwrite(args.outfile, out)
    else:
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image', out)
        print('Press any key to exit.')
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()