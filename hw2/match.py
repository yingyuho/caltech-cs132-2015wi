#!/usr/bin/env python
"""
Feature matching
"""

from __future__ import division, print_function

import argparse
from itertools import izip

import cv2
import numpy as np

from harris import draw_cross

window_size_ssd = 25
ratio_dist_th = 0.5

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='''Match feature points''')

    parser.add_argument('-w', type=int, 
        default=window_size_ssd, 
        help=('window size for the calculation of SSD'))
    parser.add_argument('-r', type=float, 
        default=ratio_dist_th, 
        help=('threshold for ratio filter'))

    parser.add_argument('-q', action='store_true', 
        help='quite mode; don\'t display result on screen')
    parser.add_argument('-o', dest='outimg', metavar='OUTIMG', 
        help='output image filename')

    parser.add_argument('img1', help='first image')
    parser.add_argument('text1', help='first list of feature points')
    parser.add_argument('img2', help='second image')
    parser.add_argument('text2', help='second list of feature points')

    args = parser.parse_args()

    if args.q and not args.outimg:
        parser.error('OUTIMG or OUTTEXT must be specified in quiet mode')
        return

    # Load image
    images = []
    for path in (args.img1, args.img2):
        images.append(cv2.imread(path, cv2.CV_LOAD_IMAGE_GRAYSCALE))
        if images[-1] is None:
            print('cannot open image {}'.format(args.infile))
            return

    # Pad border with zeros
    pad_lu = args.w // 2
    pad_rd = (args.w - 1) // 2

    for i in (0, 1):
        images[i] = np.pad(images[i], [(pad_lu, pad_rd)] * 2, mode='constant')

    # Load feature points and crop windows
    features = []
    windows = []
    for i, path in enumerate((args.text1, args.text2)):
        features.append(np.loadtxt(path, dtype=int))
        n = features[-1].shape[0]
        windows.append(np.zeros((n, args.w, args.w), dtype=int))
        for j, yx in enumerate(features[-1]):
            y, x = yx
            windows[-1][j] = images[i][y:y+args.w, x:x+args.w]

    # Remove borders and turn to color images
    for i in (0, 1):
        images[i] = (
            np.atleast_3d(images[i][pad_lu:-pad_rd, pad_lu:-pad_rd]) * 
            np.ones((1,1,3), dtype='uint8'))

    # Compute SSD
    m = features[0].shape[0]
    n = features[1].shape[0]

    ssds = np.zeros((m, n), dtype=float)

    for i in xrange(m):
        ssds[i] = np.sum((windows[0][i:i+1] - windows[1]) ** 2, axis=(1, 2))

    ssds[:] /= (args.w ** 2)

    # Find optimal and 2nd optimal matches
    f2 = np.argpartition(ssds, kth=1, axis=1)[:,:2]

    ratio = ssds[xrange(m), f2[:,0]] / ssds[xrange(m), f2[:,1]]

    # Filter by ratio test
    r_test = (ratio < args.r)
    match_idxs = f2[:,0][r_test]

    # Extract matching feature point pairs
    match_locs = [features[0][r_test], features[1][match_idxs]]

    # Draw match lines
    red = (0, 0, 255)
    radius = 5
    for yx1, yx2 in izip(*match_locs):
        y1, x1 = yx1
        y2, x2 = yx2
        cv2.line(images[0], (x1, y1), (x2, y2), red)
        draw_cross(images[0], yx1, radius)
        cv2.circle(images[0], (x2, y2), radius, red)

    # Save image
    if args.outimg:
        cv2.imwrite(args.outimg, images[0])
    # Display
    if not args.q:
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image', images[0])
        print('Press any key to exit.')
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()