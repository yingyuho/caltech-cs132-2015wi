#!/usr/bin/env python
"""
Harris corner feature detector
"""

from __future__ import division, print_function

import argparse

import cv2
import numpy as np
from scipy.ndimage.filters import convolve

from filters import hdiff_ker, vdiff_ker, gaussian_ker

k_harris = 0.19
window_size_h = 11
cut_off_threshold_f = 0.10
window_size_max_search = 21
max_detect_num = 400

def f_score(image, window_size_h=window_size_h, k=k_harris):
    """Computes Harris corner detector."""
    gx = convolve(image, hdiff_ker())
    gy = convolve(image, vdiff_ker())
    g_ker = gaussian_ker(window_size_h)
    hxx = convolve(gx ** 2, g_ker)
    hyy = convolve(gy ** 2, g_ker)
    hxy = convolve(gx * gy, g_ker)
    score = (hxx * hyy - hxy ** 2) - (k / 4) * (hxx + hyy) ** 2
    return score

def draw_cross(image, loc, size=5):
    """Draws a red cross."""
    y, x = loc
    red = (0, 0, 255)
    cv2.line(image, (x, y - size), (x, y + size), red)
    cv2.line(image, (x - size, y), (x + size, y), red)

def dilate1d(image, size, axis):
    """Dilates an image along one axis."""
    roll_back = size // 2
    n = size - 1

    # Get optimal way of dilation
    roll_queue = []
    i = 1
    while n > 0:
        roll_queue.append(min(n, i))
        n -= roll_queue[-1]
        i *= 2

    tmp = image.copy()

    for shift in roll_queue:
        np.fmax(tmp, np.roll(tmp, shift, axis), tmp)

    return np.roll(tmp, -roll_back, axis)

def dilate(image, size):
    """Dilates an image along multiple axes."""
    tmp = image.copy()
    for a, s in enumerate(size):
        tmp = dilate1d(tmp, s, a)
    return tmp

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='''Harris corner feature detector''')

    parser.add_argument('-w', type=int, 
        default=window_size_h, 
        help=('window size for the calculation of H'))
    parser.add_argument('-f', type=float, 
        default=cut_off_threshold_f, 
        help=('cut-off threshold for f'))
    parser.add_argument('-s', type=int, 
        default=window_size_max_search, 
        help=('window size for the local maximum search'))
    parser.add_argument('-n', type=int, 
        default=max_detect_num, 
        help=('maximal number of detections'))
    parser.add_argument('-k', type=float, 
        default=k_harris, 
        help=('parameter k (0.1 - 0.8) for Harris detector'))
    parser.add_argument('-q', action='store_true', 
        help='don\'t display result on screen')
    parser.add_argument('-o', dest='outimg', metavar='OUTIMG', 
        help='filename to save feature image')
    parser.add_argument('-t', dest='outtext', metavar='OUTTEXT', 
        help='filename to save list of feature points')
    parser.add_argument('infile', help='input image filename')
    args = parser.parse_args()

    if args.q and not args.outimg and not args.outtext:
        parser.error('OUTIMG or OUTTEXT must be specified in quiet mode')
        return

    # Load image
    image = cv2.imread(args.infile, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    # Error when opening failed
    if image is None:
        print('cannot open image {}'.format(args.infile))
        return

    # Get Harris score
    f = f_score(image.astype(float), args.w, args.k)

    # Cut-off threshold
    b = np.max(f)
    a = np.min(f)
    cut_off = a + (b - a) * args.f

    # Find local maxima by comparing with dilated image
    max_val = dilate(f, [args.s] * 2)
    max_loc = np.logical_and(f >= cut_off, f >= max_val)

    # Sort detections
    dtype = [('loc', '2int32'), ('score', 'float64')]
    detection = np.empty(max_loc.sum(), dtype=dtype)
    detection['loc'] = np.argwhere(max_loc)
    detection['score'] = f[list(detection['loc'].T)]
    detection.sort(order='score')

    # Plot feature points
    out = np.atleast_3d(image) * np.ones((1,1,3), dtype='uint8')
    cross_loc = detection['loc'][::-1][:args.n]
    for loc in cross_loc:
        draw_cross(out, loc)

    if not args.q:
        print('Drew {} features.'.format(len(cross_loc)))

    # Save image
    if args.outimg:
        cv2.imwrite(args.outimg, out)
    # Save location
    if args.outtext:
        np.savetxt(args.outtext, cross_loc, fmt='%d')
    # Display
    if not args.q:
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image', out)
        print('Press any key to exit.')
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()