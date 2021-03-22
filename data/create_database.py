# The copyright in this software is being made available under the BSD
# License, included below. This software may be subject to other third party
# and contributor rights, including patent rights, and no such rights are
# granted under this license.   
#
# Copyright (c) 2021, BBC Research & Development
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#  * Neither the name of the ITU/ISO/IEC nor the names of its contributors may
#    be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.


import argparse
import glob
import os

import cv2
import h5py
import numpy as np
import tqdm

# Script that creates random training and validation blocks of the desired resolution.
# Expecting the DIV2K database. More info: https://data.vision.ee.ethz.ch/cvl/DIV2K/
# Supported block sizes can be customised by modifying the variable block_shapes.

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str, default="/work/marcb/data/DIV2K/.raw")
    parser.add_argument('-o', '--output_path', type=str, default="/work/marcb/data/DIV2K")
    parser.add_argument('-a', '--area_portion', type=float, default=0.5)
    parser.add_argument('-b', '--max_blocks', type=int, default=2000)
    parser.add_argument('-e', '--ext_bound', type=bool, default=False)
    parser.add_argument('-d', '--bit_depth', type=int, default=8)
    parser.add_argument('-nm', '--normalize', type=bool, default=True)
    args = parser.parse_args()

    classes = [0, 0, 1, 2, 3, 4, 5, 6]
    block_shapes = [(4, 4), (8, 8), (16, 16)]
    keys = ['org_chroma', 'rec_boundary', 'rec_luma']

    output_path = args.output_path + '/%.2f-%d-%d' % (args.area_portion, args.max_blocks, int(args.ext_bound))
    if not os.path.exists(args.output_path): os.mkdir(args.output_path)
    if not os.path.exists(output_path): os.mkdir(output_path)

    with open(output_path + '/config.txt', 'w') as logs:
        logs.write('area portion: %.2f\n' % args.area_portion)
        logs.write('max_blocks: %d\n' % args.max_blocks)
        logs.write('ext_bound: %d\n' % int(args.ext_bound))
        logs.write('bit depth: %d\n' % args.bit_depth)
        logs.write('normalize: %d\n' % int(args.normalize))

        for mode in ['train', 'val']:
            print(mode)
            input_path = os.path.join(args.input_path, mode)
            mode_path = os.path.join(output_path, mode)
            if not os.path.exists(mode_path): os.mkdir(mode_path)

            output = {}
            for shape in block_shapes:
                boundary_size = 3 * (sum(shape) + 1) if not args.ext_bound \
                    else 3 * ((2 * shape[0]) + (2 * shape[1]) + 1)
                hf = h5py.File('%s/%dx%d.h5' % (mode_path, shape[0], shape[1]), 'w')
                hf.create_dataset('rec_luma', (1, shape[0], shape[1], 1), maxshape=(None, shape[0], shape[1], 1))
                hf.create_dataset('rec_boundary', (1, boundary_size), maxshape=(None, boundary_size))
                hf.create_dataset('org_chroma', (1, shape[0], shape[1], 2), maxshape=(None, shape[0], shape[1], 2))
                output["%dx%d" % shape] = hf

            blocks = {"%dx%d" % b: 0 for b in block_shapes}
            files = glob.glob(os.path.join(input_path, '0/*.png'))
            files.sort()
            start = int(files[0].split('/')[-1].split('.png')[0])

            for file_id in tqdm.tqdm(range(start, start + len(files))):
                cat = np.random.choice(classes, 1)
                file = "{}/{:d}/{:04d}.png".format(input_path, cat[0], file_id)
                image = cv2.imread(file)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                image = image.astype(np.float32)
                if args.normalize: image /= (2 ** args.bit_depth - 1)
                for shape in block_shapes:
                    portion = min(int((image.shape[0] * image.shape[1] * args.area_portion) /
                                      (shape[0] * shape[1])), args.max_blocks)
                    blocks["%dx%d" % shape] += portion
                    lm = 1 if not args.ext_bound else 2
                    xx, yy = np.mgrid[1:image.shape[0] - lm * shape[0]:shape[0],
                             1:image.shape[1] - lm * shape[1]:shape[1]]
                    positions = np.vstack([xx.ravel(), yy.ravel()]).T
                    positions = positions[np.random.choice(np.arange(len(positions)), portion)]
                    data = output["%dx%d" % shape]
                    info = {'rec_boundary': [], 'rec_luma': [], 'org_chroma': []}
                    for p in positions:
                        if not args.ext_bound:
                            rows = np.append(np.array([p[0] - 1] * (shape[1] + 1)),
                                             np.arange(p[0], p[0] + shape[0]))
                            cols = np.append(np.arange(p[1], p[1] + shape[1]),
                                             np.array([p[1] - 1] * (shape[0] + 1)))
                        else:
                            rows = np.append(np.array([p[0] - 1] * (1 + shape[1] * 2)),
                                             np.arange(p[0], p[0] + shape[0] * 2))
                            cols = np.append(np.arange(p[1], p[1] + shape[1] * 2),
                                             np.array([p[1] - 1] * (1 + shape[0] * 2)))
                        # rec boundaries
                        values = np.append([], [ch for ch in image[rows, cols, :].T])
                        info['rec_boundary'].append(values)
                        # rec luma
                        values = np.expand_dims(image[p[0]:p[0] + shape[0], p[1]:p[1] + shape[1], 0], -1)
                        info['rec_luma'].append(values)
                        # org chroma
                        values = image[p[0]:p[0] + shape[0], p[1]:p[1] + shape[1], 1:]
                        info['org_chroma'].append(values)

                    for key in keys:
                        data[key].resize((data[key].shape[0] + len(positions) - 1), axis=0)
                        data[key][-len(positions):] = info[key]
                        if file_id - start < len(files) - 1:
                            data[key].resize((data[key].shape[0] + 1), axis=0)

            logs.write('---\n%s\n' % mode)
            for shape in block_shapes:
                b = "%dx%d" % shape
                logs.write("%s: %d\n" % (b, blocks[b]))
                output[b].close()
