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
import os
from importlib.machinery import SourceFileLoader

import numpy as np

from models.att_multi import m_norelu, m_norelu_sa_mult_out


def get_mask():
    """ Loads a pre-computed mask tracking the intrinsic contribution
    of the receptive field of two linear 3x3 convolutional layers
    in order to make them collapse into a single 5x5 layer.

    # Returns:
        A numpy array
    """

    return np.load("resources/deploy_mask.npy")


def get_layer(model, name):
    """ Gets a layer from a model.

    # Arguments:
        model: A keras model
        name: A string, name of a layer in the model

    # Returns:
        A keras layer
    """

    assert name in model.layers
    for l in model.layers:
        if l.name == name:
            return l


def luma_conv_branch(model, cf):
    """ Applies the simplifications proposed in [1], Section IV-B and
    Section IV-D (in case of Scheme 3). A luma convolutional branch composed
    of 2 layers of 3x3 convolutions and without an activation in the middle,
    can be linearly collapsed onto a single layer of 5x5 convolutions by
    tracking the interactions of the global receptive field. More details about
    the treatment of weights and bias can be found in the paper.

    [1] M. G. Blanch, S. Blasi, A. F. Smeaton, N. E. O’Connor and M. Mrak,
    "Attention-Based Neural Networks for Chroma Intra Prediction in Video
    Coding," in IEEE Journal of Selected Topics in Signal Processing,
    vol. 15, no. 2, pp. 366-377, Feb. 2021, doi: 10.1109/JSTSP.2020.3044482.

    # Arguments:
        model: A keras model. Attention-based cross-component
        neural network defined in models/ path.
        cf: Configuration file defining the deployment parameters,
        examples can be found in config/deploy/ path.

    # Returns:
        A dict containing the weights and bias of the boundary layers.
    """

    # Layer 1
    x1 = get_layer(model, "x1")
    W1, b1 = x1.get_weights()
    W1_shape = W1.shape
    W1 = np.reshape(W1, [9, W1_shape[-1]])
    b1 = np.expand_dims(b1, 0)
    W1 = np.concatenate([W1, b1], 0)

    # Layer 2
    x2 = get_layer(model, "x2")
    W2, b_luma = x2.get_weights()
    W2_shape = W2.shape
    W2 = np.reshape(W2, [9, W2_shape[-2], W2_shape[-1]])
    W2 = np.moveaxis(W2, 1, 0)
    W2 = np.reshape(W2, [W2_shape[-2], -1])
    W_luma = np.matmul(W1, W2)
    W_luma = np.reshape(W_luma, [90, -1])
    mask = get_mask()
    W = W_luma[:81]
    W_luma1 = np.matmul(mask.T, W)
    b = np.expand_dims(np.sum(W_luma[81:], axis=0), 0)
    W_luma = np.concatenate([W_luma1, b], 0)

    if cf.scheme == 3:
        scale_bit = (2 ** cf.bit_depth) - 1
        W_luma = np.floor((W_luma1 / scale_bit) * (2 ** cf.scale_luma)).astype('int')
        b_luma = np.floor(b_luma * (2 ** cf.scale_luma)).astype('int')
        W_luma[-1] = W_luma[-1] * scale_bit

    return {'w_luma': W_luma.T,
            'b_luma': b_luma}


def boundary_branch(model, cf):
    """ Deploys the boundary branch of the proposed attention-based cross-component
    neural network and applies the simplifications proposed in [1], Section IV-C
    (in case of Scheme 2) and Section IV-D (in case of Scheme 3).

    [1] M. G. Blanch, S. Blasi, A. F. Smeaton, N. E. O’Connor and M. Mrak,
    "Attention-Based Neural Networks for Chroma Intra Prediction in Video
    Coding," in IEEE Journal of Selected Topics in Signal Processing,
    vol. 15, no. 2, pp. 366-377, Feb. 2021, doi: 10.1109/JSTSP.2020.3044482.

    # Arguments:
        model: A keras model. Attention-based cross-component
        neural network defined in models/ path.
        cf: Configuration file defining the deployment parameters,
        examples can be found in config/deploy/ path.

    # Returns:
        A dict containing the weights and bias of the simplified layer.
    """

    # layer 1
    b1 = get_layer(model, "b1")
    W1, b_bound1 = b1.get_weights()
    W_bound1 = np.zeros([W1.shape[2], W1.shape[1]])
    for i in range(W1.shape[-1]):
        W_bound1[i] = W1[..., i][0]

    # layer 2
    b2 = get_layer(model, "b2")
    W2, b_bound2 = b2.get_weights()
    W_bound2 = np.zeros([W2.shape[2], W2.shape[1]])
    for i in range(W2.shape[-1]):
        W_bound2[i] = W2[..., i][0]

    if cf.scheme == 3:
        scale_bit = (2 ** cf.bit_depth) - 1
        W_bound1 = np.floor((W_bound1 / scale_bit) * (2 ** cf.scale_bound1)).astype('int')
        b_bound1 = np.floor(b_bound1 * (2 ** cf.scale_bound1)).astype('int')

        scale_bound2 = cf.scale_bound1 - cf.shift_bound1
        W_bound2 = np.floor((W_bound2 / (2 ** scale_bound2)) * (2 ** cf.scale_bound2)).astype('int')
        b_bound2 = np.floor(b_bound2 * (2 ** cf.scale_bound2)).astype('int')

    return {'w_boundary1': W_bound1,
            'b_boundary1': b_bound1,
            'w_boundary2': W_bound2,
            'b_boundary2': b_bound2}


def attention_module(model, cf):
    """ Deploys the attention module of the proposed attention-based
    cross-component neural network and applies the simplifications proposed
    in Section IV-D (in case of Scheme 3).

    [1] M. G. Blanch, S. Blasi, A. F. Smeaton, N. E. O’Connor and M. Mrak,
    "Attention-Based Neural Networks for Chroma Intra Prediction in Video
    Coding," in IEEE Journal of Selected Topics in Signal Processing,
    vol. 15, no. 2, pp. 366-377, Feb. 2021, doi: 10.1109/JSTSP.2020.3044482.

    # Arguments:
        model: A keras model. Attention-based cross-component
        neural network defined in models/ path.
        cf: Configuration file defining the deployment parameters,
        examples can be found in config/deploy/ path.

    # Returns:
        A dict containing the weights and bias of the attention module.
    """

    # Att b
    att_b = get_layer(model, "att_b")
    W1, b_att_b = att_b.get_weights()
    W_att_b = np.zeros([W1.shape[2], W1.shape[1]])
    for i in range(W1.shape[-1]):
        W_att_b[i] = W1[..., i][0]

    # Att x
    att_x = get_layer(model, "att_x")
    W1, b_att_x = att_x.get_weights()
    W1 = W1[0]
    W_att_x = np.zeros([W1.shape[2], W1.shape[1]])
    for i in range(W1.shape[-1]):
        W_att_x[i] = W1[..., i][0]

    # Att x1
    att_x1 = get_layer(model, "att_x1")
    W1, b_att_x1 = att_x1.get_weights()
    W1 = W1[0]
    W_att_x1 = np.zeros([W1.shape[2], W1.shape[1]])
    for i in range(W1.shape[-1]):
        W_att_x1[i] = W1[..., i][0]

    if cf.scheme == 3:
        scale_attb = cf.scale_bound2 - cf.shift_bound2
        W_att_b = np.floor((W_att_b / (2 ** scale_attb)) * (2 ** cf.scale_attb)).astype('int')
        b_att_b = np.floor(b_att_b * (2 ** cf.scale_attb)).astype('int')

        scale_attx = cf.scale_luma - cf.shift_luma
        W_att_x = np.floor((W_att_x / (2 ** scale_attx)) * (2 ** cf.scale_attx)).astype('int')
        b_att_x = np.floor(b_att_x * (2 ** cf.scale_attx)).astype('int')

        scale_attx1 = cf.scale_luma - cf.shift_luma
        W_att_x1 = np.floor((W_att_x1 / (2 ** scale_attx1)) * (2 ** cf.scale_attx1)).astype('int')
        b_att_x1 = np.floor(b_att_x1 * (2 ** 23)).astype('int')

    return {'w_att_b': W_att_b,
            'b_att_b': b_att_b,
            'w_att_x': W_att_x,
            'b_att_x': b_att_x.T,
            'w_att_x1': W_att_x1,
            'b_att_x1': b_att_x1}


def prediction_head(model, cf):
    """ Applies the simplifications proposed in [1], Section IV-B and
    Section IV-D (in case of Scheme 3). A luma convolutional branch composed
    of 2 layers of 3x3 and 1x1 convolutions respectively and without an activation
    in the middle, can be linearly collapsed onto a single layer of 3x3 convolution by
    tracking the interactions of the global receptive field. More details about
    the treatment of weights and bias can be found in the paper.

    [1] M. G. Blanch, S. Blasi, A. F. Smeaton, N. E. O’Connor and M. Mrak,
    "Attention-Based Neural Networks for Chroma Intra Prediction in Video
    Coding," in IEEE Journal of Selected Topics in Signal Processing,
    vol. 15, no. 2, pp. 366-377, Feb. 2021, doi: 10.1109/JSTSP.2020.3044482.

    # Arguments:
        model: A keras model. Attention-based cross-component
        neural network defined in models/ path.
        cf: Configuration file defining the deployment parameters,
        examples can be found in config/deploy/ path.

    # Returns:
        A dict containing the weights and bias of the simplified layer.
    """

    # Layer 1
    t2 = get_layer(model, "t2")
    W1, b1 = t2.get_weights()
    W1_shape = W1.shape
    W_trunk = np.reshape(W1, [9, W1_shape[-2], W1_shape[-1]])
    W_trunk = np.moveaxis(W_trunk, 0, 1)
    W_trunk = np.reshape(W_trunk, [W1_shape[-2] * 9, W1_shape[-1]])
    b1 = np.expand_dims(b1, 0)
    W_trunk = np.concatenate([W_trunk, b1], 0)

    # Layer 2
    out = get_layer(model, "out")
    W1, b_out = out.get_weights()
    W1 = W1[0]
    W_out = np.zeros([W1.shape[2], W1.shape[1]])
    for i in range(W1.shape[-1]):
        W_out[i] = W1[..., i][0]
    W_out = np.matmul(W_out, W_trunk.T)

    if cf.scheme == 3:
        scale_bit = (2 ** cf.bit_depth) - 1
        W_out = np.floor((W_out / (2 ** cf.scale_head_in)) * (2 ** cf.scale_head_out) * scale_bit).astype('int')
        b_out = np.floor(b_out * (2 ** cf.scale_head_out) * scale_bit).astype('int')
        W_out[:, -1] = W_out[:, -1] * (2 ** cf.scale_head_in)

    return {'w_out': W_out,
            'b_out': b_out}


def deploy_model(model, cf):
    """Deploys the proposed attention-based cross-component neural network
     and apply corresponding simplifications to reduce complexity.

     # Arguments:
        model: A keras model. Attention-based cross-component
        neural network defined in models/ path.
        cf: Configuration file defining the deployment parameters,
        examples can be found in config/deploy/ path.

    # Returns:
        A dict containing the weights and bias of the deployed model.

    """

    output = luma_conv_branch(model, cf)
    output.update(boundary_branch(model, cf))
    output.update(attention_module(model, cf))
    output.update(prediction_head(model, cf))
    return output


def write_w(file, w):
    """Writes a weights matrix with correct the deployment formatting.

    # Arguments:
        file: A string, output file path.
        w: A numpy array, the weights matrix.
    """

    with open(file, 'w') as output_file:
        for row in w:
            output_file.write('{' + '\t'.join(['%.18f,' % i for i in row])[:-2] + '},\n')


def write_b(file, b):
    """Writes a bias vector with correct the deployment formatting.

    # Arguments:
        file: A string, output file path.
        w: A numpy array, the bias vector.
    """

    with open(file, 'w') as output_file:
        output_file.write('{' + '\t'.join(['%.18f,' % i for i in b])[:-2] + '}\n')


def write_w_int(file, w):
    """Writes an integer weights matrix with correct the deployment formatting.

    # Arguments:
        file: A string, output file path.
        w: A numpy array, the integer weights matrix.
    """

    with open(file, 'w') as output_file:
        for row in w:
            output_file.write('{' + ' '.join(['%d,' % i for i in row]) + '},\n')


def write_b_int(file, b):
    """Writes an integer bias vector with correct the deployment formatting.

    # Arguments:
        file: A string, output file path.
        w: A numpy array, the integer bias vector.
    """

    with open(file, 'w') as output_file:
        output_file.write('{' + ' '.join(['%d,' % i for i in b]) + '}\n')


def write_model(output, cf):
    """Writes the dict result of the deployment process (deploy_model
    function) with correct the deployment formatting.

        # Arguments:
            output: A dict containing all the weights and bias of the model
            cf: Configuration file defining the deployment parameters,
            examples can be found in config/deploy/ path.
        """

    deploy_path = "%s/scheme%d" % (cf.deploy_path, cf.scheme)
    if not os.path.exists(deploy_path):
        os.mkdir(deploy_path)

    if not cf.scheme == 3:
        for name in output:
            name_path = "%s/%s.txt" % (deploy_path, name)
            write_w(name_path, output[name])
            write_b(name_path, output[name])
    else:
        for name in output:
            name_path = "%s/%s.txt" % (deploy_path, name)
            write_w_int(name_path, output[name])
            write_b_int(name_path, output[name])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='deploy config file')
    args = parser.parse_args()

    cf_deploy = SourceFileLoader('config_deploy', args.config).load_module()
    cf = SourceFileLoader('config_model', cf_deploy.config).load_module()

    if not os.path.exists(cf.deploy.deploy_path):
        os.mkdir(cf.deploy.deploy_path)

    if cf_deploy.scheme == 1 or cf_deploy.scheme == 3:
        assert cf.model == "norelu"
        model = m_norelu.CrossIntraModel(cf).model
    elif cf_deploy.scheme == 2:
        assert cf.model == "norelu_sa_mult_out"
        model = m_norelu_sa_mult_out.CrossIntraModel(cf).model
    else:
        raise ValueError('Invalid scheme')

    experiment_path = os.path.join(cf.output_path, cf.model, cf.experiment_name)
    output_path = os.path.join(experiment_path, "multi")
    model.load_weights(output_path + '/weights.hdf5')

    output = deploy_model(model, cf_deploy)
    write_model(output, cf_deploy)
