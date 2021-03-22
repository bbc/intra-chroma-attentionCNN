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

import os
import sys

import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.layers import Conv2D, Conv1D
from keras.layers import Input, Lambda, ZeroPadding2D
from keras.layers import multiply, LeakyReLU
from keras.models import Model
from keras.optimizers import Adam

sys.path.append('../data')
sys.path.append('../models')

from data_generator import DIV2KDatasetRegMultiple as Database


# Scheme 2 architecture without sparsity


class CrossIntraModel:
    def __init__(self, cf):
        self._cf = cf
        self.name = "multi"

        l_input = Input((None, None, 1), name='l_input')
        b_input = Input((None, 3), name='b_input')
        y_input = Input((None, None, 2), name='labels')

        ae_act = LeakyReLU(alpha=0.2)
        b_w = Conv1D(self._cf.bb, kernel_size=1, strides=1, padding='same', activation=ae_act, name='b_w')(b_input)
        b_enc = Conv1D(3, kernel_size=1, strides=1, padding='same', activation=ae_act, name='b_enc')(b_w)

        att_b = Conv1D(self._cf.att_h, kernel_size=1, strides=1, padding='same', activation='relu', name='att_b')(b_w)

        # Luma branch
        x = ZeroPadding2D((2, 2))(l_input)
        x = Conv2D(self._cf.lb1, kernel_size=3, strides=1, padding='valid', activation=None, name='x1')(x)
        x = Conv2D(self._cf.lb2, kernel_size=3, strides=1, padding='valid', activation='relu', name='x2')(x)
        att_x = Conv2D(self._cf.att_h, kernel_size=1, strides=1, padding='same', activation='relu', name='att_x')(x)
        att_x1 = Conv2D(3, kernel_size=1, strides=1, padding='same', activation='relu', name='att_x1')(x)

        # Attention module
        t = self.attentive_join(att_x, att_b, b_enc, att_x1)
        t = Conv2D(3, kernel_size=3, strides=1, padding='same', activation=None, name='t2')(t)
        output = Conv2D(2, kernel_size=1, strides=1, padding='same', activation='relu', name='out')(t)
        self.model = Model([l_input, b_input, y_input], output)

        loss_mse = tf.losses.mean_squared_error(255 * y_input, 255 * output)

        # Total loss
        total_loss = self._cf.weight_mse * loss_mse

        optimizer = Adam(self._cf.lr, self._cf.beta)
        self.model.add_loss(total_loss)
        self.model.compile(loss=None, optimizer=optimizer)
        self.model.metrics_tensors.append(loss_mse)
        self.model.metrics_names.append("loss_mse")

    def attentive_join(self, att_x, att_b, b_enc, att_x1):
        def get_att(inputs):
            f1, f2 = inputs  # att_x [bs, N, N, h], att_b [bs, b, h]
            f1 = K.reshape(f1, shape=[K.shape(f1)[0], K.shape(f1)[1] * K.shape(f1)[2], K.shape(f1)[-1]])
            y = tf.matmul(f1, f2, transpose_b=True)
            return K.softmax(y / self._cf.temperature, axis=-1)

        def apply_att(inputs):
            f1, f2, f3 = inputs  # att [bs, NxN, b], b_enc [bs, b, D], att_x [bs, N, N, h]
            y = K.batch_dot(f1, f2)  # [bs, NxN, D]
            return K.reshape(y, shape=[K.shape(f3)[0], K.shape(f3)[1], K.shape(f3)[2], 3])

        att = Lambda(get_att, name='att')([att_x, att_b])
        b_out = Lambda(apply_att, name='b_masked')([att, b_enc, att_x])
        return multiply([att_x1, b_out])

    def train(self):
        print("Training model: %s" % self.name)

        model_path = os.path.join(self._cf.output_path, self._cf.model)
        experiment_path = os.path.join(model_path, self._cf.experiment_name)
        output_path = os.path.join(experiment_path, self.name)
        if not os.path.exists(self._cf.output_path): os.mkdir(self._cf.output_path)
        if not os.path.exists(model_path): os.mkdir(model_path)
        if not os.path.exists(experiment_path): os.mkdir(experiment_path)
        if not os.path.exists(output_path): os.mkdir(output_path)

        train_data = Database(data_path=self._cf.data_path,
                              block_shape=self._cf.block_shape,
                              mode='train',
                              batch_size=self._cf.batch_size,
                              shuffle=self._cf.shuffle,
                              get_vol=True,
                              seed=42)
        val_data = Database(data_path=self._cf.data_path,
                            block_shape=self._cf.block_shape,
                            mode='val',
                            batch_size=self._cf.batch_size,
                            shuffle=False,
                            get_vol=True,
                            seed=42) if self._cf.validate else None

        checkpoint = ModelCheckpoint(output_path + "/weights.hdf5",
                                     monitor='val_loss', verbose=0, mode='min',
                                     save_best_only=True, save_weights_only=True)

        early_stop = EarlyStopping(monitor='val_loss', mode="min", patience=self._cf.es_patience)
        tensorboard = TensorBoard(log_dir=output_path)
        callbacks_list = [checkpoint, early_stop, tensorboard]

        nb_block_shapes = len(self._cf.block_shape)
        validation_steps = (val_data.samples // self._cf.batch_size) * nb_block_shapes if self._cf.validate else None

        self.model.fit_generator(generator=train_data,
                                 steps_per_epoch=(train_data.samples // self._cf.batch_size) * nb_block_shapes,
                                 epochs=self._cf.epochs,
                                 validation_data=val_data,
                                 validation_steps=validation_steps,
                                 callbacks=callbacks_list,
                                 max_queue_size=10,
                                 workers=10,
                                 use_multiprocessing=self._cf.use_multiprocessing)
