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

from models.att import relu, norelu
from models.att_multi import m_relu, m_norelu, m_norelu_not, m_norelu_sa_mult_out, m_relu5, m_norelu_ae_mult_out

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, required=True, help='config file')
parser.add_argument('-g', '--gpu', type=int, default=0, help='gpu number')
args = parser.parse_args()

cf = SourceFileLoader('config', args.config).load_module()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

if not cf.multi_model:
    if cf.model == 'relu':
        model = relu.CrossIntraModel(cf)
    elif cf.model == 'norelu':
        model = norelu.CrossIntraModel(cf)
    else:
        raise ValueError('Invalid model')
else:
    if cf.model == 'relu':
        model = m_relu.CrossIntraModel(cf)
    elif cf.model == 'relu5':
        model = m_relu5.CrossIntraModel(cf)
    elif cf.model == 'norelu':
        model = m_norelu.CrossIntraModel(cf)
    elif cf.model == 'norelu_not':
        model = m_norelu_not.CrossIntraModel(cf)
    elif cf.model == 'norelu_sa_mult_out':
        model = m_norelu_sa_mult_out.CrossIntraModel(cf)
    elif cf.model == 'norelu_ae_mult_out':
        model = m_norelu_ae_mult_out.CrossIntraModel(cf)
    else:
        raise ValueError('Invalid model')

model.train()
