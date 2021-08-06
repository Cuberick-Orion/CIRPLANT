# coding=utf-8
# CIRPLANT, 2021, by Zheyuan (David) Liu, zheyuan.liu@anu.edu.au

# MIT License

# Copyright (c) 2021 Zheyuan (David) Liu

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import, division, print_function

import argparse
import os, sys
import math
import random
import numpy as np
import glob
from pytorch_lightning.core import lightning

import torch
from torch.functional import Tensor
import torch.nn as nn
from torch.nn import functional as F
import torchvision

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from pytorch_lightning.callbacks import ModelCheckpoint

from torch.utils.data import DataLoader
from model import datasets
import logging
from model.utils import utils
from model.utils import lightning_logger

from pprint import pprint as print
import socket
import warnings
import pdb

def args_formatted_to_file(args, ignore_empty=False, sorted_=False, multiline=False):
  '''Format args string as --kwargs foo 
  so that later can be directly copied
  ignore_empty: ignore empty entries
  sorted & multiline: easier to look up
  '''
  args_str = ''
  if not sorted_:
    for ak, av in args.items():
      if ignore_empty and (av == '' or av is None): continue
      _str = '--'+ak + ' ' + str(av) + ' '
      if multiline: _str += '\n'
      args_str += _str
  else:
    for ak, av in sorted(args.items(), key=lambda x: x[0]): 
      if ignore_empty and (av == '' or av is None): continue
      _str = '--'+ak + ' ' + str(av) + ' '
      if multiline: _str += '\n'
      args_str += _str
  return args_str

def init_main(args, checkpoint_monitor=None):
  '''Shared init steps for each main function
  takes care of trainer, logger
  '''
  seed_used = random.randint(1,99999) if args.randseed else args.seed
  pl.seed_everything(seed_used)

  utils.create_dir(args.output)
  # Tensorboard_logger = pl_loggers.TensorBoardLogger(args.output)
  Tensorboard_logger = pl_loggers.TensorBoardLogger(
    args.output, 
    name=socket.gethostname().split('-')[0]
  )
  textlogger_fname = 'log.txt' if not args.testonly else 'log.testonly'
  Text_logger = lightning_logger.Logger(
    os.path.join(Tensorboard_logger.log_dir, textlogger_fname),
    version=Tensorboard_logger.version #- keep the same version
  )

  Text_logger.write('===========')
  Text_logger.write('PLACEHOLDER (Insert manual comments here)')
  Text_logger.write('=====Args commenting======')
  Text_logger.write(str(args.comment))
  Text_logger.write('===========')

  #- Init ModelCheckpoint callback, monitoring loss_name
  if checkpoint_monitor:
    checkpoint_callback = ModelCheckpoint(
      monitor=checkpoint_monitor,
      mode='max', #! Use accuracy -> max
      save_last=True, #* always save the latest
    )
    trainer = pl.Trainer.from_argparse_args(
      args, 
      logger=[Tensorboard_logger, Text_logger], 
      default_root_dir=os.path.join(Tensorboard_logger.log_dir,'checkpoints'), 
      callbacks=[checkpoint_callback],
    )
  else:
    trainer = pl.Trainer.from_argparse_args(
      args, 
      logger=[Tensorboard_logger, Text_logger], 
      default_root_dir=os.path.join(Tensorboard_logger.log_dir,'checkpoints'),
    )

  #* configure logging at the root level of lightning
  logging.getLogger("pytorch_lightning").setLevel(logging.DEBUG)

  Text_logger.write('[INFO] Called with command (copy this for reproduction):')
  Text_logger.write(' '.join(sys.argv));Text_logger.write('\n')
  Text_logger.write('sorted args (complete list):')
  Text_logger.write(args_formatted_to_file(vars(args), sorted_=True, multiline=True), toTerminal=False)
  Text_logger.write('[INFO] random_seed::%i' % seed_used)
  
  return trainer, Tensorboard_logger, Text_logger