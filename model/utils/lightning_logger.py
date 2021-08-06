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

import pytorch_lightning as pl
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers.base import rank_zero_experiment

import errno
import sys, os
from pathlib import Path

try:
    from rich import print
    import logging
    from rich.logging import RichHandler
    RICH_PRINT = True    
except:
    import warnings
    warnings.warn("rich not installed, fall back to default printing. \nRecommend to install with: pip install rich", UserWarning)
    RICH_PRINT = False

if RICH_PRINT:
  '''https://rich.readthedocs.io/en/latest/logging.html'''
  FORMAT = "%(message)s"
  logging.basicConfig(
      level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
  )
  log = logging.getLogger("rich")

import pdb

class Logger(LightningLoggerBase):
  '''Customized logger
    if to_file=file_path: output saved to a file
    else: just a fancy print function

    Optional: use Rich print
  '''
  def __init__(self, log_file, version=None):
    super().__init__()
    self.log_file = log_file
    dirname = os.path.dirname(self.log_file)
    self.create_dir(dirname)
    self.exp_version = version

  @rank_zero_only
  def create_dir(self, dirname):
    if not os.path.exists(dirname):
      # os.mkdir(dirname)
      Path(dirname).mkdir(parents=True, exist_ok=True)
    with open(self.log_file, 'w') as f:
      f.write('Creating logger_txt file...\n\n')
    return

  @property
  def name(self):
    return 'Text_logger'

  @property
  @rank_zero_experiment
  def experiment(self):
    # Return the experiment object associated with this logger.
    pass

  @property
  def version(self):
    # Return the experiment version, int or str.
    return self.exp_version
    
  @rank_zero_only
  def log_hyperparams(self, params):
    # params is an argparse.Namespace
    # your code to record hyperparameters goes here
    pass
  
  @rank_zero_only
  def write(self, msg, toTerminal=True):
    with open(self.log_file, 'a') as f:
      f.write(str(msg) + '\n')
    if toTerminal:
      if RICH_PRINT:
        msg = str(msg)
        if "[LOG]" in msg or "Debug" in msg or "DEBUG" in msg:
          msg = msg.replace("[LOG]", "")
          msg = msg.replace("Debug", "")
          msg = msg.replace("DEBUG", "")
          msg = log.debug(msg)
        elif "[INFO]" in msg:
          msg = msg.replace("[INFO]", "")
          log.info(msg)
        elif "[Warning]" in msg or "WARNING" in msg or "Note" in msg or "NOTE" in msg:
          msg = msg.replace("[Warning]", "")
          msg = msg.replace("WARNING", "")
          msg = msg.replace("NOTE", "")
          msg = msg.replace("Note", "")
          log.warning(msg)
        else:
          #- additional character formatting
          msg = msg.replace('::', '[bold magenta]::[/bold magenta]')
          print(msg)

  @rank_zero_only
  def log_metrics(self, metrics, step):
    str_formatted = str(step) +'\t\t'+ '\t'.join([f'{key}:: {value}' for key, value in metrics.items()])
    self.write(msg=str_formatted, toTerminal=False)
