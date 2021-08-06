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
from multiprocessing.sharedctypes import Value
import os, sys
import math

import torch
import torchvision
from model import datasets
from pprint import pprint as print

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--reproduce', action="store_true")

  parser.add_argument('--testonly', action="store_true", help='If True, then skip training, perform testing on test1-split')
  parser.add_argument('--validateonly', action="store_true", help='If True, then skip training, perform validation on val-split')

  '''Pytorch Lightning (essential args to initialize pl/Trainer)
  For details, check documentation at:
  https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html
  '''
  parser.add_argument('--load_from_checkpoint',default=None, help='For testonly load checkpoint, the relative path of a .ckpt')
  parser.add_argument('--resume_from_checkpoint',default=None, help='For continuing training, the relative path of a .ckpt')

  parser.add_argument('--log_by',default=None, help='the val_loss name to save the best checkpoint')

  parser.add_argument('--gpus', type=int, default=1)
  parser.add_argument('--accelerator', type=str, default=None)
  parser.add_argument('--precision', type=int, default=16)
  parser.add_argument('--check_val_every_n_epoch', type=int, default=2)
  parser.add_argument('--num_sanity_val_steps', type=int, default=0)

  parser.add_argument('--max_epochs', type=int, default=300, help='Stop training once this number of epochs is reached')
  
  '''OSCAR args (essential args to initialize the OSCAR model)
  For details, see https://github.com/microsoft/Oscar.
  '''
  # Required parameters
  parser.add_argument("--model_type", default=None, type=str, required=True,
                      help="Model type selected in the list: ")
  parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                      help="Path to pre-trained model or shortcut name selected in the list")
  parser.add_argument("--task_name", default=None, type=str, required=True,
                      help="The name of the task to train selected in the list: ")

  parser.add_argument("--loss_type", default='kl', type=str, help="kl or xe")
  parser.add_argument("--use_layernorm", action='store_true', help="use_layernorm")
  parser.add_argument("--use_label_seq", action='store_true', help="use_label_seq")
  # Other parameters
  parser.add_argument("--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name")
  parser.add_argument("--tokenizer_name", default="", type=str, help="Pretrained tokenizer name or path if not the same as model_name")
  parser.add_argument("--cache_dir", default="", type=str, help="Where do you want to store the pre-trained models downloaded from s3")
  parser.add_argument("--max_seq_length", default=128, type=int,
                      help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
  parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")

  parser.add_argument("--drop_out", default=0.1, type=float, help="Drop out for BERT.")
  parser.add_argument("--classifier", default='linear', type=str, help="linear or mlp")
  parser.add_argument("--cls_hidden_scale", default=2, type=int, help="cls_hidden_scale: for classifier")

  parser.add_argument("--max_img_seq_length", default=30, type=int, help="The maximum total input image sequence length.")
  parser.add_argument("--img_feature_dim", default=2054, type=int, help="The Image Feature Dimension.") # 2054
  parser.add_argument("--img_feature_type", default='res152_w_empty', type=str, help="faster_r-cnn or mask_r-cnn") # faster_r-cnn
  parser.add_argument("--code_voc", default=512, type=int, help="dis_code_voc: 256, 512")
  parser.add_argument("--code_level", default='top', type=str, help="code level: top, botttom, both")

  parser.add_argument("--optim", default='AdamW', type=str, help="optim: AdamW, Adamax")

  parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
  parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.")
  parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                      help="Number of updates steps to accumulate before performing a backward/update pass.")
  
  parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
  parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
  parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
  parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
  
  parser.add_argument('--adamax_lr',default=0.001, type=float) # for Adamax
  parser.add_argument('--sgd_learning_rate', type=float, default=1e-2) # for SGD
  parser.add_argument('--sgd_weight_decay', type=float, default=1e-6) # for SGD
  parser.add_argument('--sgd_learning_rate_decay_frequency', type=int, default=10) # for SGD -> changed, 5000 iteration(approx.)
  
  parser.add_argument("--max_steps__", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
  parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

  '''CIRPLANT args (all other settings not appearing up here)
  '''
  parser.add_argument('--model', type=str, default='CIRPLANT-img') 
  parser.add_argument('--num_hid', type=int, default=512) 

  parser.add_argument('--dataset', type=str, default='cirr', help='Your dataset')
  parser.add_argument('--batch_size', type=int, default=32) 
  parser.add_argument('--num_batches', type=int, default=529, help='Number of batches per epoch, used to configure optimizer warm-up.')

  parser.add_argument('--usefeat', type=str, default='nlvr-resnet152_w_empty', help='The type of image feature to use')
  parser.add_argument('--output', type=str, default='saved_models/exp9999_testing')

  parser.add_argument('--num_workers_per_gpu',default=0, type=int)
  parser.add_argument('--pin_memory',action="store_true", help='Whether pin_memory for dataloaders')

  parser.add_argument('--loss', type=str, default='st', help='Loss used for CIRPLANT, st=soft_triplet')
  
  parser.add_argument('--seed', type=int, default=1111, help='Seed')
  parser.add_argument('--randseed', action="store_true", help='If randomize seed, will overwride seed')
  parser.add_argument('--comment', type=str, default='default comments...', help='Experiment information that will be recorded to the log file')

  args = parser.parse_args()
  args.gradient_clip_val = args.max_grad_norm # auto passed to pl.Trainer
  args.num_train_epochs = args.max_epochs # backwards compatibility
  return args

def load_dataset(args, logger, tokenizer):
  transforms_ = torchvision.transforms.Compose([
          torchvision.transforms.Resize(224),
          torchvision.transforms.CenterCrop(224),
          torchvision.transforms.ToTensor(),
          torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                          [0.229, 0.224, 0.225])
        ])
  if args.dataset == 'cirr':
    if args.testonly or args.validateonly:
      assert not (args.testonly and args.validateonly), ValueError('Only one can be True at a time.')
      split_ = 'test1' if args.testonly else 'val'
      # generating validation sets
      valset_img_txt = datasets.CIRR(args,
        logger,
        path='./data/cirr',
        split=split_,
        val_loader='img+txt',
        transform=transforms_,
        tokenizer=tokenizer)
      valset_img = datasets.CIRR(args,
        logger,
        path='./data/cirr',
        split=split_,
        val_loader='img',
        transform=transforms_,
        tokenizer=tokenizer)
      return [valset_img_txt, valset_img]
    else:
      trainset = datasets.CIRR(args,
        logger,
        path='./data/cirr',
        split='train',
        transform=transforms_,
        tokenizer=tokenizer)
      valset_img_txt = datasets.CIRR(args,
        logger,
        path='./data/cirr',
        split='val',
        val_loader='img+txt',
        transform=transforms_,
        tokenizer=tokenizer)
      valset_img = datasets.CIRR(args,
        logger,
        path='./data/cirr',
        split='val',
        val_loader='img',
        transform=transforms_,
        tokenizer=tokenizer)
      # test1set = datasets.CIRR(args,
      #   logger,
      #   path='./data/cirr',
      #   split='test1',
      #   transform=transforms_,
      #   tokenizer=tokenizer)
      return [trainset, valset_img_txt, valset_img]
  else:
    raise ValueError('Invalid dataset: %s' % args.dataset)

def load_dataloader(args, logger, *dsets):
  loaders = []
  logger.write('\n[INFO] In testonly:: %s\n' % (args.testonly))
  for dset in dsets:
    logger.write('Init dataloader (split->val_loader):: %s -> %s' % (dset.split, dset.val_loader))
    if args.testonly: # force no shuffle and no drop_last
      loaders.append(dset.get_loader(
          batch_size=args.batch_size,
          shuffle=False,
          drop_last=False,
          num_workers=args.num_workers_per_gpu * args.gpus,
          pin_memory=args.pin_memory)
      )
    else:
      if dset.split == 'train':
        train_loader = dset.get_loader(
          batch_size=args.batch_size,
          shuffle=True,
          drop_last=True,
          num_workers=args.num_workers_per_gpu * args.gpus,
          pin_memory=args.pin_memory)
        loaders.append(train_loader)
      else:
        val_loader = dset.get_loader(
          batch_size=args.batch_size,
          shuffle=False,
          drop_last=False,
          num_workers=args.num_workers_per_gpu * args.gpus,
          pin_memory=args.pin_memory)
        loaders.append(val_loader)
  return loaders

def create_model(args, model_id,):
  from model.OSCAR.OSCAR_CIRPLANT import OSCAR_CIRPLANT
  if model_id == 'CIRPLANT-img':
    return OSCAR_CIRPLANT(args,)
  else:
    raise ValueError('Invalid model', model_id)

if __name__ == '__main__':
  args = parse_args()
  from _trainval_base import init_main
  trainer, Tensorboard_logger, Text_logger = init_main(args, checkpoint_monitor=args.log_by)

  torch.multiprocessing.set_sharing_strategy('file_system')
  if args.reproduce:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

  model = create_model(args, args.model)

  dsets = load_dataset(args, Text_logger, model.tokenizer)
  dloaders = load_dataloader(args, Text_logger, *dsets)
  Text_logger.write('[INFO] No. batch in train: %i' % (math.ceil(len(dsets[0])/args.batch_size)))
  
  assert args.optim in ['adamax','sgd', 'AdamW']
  Text_logger.write('[INFO] Optim::%s' % args.optim)
    
  Text_logger.write("\n=== \nFinished loading train/val datasets, entering train/val function\n\n")
  if not (args.testonly or args.validateonly):
    trainer.fit(model, train_dataloader = dloaders[0], val_dataloaders = dloaders[1:])
  else:
    assert args.load_from_checkpoint, ValueError('Must pass in a valid checkpoint path')
    
    model = model.load_from_checkpoint(args.load_from_checkpoint, hparams_file=None) # must manually load weights 
    model.eval()
    model.freeze()
    Text_logger.write("\nloaded checkpoint from %s\n" % args.load_from_checkpoint)
    
    if args.validateonly:
      trainer.validate(model, val_dataloaders = dloaders, ckpt_path=None, verbose=True) # set ckpt_path to None
    else:
      trainer.test(model, test_dataloaders = dloaders, ckpt_path=None, verbose=True) # set ckpt_path to None
