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

from builtins import RuntimeWarning
from builtins import NotImplementedError
import os, sys
import numpy as np
import PIL
import torch
import json
import torch.utils.data
import torchvision
import warnings
import random
import pickle

from oscar.utils.task_utils import _truncate_seq_pair
from pprint import pprint as print

PIL.Image.MAX_IMAGE_PIXELS = 1000000000
PIL.Image.warnings.simplefilter('error', PIL.Image.DecompressionBombWarning)

# For Aux Annotation in CIRR
WORD_REPLACE = {
  '[c] None existed': 'noneexisted',
  '[cr0] Nothing worth mentioning': 'nothingworth',
  '[cr1] Covered in query': 'coveredinquery',
}

class BaseDataset(torch.utils.data.Dataset):
  """Base class for a dataset.
  This portion is based on the TIRG implementation,
  see https://github.com/google/tirg.
  """

  def __init__(self, logger):
    super(BaseDataset, self).__init__()
    self.imgs = []
    self.test_queries = []

    self.logger = logger
    self.logger.write(''); self.logger.write('Start init BaseDataset class...')
  
  def get_loader(self,
                 batch_size,
                 shuffle=False,
                 drop_last=False,
                 num_workers=0,
                 pin_memory=False):
    self.logger.write('\nNum_worker: %i, pin_memory: %s' % (num_workers, str(pin_memory)))
    return torch.utils.data.DataLoader(
        self,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=lambda i: i)

  def get_all_texts(self):
    raise NotImplementedError
  
  def get_test_queries(self):
    return self.test_queries if self.split != 'train' else self.train_queries

  def generate_random_query_target(self):
    raise NotImplementedError
  
  def get_img(self, idx, raw_img=False):
    raise NotImplementedError

  def oscar_tensorize(self,
                  text_a, # main text
                  text_b,
                  img_feat, # replaces img_key (directly get the feat, no need to fetch by key)
                  cls_token_at_end=False, pad_on_left=False,
                  cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                  sequence_a_segment_id=0, sequence_b_segment_id=1,
                  cls_token_segment_id=1, pad_token_segment_id=0,
                  mask_padding_with_zero=True):
      """ Repurposed from oscar -> run_nlvr -> dataset
      see https://github.com/microsoft/Oscar
      """

      tokens_a = self.tokenizer.tokenize(text_a)

      tokens_b = None
      if text_b:
        text_b = text_b['left'] + ' ' + text_b['right']
        tokens_b = self.tokenizer.tokenize(text_b)
        # Modifies `tokens_a` and `tokens_b` in place so that the total length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, self.args.max_seq_length - 3)
      else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > self.args.max_seq_length - 2:
          tokens_a = tokens_a[:(self.args.max_seq_length - 2)]

      tokens = tokens_a + [sep_token]
      segment_ids = [sequence_a_segment_id] * len(tokens)

      if tokens_b:
        tokens += tokens_b + [sep_token]
        segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

      if cls_token_at_end:
        tokens = tokens + [cls_token]
        segment_ids = segment_ids + [cls_token_segment_id]
      else:
        tokens = [cls_token] + tokens
        segment_ids = [cls_token_segment_id] + segment_ids

      input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

      # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
      input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

      # Zero-pad up to the sequence length.
      padding_length = self.args.max_seq_length - len(input_ids)
      if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
        segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
      else:
        input_ids = input_ids + ([pad_token] * padding_length)
        input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

      assert len(input_ids) == self.args.max_seq_length
      assert len(input_mask) == self.args.max_seq_length
      assert len(segment_ids) == self.args.max_seq_length

      # image features
      if img_feat.shape[0] > self.args.max_img_seq_length:
        warnings.warn("[Warning] Truncating image_sequence...", RuntimeWarning)
        img_feat = img_feat[0:self.args.max_img_seq_length, ]
        if self.args.max_img_seq_length > 0:
          input_mask = input_mask + [1 if mask_padding_with_zero else 0] * img_feat.shape[0]
          # segment_ids += [sequence_b_segment_id] * img_feat.shape[0]
      else:
        if self.args.max_img_seq_length > 0:
          input_mask = input_mask + [1 if mask_padding_with_zero else 0] * img_feat.shape[0]
          # segment_ids = segment_ids + [sequence_b_segment_id] * img_feat.shape[0]
        padding_matrix = torch.zeros((self.args.max_img_seq_length - img_feat.shape[0], img_feat.shape[1]))
        img_feat = torch.cat((img_feat, padding_matrix), 0)
        if self.args.max_img_seq_length > 0:
          input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_matrix.shape[0])
          # segment_ids = segment_ids + [pad_token_segment_id] * padding_matrix.shape[0]

      return (torch.tensor(input_ids, dtype=torch.long), # input_ids
              torch.tensor(input_mask, dtype=torch.long), # input_mask
              torch.tensor(segment_ids, dtype=torch.long), # segemnt_ids
              None, # label_id
              img_feat, # img_feat
              None, # entry_id
      )

  def __getitem__(self, idx):
    return self.generate_random_query_target()


class CIRR(BaseDataset):
  """ The CIRR dataset.
  This is partially based on TIRG implementation and Fashion-IQ implementation
  see https://github.com/google/tirg, https://github.com/XiaoxiaoGuo/fashion-iq.
  """
  def __init__(self, args, logger, path, split='train', val_loader=None, transform=None, tokenizer=None):
    super(CIRR, self).__init__(logger)
    '''stable dataset version, DO NOT CHANGE unless you are sure
    corresponding dataset version can be found in our repository
    https://github.com/Cuberick-Orion/CIRR/tree/cirr_dataset
    '''
    self.version = 'rc2' 

    self.args = args
    
    assert split in ['train', 'val', 'test1']
    self.split = split

    self.usefeat = self.args.usefeat.split(',')

    assert val_loader in [None, 'img+txt', 'img'] # if None, then proceed as original, if otherwise, return will be different
    self.val_loader = val_loader
    
    self.transform = transform
    self.tokenizer = tokenizer

    self.img_path = path
    data = { # hold all data read from json files
        'image_splits': {},
        'captions': {},
        'captions_ext': {} 
    }

    for subfolder_name in data: # load the corresponding json files
      for json_name in os.listdir(path + '/' + subfolder_name):
        if (split == 'train' and 'train' in json_name) \
        or (split == 'val' and 'val' in json_name) \
        or (split == 'test1' and 'test' in json_name):
          logger.write('[INFO] adding json %s' % json_name)
          json_load = json.load(open(path + '/' + subfolder_name + '/' + json_name))
          data[subfolder_name][json_name] = json_load
    
    imgs = []
    asin2id = {}; id2asin=[]
    for json_name in data['image_splits']:
      for asin_,img_path_ in data['image_splits'][json_name].items():
        asin2id[asin_] = len(imgs)
        id2asin.append(asin_)
        imgs += [{
          'asin': asin_,
          'img_feat_res152_path': os.path.join(self.img_path, 'img_feat_res152', img_path_.replace('.png','.pkl')),
          'captions': [asin2id[asin_]],
          # 'img_raw_path': os.path.join(self.img_path, 'img_raw_filtered', img_path_), #! Uncomment this line if raw img is downloaded
        }]

    # process queries from loaded data
    queries = []
    for json_name in data['captions']:
      for query in data['captions'][json_name]:
        if self.split != 'test1':
          query['source_id'] = asin2id[query['reference']]
          query['target_id'] = asin2id[query['target_hard']]
          query['captions'] = [query['caption']]
          query['target_soft_id'] = {asin2id[kkk]:vvv for kkk,vvv in query['target_soft'].items()}
          queries += [query]
        else:
          query['source_id'] = asin2id[query['reference']]
          query['captions'] = [query['caption']]
          queries += [query]

    # add Aux Annoation from cap.ext
    queries_temp = {qqq['pairid']:qqq for qqq in queries} 
    for kkk,qqq in queries_temp.items():
      queries_temp[kkk]['caption_extend'] = None
    for json_name in data['captions_ext']:
      for query in data['captions_ext'][json_name]:
        query_cap_ext_ = {}
        for kkk_,vvv_ in query['caption_extend'].items():
          if vvv_ in WORD_REPLACE.keys():
            query_cap_ext_[kkk_] = WORD_REPLACE[vvv_]
          else:
            query_cap_ext_[kkk_] = vvv_
        queries_temp[query['pairid']]['caption_extend'] = query_cap_ext_
    queries = [qqq for kkk,qqq in queries_temp.items()]
   
    self.data = data
    self.imgs = imgs
    self.asin2id = asin2id; self.id2asin = id2asin
    self.queries = queries

    # prepare a copy of test_queries from queries
    if split in ['train', 'val']:
      self.test_queries = [{
        'source_img_id': query['source_id'],
        'target_img_id': query['target_id'],
        'source_caption': query['source_id'],
        'target_caption': query['target_id'],
        'target_caption_soft': query['target_soft_id'],
        'set_member_idx': [self.asin2id[ii] for ii in query['img_set']['members'] if ii != query['reference']],
        'mod': {'str': query['captions'][0], **query['caption_extend']},
        'caption_ext': query['caption_extend'],
        'pairid': query['pairid']
      } for _, query in enumerate(queries)]
    elif split == 'test1': 
      self.test_queries = [{
        'source_img_id': query['source_id'],
        'source_caption': query['source_id'],
        'set_member_idx': [self.asin2id[ii] for ii in query['img_set']['members'] if ii != query['reference']],
        'mod': {'str': query['captions'][0], **query['caption_extend']},
        'caption_ext': query['caption_extend'],
        'pairid': query['pairid']
      } for _, query in enumerate(queries)]
    
    self.logger.write('init CIRR_%s -> %s -> %s, usefeat::%s ...' % (self.version, self.split, 'None', str(self.usefeat)))
    self.logger.write('\t total number of imgs:: %i' % (len(self.imgs)))
    self.logger.write('\t total number of pairs:: %i' % (len(self.queries)))

  def __len__(self):
    if self.split == 'train' and not self.val_loader: # in training
      return len(self.imgs)
    else: # in validation/test
      if self.val_loader == 'img+txt':
        return len(self.test_queries)
      elif self.val_loader == 'img':
        return len(self.imgs)

  def __getitem__(self, idx):
    if self.split == 'train' and self.val_loader is None:
      generated_ = self.generate_random_query_target()
      if self.tokenizer:
        oscar_input = self.oscar_tensorize(text_a=generated_['mod']['str'],
                  text_b=None,
                  img_feat=generated_['source_img_data'][0],
                  cls_token_at_end=bool(self.args.model_type in ['xlnet']), # is 'bert' # xlnet has a cls token at the end
                  cls_token=self.tokenizer.cls_token, #- '[CLS]'
                  sep_token=self.tokenizer.sep_token,
                  cls_token_segment_id=2 if self.args.model_type in ['xlnet'] else 0,
                  pad_on_left=bool(self.args.model_type in ['xlnet']), # pad on the left for xlnet
                  pad_token_segment_id=4 if self.args.model_type in ['xlnet'] else 0)  
        return (generated_, oscar_input) # add the oscar input at the end
      else:
        return generated_
    else:
      if self.val_loader == 'img+txt':
        generated_ = self.val_get_img_txt(idx,)
        if self.tokenizer:
          oscar_input = self.oscar_tensorize(text_a=generated_[1],
                  text_b=None,
                  img_feat=generated_[0],
                  cls_token_at_end=bool(self.args.model_type in ['xlnet']), # is 'bert' # xlnet has a cls token at the end
                  cls_token=self.tokenizer.cls_token, # '[CLS]'
                  sep_token=self.tokenizer.sep_token,
                  cls_token_segment_id=2 if self.args.model_type in ['xlnet'] else 0,
                  pad_on_left=bool(self.args.model_type in ['xlnet']), # pad on the left for xlnet
                  pad_token_segment_id=4 if self.args.model_type in ['xlnet'] else 0)
          return (generated_, oscar_input) # add the oscar input at the end
        else:
          return generated_
      elif self.val_loader == 'img':
        return self.val_get_img(idx,)

  def get_imgs_in_set(self, set_member_idx):
    if not set_member_idx is None:
      img_feats = []
      for feat_ in self.usefeat:
        img_feat_ = np.stack([self.get_img(d, usefeat=feat_) for d in set_member_idx])
        img_feat_ = torch.from_numpy(img_feat_).float()
        img_feats.append(img_feat_)
      return img_feats
    else:
      return None
  
  def generate_random_query_target(self):
    query_idx = random.choice(range(len(self.queries)))
    query = self.queries[query_idx]
    
    mod_str = query['captions'][0]
    mod_str_ext = query['caption_extend'] # The Aux Annotation
    
    other_set_member_asin = [ii for ii in query['img_set']['members'] if ii != query['reference']]
    other_set_member_idx = [self.asin2id[ii] for ii in other_set_member_asin]

    target_soft_within_set = {kkk:vvv for kkk,vvv in query['target_soft'].items() if kkk in query['img_set']['members'] and kkk != query['reference']} # only consider things in set, filter out target == reference image which is just mistakes
    
    random_stageI_idx = None

    return {
      'source_img_id': query['source_id'],
      'target_img_id': query['target_id'],
      'source_img_data': [self.get_img(query['source_id'], usefeat=i) for i in self.usefeat],
      'target_img_data': [self.get_img(query['target_id'], usefeat=i) for i in self.usefeat],
      'mod': {'str': mod_str, **mod_str_ext},
      'target_caption': query['target_id'],
      
      'pair_id': query['pairid'],
      'target_idx_soft': query['target_soft_id'],
      'set_id': query['img_set']['id'],
      'set_member_idx': other_set_member_idx, # excluding reference itself
      'set_member_data': self.get_imgs_in_set(other_set_member_idx), # excluding reference itself
      'set_target_soft': {other_set_member_asin.index(kkk):vvv for kkk,vvv in target_soft_within_set.items()},
    }

  def val_get_img_txt(self, idx,):
    """PyTorch Lightning adaptaion
    return img+text by sequence
    indexed by testset.test_queries
    """
    assert len(self.usefeat) == 1, ValueError('Do not support multiple features as input.')

    return (
      self.get_img(self.test_queries[idx]['source_img_id'], usefeat=self.usefeat[0]),
      self.test_queries[idx]['mod']['str'], 
      self.test_queries[idx]['mod']['0'],
      self.test_queries[idx]['mod']['1'],
      self.test_queries[idx]['mod']['2'],
      self.test_queries[idx]['mod']['3'],
    )

  def val_get_img(self, idx,):
    """PyTorch Lightning adaptaion
    return img by sequence
    indexed by testset.imgs
    """
    return (
      self.get_img(idx, usefeat=self.usefeat[0]),
    )
  
  def get_img(self, idx, raw_img=False, usefeat='resnet'):
    if usefeat == 'resnet':
      img_path = self.imgs[idx]['img_raw_path']
      with open(img_path, 'rb') as f:
        img = PIL.Image.open(f)
        img = img.convert('RGB')
      if raw_img:
        return img
      if self.transform:
        img = self.transform(img)
      return img
    elif usefeat == 'nlvr-resnet152_w_empty':
      """Append empty 6-digit positional encoding
      For OSCAR, unsqueeze the 0-dim
      """
      img_path = self.imgs[idx]['img_feat_res152_path']
      try:
        _img_feat = pickle.load(open(img_path,'rb')) # torch.Size([2048])
      except:
        _img_feat = np.zeros(2048)
      _feat = torch.from_numpy(_img_feat).float()
      return torch.unsqueeze(torch.cat((_feat, torch.zeros(6)), dim=0), 0) # pad 0 for bbox coordinates
    else:
      ValueError("Unsupported img_feat:: %s" % str(usefeat))