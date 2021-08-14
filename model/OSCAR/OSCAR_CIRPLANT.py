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

from multiprocessing.sharedctypes import Value
import os,sys
sys.path.insert(0, '../')

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from .torch_functions import TripletLoss
import pytorch_lightning as pl
import warnings
from pprint import pprint

from transformers.pytorch_transformers import (BertConfig, BertTokenizer)
from transformers.pytorch_transformers import AdamW, WarmupLinearSchedule

from .modeling.modeling_bert import ImageBertForImageFeature

import random

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig,)), ())

MODEL_CLASSES = {
  'bert': (BertConfig, ImageBertForImageFeature, BertTokenizer),
}

class NormalizationLayer(torch.nn.Module):
  """Class for normalization layer.
  """
  def __init__(self, 
              normalize_scale=1.0, learn_scale=True):
    super(NormalizationLayer, self).__init__()
    self.norm_s = float(normalize_scale)
    if learn_scale:
      self.norm_s = torch.nn.Parameter(torch.FloatTensor((self.norm_s,)))

  def forward(self, x):
    features = self.norm_s * x / torch.norm(x, dim=-1, keepdim=True).expand_as(x)
    return features

class OSCAR_CIRPLANT(pl.LightningModule):
  """Main model class.
  coded following the Pytorch Lightning framework, i.e., this module should be self-contained.
  See https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html.
  """
  def __init__(self, args, **kwargs):
    self.save_hyperparameters()
    self.args = args
    super(OSCAR_CIRPLANT, self).__init__()

    self.tokenizer, self.imageBert = self.init_oscar(args,)

    self.normalization_layer = NormalizationLayer(
        normalize_scale=4.0, learn_scale=True)
    self.soft_triplet_loss = TripletLoss(metric='pdist')

    '''supports passing additional args at testing
    this is used when calling trainer.test()
    since the loading of a checkpoint will overwrite the args loaded above this line
    '''
    if kwargs:
      print("Additional kwargs found...\n")
      for kw, kwv in kwargs.items():
        args.kw = kwv; print("\t setting self.%s:: %s" % (kw, str(kwv)))
        setattr(self, kw, kwv)

  def init_oscar(self, args,):
    '''Initialize OSCAR transformer
    and load pre-trained weights.
    '''
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    model_class = ImageBertForImageFeature

    if not os.path.exists(args.model_name_or_path):
      config = config_class.from_pretrained('data/Oscar_pretrained_models/base-vg-labels/ep_107_1192087', num_labels=2, finetuning_task=args.task_name)
      tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else 'data/Oscar_pretrained_models/base-vg-labels/ep_107_1192087', do_lower_case=args.do_lower_case)
    else:
      config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, num_labels=2, finetuning_task=args.task_name)
      tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
    
    config.img_feature_dim = args.img_feature_dim
    config.img_feature_type = args.img_feature_type
    config.code_voc = args.code_voc
    config.hidden_dropout_prob = args.drop_out
    config.loss_type = args.loss_type
    config.use_layernorm = args.use_layernorm
    self.config = config

    if not os.path.exists(args.model_name_or_path):
      imgBert_model = model_class.from_pretrained('data/Oscar_pretrained_models/base-vg-labels/ep_107_1192087', from_tf=bool('.ckpt' in 'data/Oscar_pretrained_models/base-vg-labels/ep_107_1192087'), config=config)
    else:
      imgBert_model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
    
    return tokenizer, imgBert_model
    
  def format_data(self, data_in):
    if self.args.dataset == 'cirr':
      imgs_target = torch.stack([d['target_img_data'][0] for d in data_in])

      mods_ext_1 = [str(d['mod']['0']) for d in data_in]
      mods_ext_2 = [str(d['mod']['1']) for d in data_in]
      mods_ext_3 = [str(d['mod']['2']) for d in data_in]
      mods_ext_4 = [str(d['mod']['3']) for d in data_in]

      imgs_in_set = torch.stack([d['set_member_data'][0] for d in data_in])
      target_soft_idx = [d['set_target_soft'] for d in data_in]
      
      return (mods_ext_1, mods_ext_2, mods_ext_3, mods_ext_4,
              imgs_target, imgs_in_set, target_soft_idx)
    else:
      raise ValueError('Unsupported dataset.')

  def compose_img_txt(self, oscar_in):
    def format_oscar_in(oscar_in):
      '''additional layer to dictionarize the bert input from dataset output
      '''
      return {'input_ids':      torch.stack([ii[0] for ii in oscar_in], dim=0),
              'attention_mask': torch.stack([ii[1] for ii in oscar_in], dim=0),
              'token_type_ids': torch.stack([ii[2] for ii in oscar_in], dim=0) if self.args.model_type in ['bert', 'xlnet'] else None,
              'labels':         None,
              'img_feats':      None if self.args.img_feature_dim == -1 else torch.stack([ii[4] for ii in oscar_in], dim=0)}

    oscar_in = format_oscar_in(oscar_in)
    sequence_output, pooled_output = self.imageBert(**oscar_in)
    mod_img1 = sequence_output[:,-1,:]
    return self.normalization_layer(mod_img1)
  
  def compose_candidate_img(self, candidate_img_feat):
    img2_or_3 = self.imageBert.img_feat_forward(candidate_img_feat)
    img2_or_3 = img2_or_3.squeeze()
    return self.normalization_layer(img2_or_3)

  def training_step(self, batch, batch_idx):
    '''Main training loop
    '''
    def logging_step():
      self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

      for idg, param_group in enumerate(self.trainer.optimizers[0].param_groups):
        self.log('learning_rate_group'+str(idg), param_group['lr'], on_step=True, on_epoch=False, prog_bar=False, logger=True)
      return 

    data_in = [d1[0] for d1 in batch]
    oscar_in = [d1[1] for d1 in batch]

    (mods_ext_1, mods_ext_2, mods_ext_3, mods_ext_4, # Aux Annotaions are not used
    imgs_target, imgs_in_set, target_soft_idx) = self.format_data(data_in)
    
    mod_img1 = self.compose_img_txt(oscar_in)
    img2 = self.compose_candidate_img(imgs_target)
    img3 = self.compose_candidate_img(imgs_in_set)

    if self.args.loss == 'st':
      loss = self.compute_soft_triplet_loss(mod_img1, img2, img3, target_soft_idx).cuda()
    else:
      raise ValueError("Unsupported loss: %s." % self.args.loss)

    logging_step()
    return loss

  def configure_optimizers(self):
    '''see Pytorch Lightning documentations for details.
    '''
    if self.args.max_steps__ > 0:
      t_total = self.args.max_steps__
      self.args.num_train_epochs = self.args.max_steps__ // (self.args.num_batches // self.args.gradient_accumulation_steps) + 1
    else:
      t_total = self.args.num_batches // self.args.gradient_accumulation_steps * self.args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
      {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.args.weight_decay},
      {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
      ]
    if self.args.optim == 'AdamW':
      optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
    elif self.args.optim == 'Adamax':
      optimizer = torch.optim.Adamax(optimizer_grouped_parameters, lr=self.args.adamax_lr, eps=self.args.adam_epsilon)
    else:
      raise ValueError('Unsupported optimizer: %s.' % self.args.optim)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=self.args.warmup_steps, t_total=t_total)
    return [optimizer], [scheduler]

  def compute_soft_triplet_loss(self, mod_img1, img2, img3, target_soft_idx):
    '''Based on TIRG implementation
    see https://github.com/google/tirg
    In each run, we randomly format triplets of <anchor, positive, negative>
    '''
    img3_reshape = img3.reshape(-1, img3.shape[-1]) # reshape stack them up
    
    target_soft_idx_reshape = [] # match sequence idx to reshaped tensor within the 160
    for i_count,i in enumerate(target_soft_idx):
      target_soft_idx_reshape.append(
        {kkk+i_count*5:vvv for kkk,vvv in i.items() if vvv == 1.0} # +5 +10 +15 etc. ALSO remove the c2 type
      )
    
    triplets = []
    labels = list(range(mod_img1.shape[0])) + list(range(img2.shape[0]))

    for i in range(len(labels)//2):
      target_soft_idx_reshaped_i = target_soft_idx_reshape[i] \
        if i < mod_img1.shape[0] else target_soft_idx_reshape[i-mod_img1.shape[0]]
      triplets_i = []

      for j in list(target_soft_idx_reshaped_i.keys()):
        for k in range(len(labels)):
          if (labels[i] != labels[k]) and (labels[k] not in target_soft_idx_reshaped_i.keys()):
            triplets_i.append([i, mod_img1.shape[0]+img2.shape[0]+j, k, target_soft_idx_reshaped_i[j]])
      
      np.random.shuffle(triplets_i)
      triplets += triplets_i[:6] # 192 triplets, 6 each

    assert (triplets and len(triplets) < 2000)
    return self.soft_triplet_loss(torch.cat([mod_img1, img2, img3_reshape]), triplets)
  
  def validation_step(self, batch, batch_idx, dataloader_idx):
    '''Main batch-loop of validation that requires the model's forward 
    dataloader_0: compose reference img+txt
    dataloader_1: project all candidate images
    '''
    if dataloader_idx == 0: # img_txt
      oscar_in = [d1[1] for d1 in batch]
      f = self.compose_img_txt(oscar_in)
      return {'dataloader_idx':dataloader_idx, 'batch_idx':batch_idx, 'f':f,
      }
    elif dataloader_idx == 1: # img
      img2 = torch.stack([ii[0] for ii in batch])
      f = self.compose_candidate_img(img2)
      return {'dataloader_idx':dataloader_idx, 'batch_idx':batch_idx, 'f':f,
      }
  
  def eval_test_prepare_step_CIRR(self, step_outputs, is_test_split=False):
    try:
      val_dataloader = self.trainer.val_dataloaders[0]
    except:
      val_dataloader = self.trainer.test_dataloaders[0]
    valset = val_dataloader.dataset
    test_queries = valset.get_test_queries()

    # dataloader0 / reference img+txt -> all_queries
    img_txt_outputs = sorted(step_outputs[0], key=lambda k: k['batch_idx'])
    all_queries = torch.cat([ii['f'] for ii in img_txt_outputs])
    # dataloader1 / all candidate images -> all_imgs
    img_outputs = sorted(step_outputs[1], key=lambda k: k['batch_idx'])
    all_imgs = torch.cat([ii['f'] for ii in img_outputs])

    all_queries = all_queries.data.cpu().numpy()

    all_set_member_idx = [t['set_member_idx'] for t in test_queries]
    if is_test_split:
      # all_target_captions = None
      all_target_captions_soft = None
    else:
      # all_target_captions = [t['target_caption'] for t in test_queries]
      all_target_captions_soft = [t['target_caption_soft'] for t in test_queries] # Use the more comprehensive label for validation here

    all_imgs = all_imgs.data.cpu().numpy()
    all_captions = [img['captions'][0] for img in valset.imgs]

    # feature normalization
    for i in range(all_queries.shape[0]):
      all_queries[i, :] /= np.linalg.norm(all_queries[i, :])
    for i in range(all_imgs.shape[0]):
      all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])
    # match test queries to target images, get nearest neighbors
    sims = all_queries.dot(all_imgs.T)
    # remove query image
    for i, t in enumerate(test_queries):
      sims[i, t['source_img_id']] = sims[i].min()

    nn_result = [np.argsort(-sims[i, :]) for i in range(sims.shape[0])]
    nn_result = [[all_captions[nn] for nn in nns] for nns in nn_result]

    return all_target_captions_soft, all_set_member_idx, nn_result

  def eval_test_scoring_step_CIRR(self, all_target_captions_soft, all_set_member_idx, nn_result):
    out = []
    # Recall@K
    for k in [1, 2, 5, 10, 50, 100]:
      r = 0.0
      for i, nns in enumerate(nn_result):
        highest_r = 0.0
        for ii,ss in all_target_captions_soft[i].items():
          if ii in nns[:k]:
            highest_r = max(highest_r, ss) # update the score
        r += highest_r
      r /= len(nn_result)
      self.log('recall_top' + str(k) + '_correct_composition', r)
      out += [('recall_top' + str(k) + '_correct_composition', r)]
    
    # Recall_subset@K
    for k in [1, 2, 3]:
      r = 0.0
      for i, nns in enumerate(nn_result):
        nns = [iii for iii in nns if iii in all_set_member_idx[i]]
        highest_r = 0.0
        for ii,ss in all_target_captions_soft[i].items():
          if ii in nns[:k]:
            highest_r = max(highest_r, ss)
        r += highest_r
      r /= len(nn_result)
      self.log('recall_inset_top' + str(k) + '_correct_composition', r)
      out += [('recall_inset_top' + str(k) + '_correct_composition', r)]

    print('\n\n');pprint(out);print('\n\n')
    return out

  def validation_epoch_end(self, step_outputs: list):
    '''Collect features from all batches & GPUs
    Perform score calculation
    '''
    assert self.args.accelerator is None, NotImplementedError('Not supported for multi-GPU')
    all_target_captions_soft, all_set_member_idx, nn_result = self.eval_test_prepare_step_CIRR(step_outputs)
    _ = self.eval_test_scoring_step_CIRR(all_target_captions_soft, all_set_member_idx, nn_result)
    
  def test_step(self, *args, **kwargs):
    return self.validation_step(*args, **kwargs)

  def generate_test_json(self, nn_result, all_set_member_idx, DSET, TOP_HOW_MANY, METRIC_NAME):
    import json
    DSET_VERSION = DSET.version

    t_json = {'version':DSET_VERSION, 'metric': METRIC_NAME}
    if METRIC_NAME == 'recall':
      for n_, t_ in zip(nn_result, DSET.test_queries):
        pair_id_ = str(t_['pairid'])
        nn_ranks_asin_ = [DSET.id2asin[nn_] for nn_ in n_]
        t_json[pair_id_] = nn_ranks_asin_[:TOP_HOW_MANY]
    elif METRIC_NAME == 'recall_subset':
      for n_, a_, t_ in zip(nn_result, all_set_member_idx, DSET.test_queries):
        pair_id_ = str(t_['pairid'])
        nn_ranks_asin_ = [DSET.id2asin[nn_] for nn_ in n_ if nn_ in a_]
        t_json[pair_id_] = nn_ranks_asin_[:TOP_HOW_MANY]

    '''Directory for saving:
    e.g., saved_models/cirr_rc2/my_computer/version_0/checkpoints/test1_pred_ranks_recall.json
    '''
    ckp_dir = self.trainer.default_root_dir
    if not os.path.exists(ckp_dir):
      os.mkdir(ckp_dir)
    
    t_json_path = os.path.join(ckp_dir, 'test1_pred_ranks_'+METRIC_NAME+'.json')
    json.dump(t_json, open(t_json_path,'w'))
    print('\n|> Prediction file saved to %s' % t_json_path)
    return

  def test_epoch_end(self, step_outputs: list):
    assert self.args.accelerator is None, NotImplementedError('Not supported for multi-GPU')
    _, all_set_member_idx, nn_result = self.eval_test_prepare_step_CIRR(step_outputs, is_test_split=True)
    
    test_dataset = self.trainer.test_dataloaders[0].dataset
    self.generate_test_json(nn_result, all_set_member_idx, DSET=test_dataset, TOP_HOW_MANY=50, METRIC_NAME='recall')
    self.generate_test_json(nn_result, all_set_member_idx, DSET=test_dataset, TOP_HOW_MANY=3, METRIC_NAME='recall_subset')
    
    
    