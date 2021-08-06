

# MIT License

# Copyright (c) 2018 Nam Vo

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

"""Metric learning functions.

Codes are modified from:
https://github.com/lugiavn/generalization-dml/blob/master/nams.py
"""

import numpy as np
import torch
import pytorch_lightning as pl

import pdb

def pairwise_distances(x, y=None):
  """Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between
    x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    source:
    https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/2
    """
  x_norm = (x**2).sum(1).view(-1, 1)
  if y is not None:
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y**2).sum(1).view(1, -1)
  else:
    y_t = torch.transpose(x, 0, 1)
    y_norm = x_norm.view(1, -1)

  dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)

  return torch.clamp(dist, 0.0, np.inf)

def pairwise_cossim(x, y=None):
  """Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between
    x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    source:
    https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/2
  """
  x1 = x.unsqueeze(1).repeat(1,x.shape[0],1)
  x2 = x1.detach().clone().transpose(0,1).contiguous()
  
  x1_dim0 = x1.shape[0]
  x1_ = x1.view(-1, x1.shape[-1])
  x2_ = x2.view(-1, x2.shape[-1])  
  return 1 - torch.nn.CosineSimilarity(dim=1)(x1_,x2_).view(x1_dim0,x1_dim0,-1).squeeze()

class MyTripletLossFuncV1(torch.autograd.Function):
  '''accept also the soft accuracy score
  But for now nothing was used, because we exclude anything other than s == 1.0 during training
  '''    
  @staticmethod
  def forward(ctx, features, triplets, metric):
    assert metric in ['pdist','cdist']; ctx.metric = metric
    ctx.triplets = triplets
    ctx.triplet_count = len(triplets)
    ctx.save_for_backward(features)

    if ctx.metric == 'pdist':
      ctx.distances = pairwise_distances(features).cpu().numpy()
    elif ctx.metric == 'cdist':
      ctx.distances = pairwise_cossim(features).cpu().numpy()
    loss = 0.0
    triplet_count = 0.0
    correct_count = 0.0
    for i, j, k, s in ctx.triplets:
      w = 1.0
      triplet_count += w
      loss += w * np.log(1 +
                         np.exp(ctx.distances[i, j] - ctx.distances[i, k]))
      if ctx.distances[i, j] < ctx.distances[i, k]:
        correct_count += 1 * s 

    loss /= triplet_count
    return torch.FloatTensor((loss,))

  @staticmethod
  def backward(ctx, grad_output):
    features, = ctx.saved_tensors
    features_np = features.cpu().numpy()
    grad_features = features.clone() * 0.0
    grad_features_np = grad_features.cpu().numpy()

    for i, j, k, _ in ctx.triplets:
      w = 1.0
      f = 1.0 - 1.0 / (
          1.0 + np.exp(ctx.distances[i, j] - ctx.distances[i, k]))
      grad_features_np[i, :] += w * f * (
          features_np[i, :] - features_np[j, :]) / ctx.triplet_count
      grad_features_np[j, :] += w * f * (
          features_np[j, :] - features_np[i, :]) / ctx.triplet_count
      grad_features_np[i, :] += -w * f * (
          features_np[i, :] - features_np[k, :]) / ctx.triplet_count
      grad_features_np[k, :] += -w * f * (
          features_np[k, :] - features_np[i, :]) / ctx.triplet_count

    for i in range(features_np.shape[0]):
      grad_features[i, :] = torch.from_numpy(grad_features_np[i, :])
    grad_features *= float(grad_output.data[0])
    return grad_features, None, None

class TripletLoss(torch.nn.Module):
  """Class for the triplet loss.
  """
  def __init__(self, pre_layer=None, metric='pdist'):
    super(TripletLoss, self).__init__()
    self.pre_layer = pre_layer
    self.metric = metric

  def forward(self, x, triplets):
    if self.pre_layer is not None:
      x = self.pre_layer(x)

    if len(triplets[0]) == 3: #* append default weight to the triplet
      triplets = [ii + [1.0] for ii in triplets]
    assert len(triplets[0]) == 4, NotImplementedError
    loss = MyTripletLossFuncV1.apply(x, triplets, self.metric)
    return loss
