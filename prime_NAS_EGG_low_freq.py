from absl import app
from absl import flags
from absl import logging

# These tensorflow installs are automatically provided by the
# Google colab runtime. If you want to run this code locally,
# make sure to install tensorflow and tensorflow_probability.

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
import pandas as pd
import numpy as np
import os
import pickle
import csv
from typing import Optional, Dict, List
from copy import deepcopy

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')), flush=True)
gfile = tf.io.gfile.GFile



def get_angles(pos, i, d_model):
  """Get angles for using tansformer."""
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates


def positional_encoding(position, d_model):
  """Obtain positional encdoing for training the PRIME Transformer."""
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
  
  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)


class SplitEmbeddingLayer(tf.keras.layers.Layer):
  """Layer for embedding individual components in a split way"""
  def __init__(self, softmax_splits=None, output_size=32):
    """
    Initialize the layer to split the input and generate embeddings for
    each field.
    """
    super(SplitEmbeddingLayer, self).__init__(trainable=True)
    self.softmax_splits = softmax_splits
    self.output_size = output_size

    # create layers
    self.dense_layers = []
    print (self.softmax_splits)
    for idx, val in enumerate(self.softmax_splits):
      self.dense_layers.append(
          tf.keras.layers.Dense(
              self.output_size, name='insidelayer_' + str(idx)))

    # Add position embeddings
    self.pos_encoding = positional_encoding(position=200, d_model=output_size)

  def call(self, x):
    """Call the Split embedding function."""
    split_x = tf.split(x, num_or_size_splits=self.softmax_splits, axis=-1)
    modified_splits = []
    idx = 0
    for param in split_x:
      out = self.dense_layers[int(idx)](param)
      modified_splits.append(tf.expand_dims(out, axis=1))
      idx += 1
    out = tf.concat(modified_splits, axis=1)
    # print ('Out shape before: ', out)
    out = out + self.pos_encoding[:, :len(modified_splits), :]
    # print ('Out shape after: ', out)
    return out


def scaled_dot_product_attention(q, k, v, mask):
  """Scaled dot product attention in transformer."""
  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(
      scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
  return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])


class MultiHeadAttention(tf.keras.layers.Layer):
  """Multi Head Attention for the model."""

  def __init__(self, d_model, num_heads):
    """Initialize the multi-head attention model."""
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)

    self.dense = tf.keras.layers.Dense(d_model)

  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the
    shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]

    q = self.wq(q)
    k = self.wk(k)
    v = self.wv(v)

    q = self.split_heads(q, batch_size)  
    k = self.split_heads(k, batch_size) 
    v = self.split_heads(v, batch_size)

    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3]) 

    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model)) 

    output = self.dense(concat_attention) 

    return output, attention_weights


class TransformerLayer(tf.keras.layers.Layer):
  """Define the transformer layer to be used in the PRIME Transformer model."""

  def __init__(self, d_model, num_heads, dff, rate=0.1):
    """Initialize the transformer layer."""
    super(TransformerLayer, self).__init__()

    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)

  def call(self, x, training=True, mask=None):
    attn_output, _ = self.mha(x, x, x, mask)  
    # (batch_size, input_seq_len, d_model)
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x + attn_output)  
    # (batch_size, input_seq_len, d_model)

    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output) 
    # (batch_size, input_seq_len, d_model)
    return out2


def weighted_mse_loss(input, target, weight):
  """Compute weighted MSE Loss"""
  mse_loss_val = (tf.squeeze(input) - tf.squeeze(target))**2
  return tf.reduce_mean(mse_loss_val * tf.squeeze(weight))


def weighted_huber_loss(input, target, weight):
  """Compute weighted Huber Loss"""
  mse_loss = tf.keras.losses.Huber(
      reduction=tf.keras.losses.Reduction.NONE)
  return tf.reduce_mean(mse_loss(
      y_pred=tf.squeeze(input),
      y_true=tf.squeeze(target)) * tf.squeeze(weight))


def weighted_approx_loss(input, target, weight):
  """Compute weighted Approximation Loss"""
  abs_diff = tf.abs(tf.squeeze(input) - tf.squeeze(target))
  ratio_diff = abs_diff / (tf.abs(tf.squeeze(target)) + 1e-6)
  return tf.reduce_mean(ratio_diff * tf.squeeze(weight))

#@title Helper functions for ranking loss computation

def ranking_loss(input, target, context=None):
  """Compute measures of ranking for the PRIMETransformerModel."""
  if context is not None:
    # Compute ranking loss per context, and then average it.
    unique_contexts, indices = tf.unique(
        tf.squeeze(tf.cast(context, tf.int32)), name='None')
    all_corr = []
    for idx in range(unique_contexts.shape[0]):
      curr_context = unique_contexts[idx]
      locations_idx = tf.squeeze(tf.where(tf.equal(indices, curr_context)))
      input_tmp = tf.gather(
          tf.squeeze(input), indices=locations_idx)
      target_tmp = tf.gather(
          tf.squeeze(target), indices=locations_idx)
      input_ranks = tf.argsort(input_tmp, axis=-1)
      target_ranks = tf.argsort(target_tmp, axis=-1)
      input_ranks = tf.cast(tf.argsort(input_ranks, axis=-1), dtype=tf.float32)
      target_ranks = tf.cast(tf.argsort(target_ranks, axis=-1),
                             dtype=tf.float32)
      std_input = tf.math.reduce_std(input_ranks)
      std_target = tf.math.reduce_std(target_ranks)
      cov = tf.reduce_mean((target_ranks - tf.reduce_mean(target_ranks)) *\
                           (input_ranks - tf.reduce_mean(input_ranks)))
      pearson_corr = cov/ (std_target * std_input)
      all_corr.append(pearson_corr)
    print (all_corr)
    pearson_corr = tf.reduce_mean(pearson_corr)
  else:
    input = tf.squeeze(input)
    target = tf.squeeze(target)
    input_ranks = tf.argsort(input, axis=-1)
    target_ranks = tf.argsort(target, axis=-1)
    input_ranks = tf.cast(tf.argsort(input_ranks, axis=-1), dtype=tf.float32)
    target_ranks = tf.cast(tf.argsort(target_ranks, axis=-1), dtype=tf.float32)
    std_input = tf.math.reduce_std(input_ranks)
    std_target = tf.math.reduce_std(target_ranks)
    cov = tf.reduce_mean((target_ranks - tf.reduce_mean(target_ranks)) *\
                         (input_ranks - tf.reduce_mean(input_ranks)))
    pearson_corr = cov/ (std_target * std_input)
  return pearson_corr


def ranking_trainable_loss(input, target, context=None):
  """Compute a differentiable ranking loss, that can be used for training."""
  if context is not None:
    unique_contexts, indices = tf.unique(
        tf.squeeze(tf.cast(context, tf.int32)), name='None')
    all_corr = []
    for idx in range(unique_contexts.shape[0]):
      curr_context = unique_contexts[idx]
      locations_idx = tf.squeeze(tf.where(tf.equal(indices, curr_context)))
      input_tmp = tf.expand_dims(tf.gather(
          tf.squeeze(input), indices=locations_idx), 1)
      target_tmp = tf.expand_dims(tf.gather(
          tf.squeeze(target), indices=locations_idx), 1)
      input_transpose = tf.transpose(input_tmp, [1, 0]) # 1 x B
      target_transpose = tf.transpose(target_tmp, [1, 0]) # 1 x B
      diff_true = input_tmp - input_transpose # B x 1 - 1 x B = B x B = y_i - y_j
      diff_pred = target_tmp - target_transpose # fx_i - fx_j
      product = tf.sign(diff_true) * diff_pred  # sign(y_i = y_j) * (fx_i - fxj)
      bce_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
          labels=tf.ones_like(product), logits=product))
      all_corr.append(bce_loss)
    bce_loss = tf.reduce_mean(all_corr)
  else:
    input_transpose = tf.transpose(input, [1, 0]) # 1 x B
    target_transpose = tf.transpose(target, [1, 0]) # 1 x B
    diff_true = input - input_transpose # B x 1 - 1 x B = B x B = y_i - y_j
    diff_pred = target - target_transpose # fx_i - fx_j
    product = tf.sign(diff_true) * diff_pred  # sign(y_i = y_j) * (fx_i - fxj)
    bce_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(product), logits=product)
  return tf.reduce_mean(bce_loss)

#@title Helper function for Kendall correlation

def kendall_correlation(input, target, context=None):
  """Compute Kendall's correlation over the input, target and context."""
  if context is not None:
    unique_contexts, indices = tf.unique(
        tf.squeeze(tf.cast(context, tf.int32)), name='None')
    all_corr = []
    for idx in range(unique_contexts.shape[0]):
      curr_context = unique_contexts[idx]
      locations_idx = tf.squeeze(tf.where(tf.equal(indices, curr_context)))
      input_tmp = tf.expand_dims(tf.gather(
          tf.squeeze(input), indices=locations_idx), 1)
      target_tmp = tf.expand_dims(tf.gather(
          tf.squeeze(target), indices=locations_idx), 1)
      input_transpose = tf.transpose(input_tmp, [1, 0])
      target_transpose = tf.transpose(target_tmp, [1, 0])
      diff_true = input_tmp - input_transpose
      diff_pred = target_tmp - target_transpose
      product = tf.sign(diff_true) * tf.sign(diff_pred)
      positive_pairs = tf.where(tf.greater_equal(product, tf.zeros_like(product)),
                                tf.ones_like(product), tf.zeros_like(product))
      n = tf.cast(tf.shape(input_tmp)[0], dtype=tf.float32)
      total_positive = tf.reduce_sum(positive_pairs) - n
      ratio = total_positive/ (n * (n-1))
      all_corr.append(ratio)
    ratio = tf.reduce_mean(all_corr)
  else:
    input_transpose = tf.transpose(input, [1, 0])
    target_transpose = tf.transpose(target, [1, 0])
    diff_true = input - input_transpose
    diff_pred = target - target_transpose
    product = tf.sign(diff_true) * tf.sign(diff_pred)
    positive_pairs = tf.where(tf.greater_equal(product, tf.zeros_like(product)),
                              tf.ones_like(product), tf.zeros_like(product))
    n = tf.cast(tf.shape(input)[0], dtype=tf.float32)
    total_positive = tf.reduce_sum(positive_pairs) - n
    ratio = total_positive/ (n * (n-1))
  return 2 * ratio - 1.0

"""## Code for the PRIME surrogate"""

#@title Definition of the PRIME surrogate model, training procedure

class PRIMETransformerModel(tf.keras.Model):
  """
  The transformer model used by PRIME. This class implements ability to 
  instantiate a transformer model, and train it via the PRIME training objective
  (Equation 3 in https://arxiv.org/abs/2110.11346). 
  
  Additionally it also implements the ability to train a contextual model,
  conditioned on the context. 
  """

  def __init__(self,
               num_outputs,
               num_inputs,
               optimizer,
               layers=(256, 256, 256),
               penalty_weight=10.0,
               negative_sampler=None,
               contextual=False,
               params_dict=None):
    """Initializes the PRIMETransformer model.

    Args:
      num_outputs: the dimensionality of the output of the PRIME surrogate. 
        Typically set to 1, but you can increase it to model multiple cost
        functions together.
      num_inputs: the dimensionality of the total number of inputs to the model.
      optimizer: the optimizer to optimize the trainable model.
      layers: hidden layer sizes for the feed-forward layers after extracting
        the transformer embedding.
      penalty_weight: the value of alpha in Equation 2 in PRIME.
      negative_sampler: an instance of a negative sampler. A negative sampler
        is basically an optimizer that can take in the current snapshot of the
        this PRIMETransformerModel, and optimize the predictions of the current
        model snapshot w.r.t its input. In the paper, we utilize an evolutionary
        optimizer to optimize the predictions. For this code release, we present
        a simple gradient-descent based optimizer for optimization as a
        demonstration. Users are encouraged to pass in their relevant
        negative sampler here.
      contextual: bool, indicates whether we are training a contextual model
        or a non-contextual model. Contextual is used for multi-model and
        zero-shot experiments. 
      params_dict: dictionary. Can store additional parameters and their values.
        This dictionary provides an easy and convenient way to add new hyper-
        parameters, via keys of this dictionary. 
    """
    super().__init__()
    self.num_inputs = num_inputs
    self.num_outputs = num_outputs
    self.optimizer = optimizer
    self.params_dict = params_dict
    self.penalty_weight = penalty_weight
    self.contextual = contextual

    # Setting the following variable to True shouldn't cause issues since
    # it is not passed into the GradientTape, but better to be safe, and set
    # it to false if the variable is not used.

    # This variable determines the alpha multiplier in Equation 2.
    self.log_cql_alpha = tf.Variable(tf.math.log(self.penalty_weight + 1e-6),
                                     trainable=False)
    self.cql_alpha_value = tf.Variable(self.penalty_weight, trainable=False)

    self.negative_sampler = negative_sampler

    # In the paper, we use an evolutionary optimizer for obtaining adversarial
    # examples. However, unfortunately, this optimizer is `proprietary`, and so
    # we provide the example negative sampler that uses gradient ascent, similar
    # to conservative objective mocels https://arxiv.org/abs/2107.06882.
    self.num_gradient_infer_steps = 0
    if 'num_gradient_steps' in params_dict:
      self.num_gradient_infer_steps = params_dict['num_gradient_steps']

    self.opt_lr = 1e-3
    if 'opt_lr' in params_dict:
      self.opt_lr = params_dict['opt_lr']

    # the multiplier beta in Equation 3 in the paper.
    self.infeasible_alpha = 0.01
    if 'infeasible_alpha' in params_dict:
      self.infeasible_alpha = params_dict['infeasible_alpha']

    # Since the input to the model is a concatenation of one-hot values
    # representing each field, using the input_splits parameter, we partition
    # this big input vector into a list of one-hot vectors, one corresponding 
    # to each discrete parameter. 
    self.input_splits = None
    if 'input_splits' in params_dict:
      self.input_splits = params_dict['input_splits']

    # We use an architecture which resembles a mixture of experts, and so the
    # following parameter decides how many parameters we wish to have.
    self.num_votes = 1
    if 'num_votes' in params_dict:
      self.num_votes = params_dict['num_votes']

    # Whether to add dropout or not, in intermediate layers of the model, as 
    # a means to prevent overfitting.
    use_dropout = False
    if 'use_dropout' in params_dict:
      use_dropout = params_dict['use_dropout']

    if self.contextual:
      """For contextual version of PRIME"""
      self.num_contexts = 0
      if 'num_contexts' in params_dict:
        self.num_contexts = params_dict['num_contexts']

    print ('Infeasible alpha: ', self.infeasible_alpha)
    print ('CQL Alpha: ', self.log_cql_alpha)
    print ('Num votes: ', self.num_votes)

    self.input_layer = tf.keras.Input(num_inputs)
    temp_num_inputs = num_inputs

    # The following layer splits the input into a list of embeddings for 
    # each parameter. Check the SplitEmbeddingLayer class for details.
    x = SplitEmbeddingLayer(softmax_splits=self.input_splits,
                            output_size=64)(self.input_layer)
    if use_dropout:
      x = tf.keras.layers.Dropout(rate=0.1)(x)
  
    # Now feed the split embedding layer output into TransformerLayer
    x = TransformerLayer(d_model=64, num_heads=8, dff=256)(x)
    x = TransformerLayer(d_model=64, num_heads=8, dff=256)(x)
    
    x = tf.keras.layers.Reshape(target_shape=(512,))(x)
    
    if self.contextual:
      context_input = tf.keras.Input(self.num_contexts)
      out_context = tf.keras.layers.Dense(512, use_bias=False)(context_input)

      # Pointwise multiply the contexts to make sure that the context
      # conditioning is done properly. From https://arxiv.org/abs/1912.13465.
      x = x * out_context
      self._base_network = tf.keras.Model(
        inputs=[self.input_layer, context_input], outputs=x)
    else:
      self._base_network = tf.keras.Model(
          inputs=self.input_layer, outputs=x)

    self.optimize_networks = [self._base_network,]

    # Now feedforward layers to finish the model
    layers = list(layers)
    layers[0] = 64 * len(self.input_splits)

    """Voting based routing"""
    num_networks = self.num_votes
    self._all_networks = []
    for jdx in range(num_networks):
      # Make each of the networks used in routing
      new_network = tf.keras.Sequential()
      for idx in range(len(layers) - 1):
        new_network.add(
            tf.keras.layers.Dense(layers[idx+1], input_shape=(layers[idx],)))
        new_network.add(tf.keras.layers.LeakyReLU(0.1))
        if use_dropout:
          new_network.add(tf.keras.layers.Dropout(rate=0.1))

      new_network.add(tf.keras.layers.Dense(
          num_outputs, input_shape=(layers[idx],)))
      self._all_networks.append(new_network)

    self.optimize_networks.extend(self._all_networks)

    # Now make the network that decides the contribution of these
    self.voting_network = tf.keras.Sequential()
    if self.contextual:
      self.voting_network.add(
          tf.keras.layers.Dense(layers[1], input_shape=(2*layers[0],)))
    else:
      self.voting_network.add(
          tf.keras.layers.Dense(layers[1], input_shape=(layers[0],)))
    self.voting_network.add(tf.keras.layers.LeakyReLU(0.1))
    if use_dropout:
      self.voting_network.add(tf.keras.layers.Dropout(rate=0.1))

    self.voting_network.add(
        tf.keras.layers.Dense(self.num_votes, input_shape=(layers[1],)))

    if self.contextual:
      # Add the vote generation network input again
      self.embedding_network = tf.keras.Sequential()
      self.embedding_network.add(
          tf.keras.layers.Dense(256))
      self.embedding_network.add(tf.keras.layers.LeakyReLU(0.1))
      self.embedding_network.add(
          tf.keras.layers.Dense(layers[0]))

      self.optimize_networks.append(self.embedding_network)
    self.optimize_networks.append(self.voting_network)

    print ('All networks: ', len(self.optimize_networks))

  @tf.function
  def call(self, inputs, training=True, with_logging=False):
    """Function to call one forward pass on the PRIME Transformer."""
    extra_dict = dict()
    if not self.contextual:
      transformer_embedding = self._base_network(inputs, training=training)
    else:
      # TODO(aviralkumar): Fix the hardcoded 77 input dimensionality in code
      if not isinstance(inputs, list) and not isinstance(inputs, tuple):
        inputs = (inputs[:, 163], inputs[:, 163:])

      transformer_embedding = self._base_network(inputs, training=training)
      
    # Get all outputs from each expert
    all_outputs = []
    for idx in range(self.num_votes):
      all_outputs.append(
          self._all_networks[idx](transformer_embedding, training=training))
    
    # Get the voting probabilities
    if self.contextual:
      vote_input = self.embedding_network(inputs[1])
      vote_input = tf.concat([transformer_embedding, vote_input], axis=-1)
      vote_logit = self.voting_network(vote_input, training=training)
    else:
      vote_logit = self.voting_network(transformer_embedding,
                                        training=training)

    # Append all_outputs in a list and compute average score
    all_outputs = tf.concat(all_outputs, axis=-1)   # [B x num_votes]
    vote_prob = tf.nn.softmax(vote_logit, axis=-1)  # [B x num_votes]
    vote_entropy = tf.reduce_sum(
        tf.nn.log_softmax(vote_logit, axis=-1) * vote_prob, axis=-1)
    extra_dict['vote_entropy'] = tf.reduce_mean(vote_entropy)
    fwd_model_pred = tf.reduce_sum(vote_prob * all_outputs, axis=-1)
    fwd_model_pred = tf.expand_dims(fwd_model_pred, axis=-1)
    
    if with_logging:
      return fwd_model_pred, extra_dict

    return fwd_model_pred

  def compute_loss(self, data_batch, loss_type='mse', training=True,
                   ranking_penalty_weight=0.0, inp_batch_type='valid'):
    """
    Compute the loss function and additional logging metrics for training.

    Args:
      data_batch: A dictionary of various input fields, and their corresponding
        tensor values. The keys for this dictionary are:
        - design --> denotes the input (accelerator config in this case)
        - objective --> denotes the objective value for the given input
        - context_id --> denotes the context vector for the case of contextual

      loss_type: string, either mse or mse+rank. It essentially computes the
        training loss used to train the PRIME model. We can optionally add some
        ranking regularization for training if needed. Though, we did not find
        this to be essential. 

      inp_batch_type: string, either 'valid' or 'mixed'. Mixed indicates that
        the batch consists of both valid and invalid samples, whereas valid
        indicates the samples are only valid samples.

      ranking_penalty_weight: float, the weight on the ranking loss function
        in addition to the PRIME objectives. This is not needed for PRIME, but
        can help in some cases. So, leaving the facility here.
    """
    loss_dict = dict()
    if loss_type == 'mse':
      fwd_loss = weighted_mse_loss
    elif loss_type == 'mse+rank':
      fwd_loss = weighted_mse_loss
      ranking_loss_fn = ranking_trainable_loss

    loss_dict['y_values_max'] = tf.reduce_max(data_batch['objective'])
    loss_dict['y_values_mean'] = tf.reduce_mean(data_batch['objective'])

    data_batch = data_batch.copy()
    weights = tf.ones_like(data_batch['objective'])
    
    if self.contextual:
      model_pred, extra_dict = self(
            inputs=[data_batch['design'], data_batch['context_id']],
            training=training, with_logging=True)
    else:
      model_pred, extra_dict = self(
          data_batch['design'], training=training, with_logging=True)

    loss_dict.update(extra_dict)

    if self.negative_sampler is not None:
      # This branch of the code will not run off-the-shelf, since it assumes 
      # access to a negative_sampler. A negative sampler is simply any kind of
      # optimizer that can take in the current PRIMETransformerModel and
      # optimize its predictions.
      negatives_batch = self.negative_sampler.run_inference(
          num_iters=5, model=self)
      negatives_pred = self(inputs=negatives_batch, training=training)
    else:
      negatives_batch = self.infer_negatives(data_batch)
      if self.contextual:
        negatives_pred = self(
            (negatives_batch['design'], negatives_batch['context_id']),
            training=training)
      else:
        negatives_pred = self(negatives_batch['design'], training=True)
    
    negatives_pred = tf.clip_by_value(negatives_pred, clip_value_min=-4000.0,
                                      clip_value_max=4000.0)
    
    cql_loss = tf.reduce_mean(negatives_pred)
    cql_loss = tf.clip_by_value(cql_loss, 
                                clip_value_min=-4000, 
                                clip_value_max=1e6)
    loss_dict['negatives_dist'] = tf.reduce_mean(negatives_pred)

    mse_loss = weighted_mse_loss(
        model_pred, data_batch['objective'], weights)

    if loss_type == 'mse+rank':
      if self.contextual:
        avg_ranking_train_loss = ranking_loss_fn(
            model_pred, data_batch['objective'],
            context=data_batch['raw_context'])
      else:
        avg_ranking_train_loss = ranking_loss_fn(
            model_pred, data_batch['objective'])
    else:
      avg_ranking_train_loss = 0.0

    # Only used for logging, measures how big the MSE error is relative to
    # the output of the model. 
    avg_approx_loss = weighted_approx_loss(
        model_pred, data_batch['objective'], weights)
    passed_context = None

    if self.contextual:
      passed_context = data_batch['raw_context']
    
    avg_ranking_loss = ranking_loss(
        model_pred, data_batch['objective'], context=passed_context)
    avg_kendall_loss = kendall_correlation(
        model_pred, data_batch['objective'], context=passed_context)

    train_loss = mse_loss
    loss_dict['mse_loss'] = mse_loss
    loss_dict['avg_approx_loss'] = avg_approx_loss
    loss_dict['avg_ranking_loss'] = avg_ranking_loss
    loss_dict['avg_ranking_train_loss'] = avg_ranking_train_loss
    loss_dict['avg_kendall_loss'] = avg_kendall_loss
    loss_dict['cql_loss'] = cql_loss
    loss_dict['positives_pred'] = tf.reduce_mean(negatives_pred)
    loss_dict['model_pred_average'] = tf.reduce_mean(model_pred)
    train_loss = train_loss + ranking_penalty_weight * avg_ranking_train_loss
    train_loss = train_loss - self.cql_alpha_value * cql_loss
    
    if inp_batch_type is not 'valid':
      weights_negatives = tf.ones_like(data_batch['objective'])
      if self.contextual:
        model_pred_invalid, invalid_dict = self(
          inputs=(data_batch['invalid/design'], data_batch['context_id']),
          training=training, with_logging=True)
      else:
        model_pred_invalid, invalid_dict = self(
            data_batch['invalid/design'], training=training, with_logging=True)

      for key in invalid_dict:
        loss_dict['invalid/'+key] = invalid_dict[key]

      ## Conservatism training
      loss_dict['y_value_infeasible'] = tf.reduce_mean(model_pred_invalid)
      loss_dict['y_value_infeasible'] = tf.clip_by_value(
          loss_dict['y_value_infeasible'], 
          clip_value_min=-1000, clip_value_max=1e6)
      train_loss = train_loss + self.infeasible_alpha *\
            loss_dict['y_value_infeasible']

      mse_loss_invalid = weighted_mse_loss(
          model_pred_invalid, data_batch['invalid/objective'], 
          weights_negatives)
      avg_approx_loss_invalid = weighted_approx_loss(
          model_pred_invalid, data_batch['invalid/objective'], 
          weights_negatives)
      mse_loss = mse_loss + mse_loss_invalid
      loss_dict['mse_loss_invalid'] = mse_loss_invalid
      loss_dict['mse_loss_overall'] = mse_loss
      loss_dict['avg_approx_loss_invalid'] = avg_approx_loss_invalid
    return loss_dict, train_loss

  def perform_training(self, batch, loss_type,
                       ranking_penalty_weight=0.0, **kwargs):
    """
    Actually perform training by computing loss, and then taking gradients
    through it. Makes sure to backpropagate through all networks.
    """
    with tf.GradientTape(
        watch_accessed_variables=False, persistent=True) as tape:
      tape.watch(
          [v for net in self.optimize_networks\
           for v in net.trainable_variables])
      loss_dict, loss_train = self.compute_loss(
          batch, loss_type, training=True,
          ranking_penalty_weight=ranking_penalty_weight)

    grads = tape.gradient(loss_train,
                          [v for net in self.optimize_networks\
                           for v in net.trainable_variables])
    gen_grads_op = self.optimizer.apply_gradients(
        zip(grads, [v for net in self.optimize_networks\
                    for v in net.trainable_variables]))
    return loss_dict

  def measure_stats(self, batch, batch_type='valid', **kwargs):
    """Simply make a forward pass through compute_loss to measure losses."""
    loss_dict, _ = self.compute_loss(batch, loss_type='mse+rank',
                                     training=False,
                                     inp_batch_type=batch_type)
    return loss_dict

  def infer_negatives(self, batch):
    """Run gradient descent to obtain negative examples"""
    temp_batch = dict()
    log_probs = batch['design']
    if self.contextual:
      contexts = batch['context_id']
    for _ in range(self.num_gradient_infer_steps):
      with tf.GradientTape(
          watch_accessed_variables=False, persistent=False) as tape:
        tape.watch(log_probs)
        if self.contextual:
          model_pred = self((log_probs, contexts), training=False)
        else:
          model_pred = self(log_probs, training=False)
      grad = tape.gradient(model_pred, log_probs)
      log_probs = log_probs + self.opt_lr * grad[0]
    temp_batch['design'] = tf.stop_gradient(log_probs)
    if 'context_id' in batch and self.contextual:
      temp_batch['context_id'] = batch['context_id']
    return temp_batch

"""# Data Loading and Problem Definition

## Hardware Optimization Problem & Offline Data
"""

#@title Define the hardware optimization problem

class HardwareOptProblem:
  """
  Problem for loading the task dataset and training
  """
  def __init__(self,
               config: dict,
               data_file: dict, 
               params_dict: Optional[dict] = None):
    """Initialize a hardware optimization problem.

    config: a dictionary of various input fields and their corresponding
      possible valid number of discrete values. 
    data_file: a dictionary of a list of various input fields.
    params_dict: a dictionary of additional inputs to the HardwareOptProblem.
    """

    # Batch size for the batch sampling
    self.batch_size = 256
    if 'batch_size' in params_dict:
      self._batch_size = params_dict['batch_size']


    # Add any area constraints or not: this flag enables filtering the data
    # based on whether the area constraint is not satisfied
    self._add_area_constraints = False
    if 'add_area_constraints' in params_dict:
      self._add_area_constraints = params_dict['add_area_constraints']

    self.dataset = PRIMEDataset(config=config,
                                data_dict=data_file)
    

    # Choose what kind of batch to provide while training the model
    self.get_training_batch = self.get_all_batch
    self.get_valid_batch = self.get_all_batch

  def get_all_batch(self,):
    """Sample i.i.d. from the entire dataset."""
    indices = np.random.randint(1, 
                                self.dataset._top, self._batch_size)
    batch_x, batch_y = self.dataset._get_batch(indices)
    batch_dict = dict()
    batch_dict['design'] = batch_x
    batch_dict['objective'] = batch_y
    return batch_dict

  def get_full_valid_batch(self,):
    """Sample i.i.d. from the entire dataset."""
    indices = np.random.randint(1, 
                                self.dataset._top, self.dataset.size)
    batch_x, batch_y = self.dataset._get_batch(indices)
    batch_dict = dict()
    batch_dict['design'] = batch_x
    batch_dict['objective'] = batch_y
    return batch_dict

  def get_top_batch(self,):
    """Get only the top scoring batch for eval"""
    indices = self.dataset._tf_dataset['argsort'][-self.batch_size:]
    batch_x, batch_y = self.dataset._get_batch(indices)
    batch_dict = dict()
    batch_dict['design'] = batch_x
    batch_dict['objective'] = batch_y
    return batch_dict



class PRIMEDataset(tf.Module):
  """
  Load the dataset to be able to train the PRIMETransformerModel.  
  """
  def __init__(self,
               config,
               data_dict: dict,
               **kwargs):
    """Create a dataset for training PRIME."""
    self._config = config

    self.data_dict = data_dict
    self._design_space_dict = {}
    self._segment_lengths = {}
    self._max_ctr = 0
    self._eval_metric_keys = ['accuracy']

    self._active_training_keys = ['param_1', 'param_2', 'param_3',
                                  'param_4', 'param_5', 'param_6', 'param_7', 'param_8']

    self._tf_dataset = {}
    self._top = 0
    if self.data_dict is not None:
      self._setup_dataset()

  def _setup_dataset(self,):
    """Main function to setup the dataset"""
    self.load_or_refresh_config()
    logging.info('Loading dataset..')
    self._convert_to_tf_dataset()
    self.get_score_function()
    print ('Loaded dataset....', self.size)

  def get_input_splits(self,):
    """Get the splits of input of the dataset."""
    lengths = []
    for key in self._active_training_keys:
      ctr_idx = self._design_space_dict[key]['ctr']
      lengths.append(self._segment_lengths[ctr_idx])
    self._active_lengths = lengths
    return lengths

  def get_score_function(self,):
    """Get the objective function which is being maximized"""
    accuracy = self._tf_dataset['accuracy'].numpy()
    scores = accuracy
    self._tf_dataset['score'] = tf.convert_to_tensor(
        scores, dtype=tf.float32)
    print ('Score stats: ')
    print ('--------------------------------------------')
    print ('Max: ', scores.max())
    print ('Mean: ', scores.mean())
    print ('Min: ', scores.min())
    print ('--------------------------------------------')

    # Since we need top batch for eval, store top scores
    self._tf_dataset['argsort'] = np.argsort(
        self._tf_dataset['score'].numpy())
    return scores

  def _convert_to_tf_dataset(self,):
    """Convert the dataset to a tensorflow dataset, easy to read from."""
    tf_dataset = {}
    for key in self._active_training_keys +\
        self._eval_metric_keys:
      tf_dataset[key] = []

    # Load the data from the data file. Note that most of the fields are
    # actually not one-hots, and essentially corresponds to the original data
    # with field-value pairs for each field, and the value is a discrete value.
    tf_actual_dataset = {}
    parsed_dataset = self.data_dict
    for p in parsed_dataset:
      tf_dataset[p] = parsed_dataset[p]
      tf_actual_dataset[p] =  tf.convert_to_tensor(tf_dataset[p])

      if key in self._active_training_keys:
        tf_actual_dataset[p] = tf.cast(tf_actual_dataset[p], tf.int32)

    self._design_space_dict_copy = deepcopy(self._design_space_dict)

    # Now convert the dataset to actually use one-hot representations. This is
    # used for training, and so it is important to use this.
    tf_actual_temp_dataset = {}
    for key in self._active_training_keys:
      design_space_map = dict(
          self._design_space_dict[key]['mapping_one_hot_to_value'])
      if self._design_space_dict[key]['data_type'] == 'discrete':
        data_val = tf_actual_dataset[key].numpy().astype(np.int32).tolist()
      else:
        data_val = tf_actual_dataset[key].numpy().astype(np.float32).tolist() 
      out_vals = []
      for x in data_val:
        out_vals.append(design_space_map[x])

      tf_actual_temp_dataset[key] = tf.constant(out_vals, dtype=tf.int32)

    ## Finally load the tf_actual_temp_dataset into the tf_dataset
    for key in tf_actual_temp_dataset:
      tf_actual_dataset[key] = tf_actual_temp_dataset[key]

    self._tf_dataset = tf_actual_dataset
    # self._infeasible_np = self._tf_dataset['infeasible'].numpy().astype(
    #     np.float32)
    self._top = self._tf_dataset['param_1'].shape[0]

  def load_or_refresh_config(self):
    """Load config file with specifications."""
    self._design_space_dict = {}
    self._segment_lengths = {}
    
    try:
      # The case when the config is a file to open
      with gfile.Open(self._config, 'r') as f:
        line = f.readline()
        line = line.replace('\n', '')
        # print ('Line: ', line)
        ctr = 0
        while line:
          ind_field = dict()
          split_line = line.split(':')
          ind_field['data_type'] = split_line[0]
          ind_field['value_range'] = [int(x) for x in split_line[-1].split(',')]
          index_vals = np.arange(len(ind_field['value_range']))
          ind_field['mapping_one_hot_to_value'] = zip(
              ind_field['value_range'], index_vals)
          ind_field['ctr'] = ctr
          self._design_space_dict[split_line[1]] = ind_field
          self._segment_lengths[ctr] = len(ind_field['value_range'])
          self._max_ctr += 1
          line = f.readline()
          ctr += 1
    except:
      # When config is a string of the contents of the file
      lines = self._config.split("\n")
      lines = [line.replace('\n', '') for line in lines]
      ctr = 0
      for line in lines:
        ind_field = dict()
        split_line = line.split(':')
        ind_field['data_type'] = split_line[0]
        if ind_field['data_type'] == 'discrete':
            ind_field['value_range'] = [int(x) for x in split_line[-1].split(',')]
        else:
            ind_field['value_range'] = [float(x) for x in split_line[-1].split(',')]
        index_vals = np.arange(len(ind_field['value_range']))
        ind_field['mapping_one_hot_to_value'] = zip(
            ind_field['value_range'], index_vals)
        ind_field['ctr'] = ctr
        self._design_space_dict[split_line[1]] = ind_field
        self._segment_lengths[ctr] = len(ind_field['value_range'])
        self._max_ctr += 1
        ctr += 1

    split_lengths = []
    for key in self._active_training_keys:
      split_lengths.append(
          self._segment_lengths[self._design_space_dict[key]['ctr']])
    total_length_split = 0
    
    if total_length_split > 0:
      split_lengths.append(total_length_split)
    self.split_lengths = split_lengths  # later used to split input when needed
    self.continuous_or_not = (total_length_split > 0)

  @property
  def size(self,):
    return self._top

  @property
  def input_properties(self):
    """Get the total length of the vector to be fed as input to the model."""
    length = 0
    for val in self._active_lengths:
      length += val
    return length

  
  def _get_batch(self, indices):
    """Sample a batch from the dataset."""
    all_train_elements = []  # this is the training elements in one-hot form
    all_test_elements = []  # this is the evaluation fields (area, runtime, score, etc)

    # Discrete training input keys
    for key in self._active_training_keys:
      all_train_elements.append(
              tf.one_hot(tf.gather(self._tf_dataset[key], indices),
                         depth=self._segment_lengths[
                                self._design_space_dict[key]['ctr']]))

    # Eval keys
    all_test_elements = tf.expand_dims(
        tf.gather(self._tf_dataset['score'], indices), 1)
    return tf.concat(all_train_elements, 1), all_test_elements

#@title Firefly Discrete Optimizer used after supervised training
class FireflyAlg():
  def __init__(self, initial_dataset:dict, config:dict, population=23, alpha=1.0, betamin=1.0, gamma=1.0, remainder=True, random_fireflies=True):
    self._config = config
    self.initial_dataset = initial_dataset
    self._active_training_keys = ['param_1', 'param_2', 'param_3',
                                  'param_4', 'param_5', 'param_6', 'param_7', 'param_8']
    self.population = population
    self.alpha = alpha
    self.betamin = betamin
    self.gamma = gamma
    self.remainder = remainder
    self.random_fireflies = random_fireflies

    self._setup_datset()
    if self.random_fireflies:
      self.fireflies = self.generate_fireflies(self.population)
    else:
      self.fireflies = self.get_best_fireflies()
  def _setup_datset(self,):
    """Construct a proper datset for the FireflyAlg object, needed for the conversion function below."""
    self._design_space_dict = {}
    self._segment_lengths = {}

    lines = self._config.split("\n")
    lines = [line.replace('\n', '') for line in lines]
    ctr = 0
    for line in lines:
      ind_field = dict()
      split_line = line.split(':')
      ind_field['data_type'] = split_line[0]
      ind_field['value_range'] = [int(x) for x in split_line[-1].split(',')]
      # index_vals = np.arange(len(ind_field['value_range']))
      ind_field['ctr'] = ctr
      self._design_space_dict[split_line[1]] = ind_field
      self._segment_lengths[ctr] = len(ind_field['value_range'])
      ctr += 1
    split_lenghts = []
    
    for key in self._active_training_keys:
      split_lenghts.append(
        self._segment_lengths[self._design_space_dict[key]['ctr']])
    total_length_split = 0
    if total_length_split > 0:
      split_lenghts.append(total_length_split)  
    self.split_lengths = split_lenghts
  def rand_bin_array(self, length, ones=1):
    arr = np.zeros(length)
    arr[:ones]  = 1
    np.random.shuffle(arr)
    return arr
  
  def get_best_fireflies(self, ):
    """Find the best desgins existing in the dataset with their scores"""
    fireflies = []
    scores = self.initial_dataset['accuracy']
    #maximize accuracy
    best_scores = np.argsort(-scores)[:self.population]

    #print the scores and the designs
    print('---Initial Designs and Scores/Runtimes-----')
    for idx, firefly in enumerate(best_scores):
      designs = []
      for key in self._active_training_keys:
        designs.append(self.initial_dataset[key][firefly])
      print('{} Score: {}'.format(idx+1, scores[firefly]))
      print('{} Design: {}'.format(idx+1, designs))
      designs = np.expand_dims(np.array(designs), axis=0)
      fireflies.append(designs)
    integer_fireflies = np.array(fireflies).squeeze(axis=1)
    return self.integer_2_onh_conv(integer_fireflies)
  def generate_fireflies(self, num_fireflies):
    """Initial M number of fireflies to random valid designs."""
    fireflies = []
    for _ in range(num_fireflies):
      onh_list = []
      for length in self.split_lengths:
        add_element = self.rand_bin_array(ones=1, length=length)
        add_element = np.expand_dims(add_element, axis=0)
        onh_list.append(add_element)
      random_design_np = np.concatenate(onh_list, axis=1).squeeze(axis=0)
      fireflies.append(random_design_np)
    return tf.convert_to_tensor(fireflies, dtype=tf.float32)   
  def run_inference(self, num_iters, model, mode_opt=True):
    """The actual implementation of the Firefly Algorithm."""
    if mode_opt:
      scores = self.initial_dataset['accuracy']
      indices = np.argsort(-scores)[:self.population]
      intensity_list = []
      for index in indices:
        intensity_list.append(scores[index])
      intensity = np.array(intensity_list)
      print('Intensity {}'.format(intensity))
    else: 
      print('---Initial Designs with predicted Scores/Runtimes-----')
      for single_firefly in self.fireflies:
        print('Design {}'.format(self.onh_2_integer_conv(np.expand_dims(single_firefly.numpy(), axis=0))))
      intensity = model(inputs=self.fireflies, training=False).numpy()
      print('Scores: {}'.format(intensity))
    
    best = np.max(intensity)
    best_arg = np.argmax(intensity)
    
    np_best_firefly = self.fireflies[best_arg].numpy()
    reshaped_best_firefly = np.expand_dims(np_best_firefly, axis=0)
    best_firefly = tf.convert_to_tensor(reshaped_best_firefly, dtype=tf.float32)    

    new_alpha = self.alpha
    int_fireflies = self.onh_2_integer_conv(self.fireflies.numpy()).numpy()
    intensity_count = 0
    best_count = 0 
    for iter_count in range(num_iters):
      new_alpha *= 0.97
      for i in range(self.population):
        for j in range(self.population):
          if i != j and intensity[i] <= intensity[j]:
            r = np.sum(np.square(int_fireflies[i] - int_fireflies[j]), axis=-1)
            beta = self.betamin * np.exp(-self.gamma * r)
            temp_firefly = self.single_param_serach(int_fireflies[i].copy(), int_fireflies[j].copy(), new_alpha, beta)
            new_firefly = self.integer_2_onh_conv(np.expand_dims(temp_firefly, axis=0))
            new_intensity = model(inputs=new_firefly, training=False).numpy()
            if new_intensity > intensity[i]:
              intensity_count += 1
              int_fireflies[i] = temp_firefly.copy()
              intensity[i] = new_intensity.copy()
              temp_best = best
              best = max(intensity[i], best)
              if best != temp_best:
                best_count += 1
                best_firefly = new_firefly.numpy().copy()
                print('Best firely: {}'.format(self.onh_2_integer_conv(best_firefly)))
                print('Best Score: {}'.format(best))
                print('Iter count: {}, i counter: {}'.format(iter_count, i))
                print('-----------------------------------------------------')
    self.fireflies = self.integer_2_onh_conv(int_fireflies)
    print('Intensity count: {}, Best count: {}'.format(intensity_count, best_count))
    return tf.convert_to_tensor(best_firefly, dtype=tf.float32)
  def onh_2_integer_conv(self, onh_fireflies):
    """Convert the given one-hot enconding with shape (N, 77) to integer encoding with shape (N, 10)."""
    integer_enc = []
    for i in range(onh_fireflies.shape[0]):
      index = 0
      integer_row = []
      for j in range(len(self.split_lengths)):
        arg_max = np.argmax(onh_fireflies[i][index:index + self.split_lengths[j]])
        index += self.split_lengths[j]
        res = self._design_space_dict[f'param_{j+1}']['value_range']
        integer_row.append(res[arg_max])
        if j == len(self.split_lengths) - 1:
          integer_enc.append(np.array(integer_row))
    return tf.convert_to_tensor(integer_enc, dtype=tf.float32)
  def integer_2_onh_conv(self, int_fireflies):
    """Convert the given integer enconding with shape (N, 10) to one-hot encoding with shape (N, 77)."""
    onh_enc = []
    for i in range(int_fireflies.shape[0]):
      onh_row = []
      for j in range(int_fireflies.shape[1]):
        value_range = self._design_space_dict[f'param_{j+1}']['value_range']
        indice = value_range.index(int_fireflies[i][j])
        onh_row.append(tf.one_hot(indice, depth=self.split_lengths[j]))
        if j == int_fireflies.shape[1] - 1:
          onh_row = tf.concat(onh_row, 0)
          onh_enc.append(onh_row)
    return tf.convert_to_tensor(onh_enc, dtype=tf.float32)
  def single_param_serach(self, i_firefly, j_firefly, new_alpha, beta):
    random_valid_design =  self.onh_2_integer_conv(self.generate_fireflies(num_fireflies=1).numpy()).numpy().squeeze(axis=0)
    for k in range(len(random_valid_design)):
      value_range = self._design_space_dict[f'param_{k+1}']['value_range']
      ub, lb = value_range[-1], value_range[0] #upper_bound, lower_bound
      scale = (ub - lb) 
      steps = new_alpha * (random_valid_design[k] - 0.5) * scale
      temp_calc = beta * (j_firefly[k] - i_firefly[k]) + steps
      temp_calc += i_firefly[k]
      if self.remainder:
        _remainder = temp_calc % value_range[-1]
        if temp_calc == 0 or _remainder != 0: #_remainder != 0
          distance = np.absolute(value_range - _remainder)
          arg_min = np.argmin(distance)
          valid_firefly = value_range[arg_min]
        else:
          valid_firefly = value_range[-1]
      else:
        distance = np.absolute(value_range - temp_calc)
        arg_min = np.argmin(distance)
        valid_firefly = value_range[arg_min]
      i_firefly[k] = valid_firefly
    return i_firefly

"""# Training loop and training"""
#@title Defining the function that runs training

def train_eval_offline(
    # Data flags
    config=None,
    training_dataset=None,
    validation_dataset=None,   
    # Train flags
    train_steps=int(1e6),
    summary_freq=1000,
    eval_freq=1000,
    # Train hparams
    add_summary=True,
    save_dir=None,
    loss_type='mse',
    layers=(512, 512, 512),
    opt_lr=1e-4,
    opt_betas=(0.9, 0.999),
    with_ranking_penalty=False,
    ranking_penalty_weight=0.1,
    batch_size=256,
    # params of the model
    use_dropout=False,
    num_votes=1,
    # PRIME parameters:
    cql_alpha=1.0,
    infeasible_alpha=1.0,
    # accelerated search bools
    enable_discrete_optimizer=False,
    skip_training=False):
  """Training loop for the PRIME model. 
  
  Most of the input arguments are primarily hyperparameters for training the
  PRIME model, and self explanatory. Other arguments explained below. 
  
  save_dir: the directory where the store the saved model, and the training
    summaries. Can be a string or None.
  training_dataset: a dictionary of fields in the training dataset, and their
    corresponding values used to train.
  validation_dataset: a dictionary of fields in the validation dataset, and
    their corresponding values to measure cross-validation. 
  """

  # First create the training dataset, note that the dataset below is a
  # dummy dataset, that is only well-suited for training as a representative
  # example. You can plug in the dataset from the other colab that provides
  # the data for training, or you can add your own dataset here.  
  
  params_dict = dict()
  params_dict['batch_size'] = batch_size
  params_dict['add_area_constraints'] = False
  # Defining the problem automatically does dataset loading
  train_problem = HardwareOptProblem(config,
                                    training_dataset, params_dict)
  
  # Now define the validation dataset (or val_problem)
  val_params_dict = dict()
  val_params_dict['batch_size'] = batch_size
  val_params_dict['add_area_constraints'] = False
  # Only validate on the valid samples in the validation dataset
  val_problem = HardwareOptProblem(config, validation_dataset,
                                   val_params_dict)

  # The dimensionality of each parameter. this input_splits parameter goes
  # into the PRIMETransformer, as it enables us to pass in inputd as a big
  # vector of concatenated one-hot vectors for each discrete parameter, and
  # then unpack it in the model training. This gives the flexibility of actually
  # being able to use the input one-hot vectors in any way as needed. 
  input_splits = train_problem.dataset.get_input_splits()
  print ('Input splits: ', input_splits)

  # Number of inputs in all: the total dimensionality of the input is given by
  # the sum of number of possible values each discrete parameter can take
  input_properties = train_problem.dataset.input_properties
  print ('Loaded validation dataset..', train_problem.dataset.size, 
          val_problem.dataset.size, input_properties)

  

  fwd_optimizer = tf.keras.optimizers.Adam(learning_rate=opt_lr,
                                           beta_1=opt_betas[0],
                                           beta_2=opt_betas[1], name='opt')

  training_dict = dict()
  training_dict['use_dropout'] = use_dropout
  training_dict['infeasible_alpha'] = infeasible_alpha
  training_dict['input_splits'] = input_splits
  training_dict['num_votes'] = num_votes
  training_dict['num_gradient_steps'] = 20

  model = PRIMETransformerModel(
        num_outputs=1,
        num_inputs=input_properties,
        optimizer=fwd_optimizer,
        layers=layers,
        penalty_weight=cql_alpha,
        negative_sampler=None,
        params_dict=training_dict)
  

  tf.summary.create_noop_writer()

  print ('save dir : ', save_dir)

  # Now start the training
  if skip_training:
    batch = train_problem.get_training_batch()
    #just to build the model
    _ = model.measure_stats(batch, batch_type='valid')
    model.load_weights(f'./results/{save_dir}_55000')
  else:
    avg_kendall_loss_list = dict()
    for step in range(train_steps):
      batch = train_problem.get_training_batch()
      # This is just to build the models.
      if step == 0:
        _ = model.measure_stats(batch, batch_type='valid')
      loss_dict = model.perform_training(
          batch, loss_type=loss_type,
          ranking_penalty_weight=ranking_penalty_weight)

      if step % summary_freq == 0:
        # regular logging
        print ('-------------------------------------------------------')
        for key in loss_dict:
          tf.summary.scalar('train/' + key, loss_dict[key], step=step)
          print ('Step: ', step, 'train/' + key, ':', loss_dict[key])
        print ('-------------------------------------------------------')

        if save_dir is not None:
          if step == 0:
            model.save(save_dir, overwrite=True)
          if step % 10000 == 0:
            model.save_weights(os.path.join(save_dir, "ckpt-"+str(step)), overwrite=True)
    
      if step % eval_freq == 0:
        val_batch = val_problem.get_valid_batch()
        # validation batches are only valid batches
        val_loss_dict = model.measure_stats(val_batch, batch_type='valid')
        print ('-------------------------------------------------------')
        for key in val_loss_dict:
          if key=='avg_kendall_loss':
            if step==0:
              avg_kendall_loss_list[key] = []
              avg_kendall_loss_list['step'] = []
            avg_kendall_loss_list[key].append(val_loss_dict[key])
            avg_kendall_loss_list['step'].append(step)
          tf.summary.scalar('val/' + key, val_loss_dict[key], step=step)
          print ('Step: ', step, 'val/' + key, ':', val_loss_dict[key])
        print ('-------------------------------------------------------')

    print ('============Finished Training============')
    if save_dir is not None:
      print('===========Saving weights================')
      model.save_weights(f'./saved_weights_dir_low_freq/{save_dir}_{step}', overwrite=True)
    print('===Avg kendall loss found during traing===')
    for step in range(len(avg_kendall_loss_list['step'])):
      print('Step: {}, val_avg_kendall_loss {}'.format(avg_kendall_loss_list['step'][step], avg_kendall_loss_list['avg_kendall_loss'][step]))
    print('==========================================')
    

  if enable_discrete_optimizer:
    print('Start Discerte Optimizer (Metaheuristic (Firelfy) Algorithm) for random designs')
    # initial_dataset = mergeDictionary(training_dataset, validation_dataset)
    discrete_optimizer = FireflyAlg(initial_dataset=None, config=config, population=25, remainder=True, random_fireflies=True)
    best_firefly = discrete_optimizer.run_inference(num_iters=int(1e3), model=model, mode_opt=False)
    print('---- The best firelfy found by the discrete optimizer is the following---')
    print('Configuration: {}'.format(discrete_optimizer.onh_2_integer_conv(best_firefly.numpy())))
    best_score = model(inputs=best_firefly, training=False)
    print('Score/Error: {}'.format(best_score))
    print('----Full list of optimized designs-----')
    print('{}'.format(discrete_optimizer.onh_2_integer_conv(discrete_optimizer.fireflies.numpy())))
    print('---Predicted scores/error for these accelerated designs-----')
    scores = model(inputs=discrete_optimizer.fireflies, training=False)
    print('{}'.format(scores))
    
    random_data = discrete_optimizer.onh_2_integer_conv(discrete_optimizer.fireflies.numpy()).numpy()
    random_dataset = pd.DataFrame({'param_1': random_data[:, 0], 'param_2': random_data[:, 1], 'param_3': random_data[:, 2], 
                        'param_4': random_data[:, 3], 'param_5': random_data[:, 4], 'param_6': random_data[:, 5], 
                        'param_7': random_data[:, 6], 'param_8': random_data[:, 7]})
    param_7_series = random_dataset['param_7'].squeeze()
    param_8_series = random_dataset['param_8'].squeeze()
    random_dataset['param_7'] = param_7_series.map({1: 0.0125, 2: 0.0225, 3: 0.0325, 4: 0.0425, 5: 0.0525, 6: 0.0625, 7: 0.0725, 8: 0.0825, 9: 0.0925})
    random_dataset['param_8'] = param_8_series.map({0: 0, 1: 0.00011, 2: 0.00023, 3: 0.00034, 4: 0.00045, 5: 0.00056, 6: 0.00068, 7: 0.00079, 9: 0.0009})
    random_dataset.to_csv('./results_dir_optimized_datasets/random_dataset_optimized_low_freq.csv')
    random_dataset.to_excel('./results_dir_optimized_datasets/random_dataset_optimized_low_freq.xlsx')

    print('Start Discerte Optimizer (Metaheuristic (Firelfy) Algorithm) for training_dataset designs')
    discrete_optimizer2 = FireflyAlg(initial_dataset=training_dataset, config=config, population=25, remainder=True, random_fireflies=False)
    best_firefly2 = discrete_optimizer2.run_inference(num_iters=int(1e3), model=model, mode_opt=False)
    print('---- For training dataset designs: The best firelfy found by the discrete optimizer is the following---')
    print('Configuration: {}'.format(discrete_optimizer2.onh_2_integer_conv(best_firefly2.numpy())))
    best_score2 = model(inputs=best_firefly2, training=False)
    print('Score/Error: {}'.format(best_score2))
    print('----Full list of optimized designs-----')
    print('{}'.format(discrete_optimizer2.onh_2_integer_conv(discrete_optimizer2.fireflies.numpy())))
    print('---Predicted scores/error for these accelerator designs-----')
    scores2 = model(inputs=discrete_optimizer2.fireflies, training=False)
    print('{}'.format(scores2))
    
    train_data = discrete_optimizer2.onh_2_integer_conv(discrete_optimizer2.fireflies.numpy()).numpy()
    train_dataset = pd.DataFrame({'param_1': train_data[:, 0], 'param_2': train_data[:, 1], 'param_3': train_data[:, 2], 
                        'param_4': train_data[:, 3], 'param_5': train_data[:, 4], 'param_6': train_data[:, 5], 
                        'param_7': train_data[:, 6], 'param_8': train_data[:, 7]})
    param_7_series = train_dataset['param_7'].squeeze()
    param_8_series = train_dataset['param_8'].squeeze()
    train_dataset['param_7'] = param_7_series.map({1: 0.0125, 2: 0.0225, 3: 0.0325, 4: 0.0425, 5: 0.0525, 6: 0.0625, 7: 0.0725, 8: 0.0825, 9: 0.0925})
    train_dataset['param_8'] = param_8_series.map({0: 0, 1: 0.00011, 2: 0.00023, 3: 0.00034, 4: 0.00045, 5: 0.00056, 6: 0.00068, 7: 0.00079, 9: 0.0009})
    train_dataset.to_csv('./results_dir_optimized_datasets/train_dataset_optimized_low_freq.csv')
    train_dataset.to_excel('./results_dir_optimized_datasets/train_dataset_optimized_low_freq.xlsx')
    
    print('Start Discerte Optimizer (Metaheuristic (Firelfy) Algorithm) for validation_dataset designs')
    discrete_optimizer3 = FireflyAlg(initial_dataset=validation_dataset, config=config, population=25, remainder=True, random_fireflies=False)
    best_firefly3 = discrete_optimizer3.run_inference(num_iters=int(1e3), model=model, mode_opt=False)
    print('---- For validation dataset: The best firelfy found by the discrete optimizer is the following---')
    print('Configuration: {}'.format(discrete_optimizer3.onh_2_integer_conv(best_firefly3.numpy())))
    best_score3 = model(inputs=best_firefly3, training=False)
    print('Score/Error: {}'.format(best_score3))
    print('----Full list of optimized designs-----')
    print('{}'.format(discrete_optimizer3.onh_2_integer_conv(discrete_optimizer3.fireflies.numpy())))
    print('---Predicted scores/error for these accelerator designs-----')
    scores3 = model(inputs=discrete_optimizer3.fireflies, training=False)
    print('{}'.format(scores3))
    
    val_data = discrete_optimizer3.onh_2_integer_conv(discrete_optimizer3.fireflies.numpy()).numpy()
    val_dataset = pd.DataFrame({'param_1': val_data[:, 0], 'param_2': val_data[:, 1], 'param_3': val_data[:, 2], 
                        'param_4': val_data[:, 3], 'param_5': val_data[:, 4], 'param_6': val_data[:, 5], 
                        'param_7': val_data[:, 6], 'param_8': val_data[:, 7]})
    
    param_7_series = val_dataset['param_7'].squeeze()
    param_8_series = val_dataset['param_8'].squeeze()
    val_dataset['param_7'] = param_7_series.map({1: 0.0125, 2: 0.0225, 3: 0.0325, 4: 0.0425, 5: 0.0525, 6: 0.0625, 7: 0.0725, 8: 0.0825, 9: 0.0925})
    val_dataset['param_8'] = param_8_series.map({0: 0, 1: 0.00011, 2: 0.00023, 3: 0.00034, 4: 0.00045, 5: 0.00056, 6: 0.00068, 7: 0.00079, 9: 0.0009})
    val_dataset.to_csv('./results_dir_optimized_datasets/val_dataset_optimized_low_freq.csv')
    val_dataset.to_excel('./results_dir_optimized_datasets/val_dataset_optimized_low_freq.xlsx')

config_str = """discrete:param_1:float64:true:25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54
discrete:param_2:float64:true:10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35
discrete:param_3:float64:true:25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53
discrete:param_4:float64:true:50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82
discrete:param_5:float64:true:10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29
discrete:param_6:float64:true:2,3,4,5,6,7,8
discrete:param_7:float64:true:1,2,3,4,5,6,7,8,9
discrete:param_8:float64:true:0,1,2,3,4,5,6,7,9"""


df = pd.read_csv(r'./final_dataset_low_freq.csv',
            index_col=0,
            names=["param_1", "param_2", "param_3", "param_4", "param_5", "param_6", "param_7", "param_8", "accuracy"])

# Drop first row by selecting all rows from first row onwards
df = df.iloc[1: , :]
df_actual = df.drop_duplicates()
df_sorted = df_actual.sort_values(by=["accuracy"])
print(df_sorted)

train_len = int(0.8 * (len(df_sorted) - 1))
df_train = df_sorted.iloc[: train_len, :]
df_valid = df_sorted.iloc[train_len + 1 : , :]

training_data = df_train.to_dict('list')
validation_data = df_valid.to_dict('list')

# Making all the parameters discrete
for key in training_data:
    training_data[key] = np.array(training_data[key], dtype=np.float32)
training_data['param_6'] = np.array(int(1e1)*training_data['param_6'], dtype=np.float32)
training_data['param_7'] = np.array(int(1e2)*training_data['param_7'], dtype=np.int32)
training_data['param_7'] = np.array(training_data['param_7'], dtype=np.float32)
training_data['param_8'] = np.array(int(1e4)*training_data['param_8'], dtype=np.int32)
training_data['param_8'] = np.array(training_data['param_8'], dtype=np.float32) 
# Making all the parameters discrete
for key in validation_data:
    validation_data[key] = np.array(validation_data[key], dtype=np.float32)
validation_data['param_6'] = np.array(int(1e1)*validation_data['param_6'], dtype=np.float32)
validation_data['param_7'] = np.array(int(1e2)*validation_data['param_7'], dtype=np.int32)
validation_data['param_7'] = np.array(validation_data['param_7'], dtype=np.float32)
validation_data['param_8'] = np.array(int(1e4)*validation_data['param_8'], dtype=np.int32)
validation_data['param_8'] = np.array(validation_data['param_8'], dtype=np.float32) 

unique_list = []
for key in training_data:
    if key != 'accuracy':
        unique_list.append(np.unique(training_data[f'{key}']))
print(unique_list)

print ('Keys in the dataset: ', training_data.keys())

 
train_eval_offline(
  config=config_str,
  training_dataset=training_data,
  validation_dataset=validation_data,
  train_steps=60001,
  summary_freq=250,
  eval_freq=500,
  add_summary=True,
  save_dir='./tragos_discrete_low_freq',
  loss_type='mse+rank',
  layers=(256, 256, 256),
  with_ranking_penalty=True,
  ranking_penalty_weight=0.1,
  batch_size=256,
  use_dropout=True,
  num_votes=7,
  cql_alpha=1.0,
  infeasible_alpha=1.0,
  enable_discrete_optimizer=True,
  skip_training=False
)