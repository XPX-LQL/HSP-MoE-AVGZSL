#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import glob
import shutil
import socket
import random
import itertools
import numpy as np
import multiprocessing
import configparser as cp
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score

import torch

np.random.seed(0)

def get_model_params(
    lr,
    reg_loss,
    dropout_encoder,
    dropout_decoder,
    additional_dropout,
    encoder_hidden_size,
    decoder_hidden_size,
    embeddings_batch_norm,
    rec_loss,
    cross_entropy_loss,
    transformer_use_embedding_net,
    transformer_dim,
    transformer_depth,
    transformer_heads,
    transformer_dim_head,
    transformer_mlp_dim,
    transformer_dropout,
    transformer_embedding_dim,
    transformer_embedding_time_len,
    transformer_embedding_dropout,
    transformer_embedding_time_embed_type,
    transformer_embedding_fourier_scale,
    transformer_embedding_embed_augment_position,
    lr_scheduler,
    optimizer,
    use_self_attention,
    use_cross_attention,
    transformer_average_features,
    audio_only,
    video_only,
    transformer_use_class_token,
    transformer_embedding_modality,
    modality,
    word_embeddings,
    av_fusion_module=False,
    av_fusion_dropout=0.1,
    semantic_consensus_routing=False,
    semantic_consensus_weight=0.0,
    semantic_consensus_temperature=1.0,
    semantic_distill_loss=False,
    semantic_distill_weight=0.0,
    semantic_distribution_weight=None,
    semantic_relation_weight=None,
    semantic_distill_mode='distribution',
    semantic_distill_temperature=2.0,
    semantic_hyperspherical=False,
    semantic_hyperspherical_temperature=1.0,
    semantic_radial_mode='none',
    semantic_radial_alpha=0.0,
    semantic_hup_loss=False,
    semantic_hup_weight=0.0,
    semantic_hup_mode='uniformity',
    semantic_hup_temperature=2.0,
    semantic_hup_margin=0.0,
    semantic_hup_cross_class_only=True
    ):

    params_model = dict()
    # Dimensions
    params_model['dim_out'] = 64
    params_model['cross_entropy_loss']=cross_entropy_loss

    # Optimizers' parameters
    params_model['lr'] = lr
    params_model['optimizer'] = optimizer
    if encoder_hidden_size==0:
        encoder_hidden_size=None
    if decoder_hidden_size==0:
        decoder_hidden_size=None



    params_model['additional_dropout']=additional_dropout
    params_model['reg_loss']=reg_loss
    params_model['dropout_encoder']=dropout_encoder
    params_model['dropout_decoder']=dropout_decoder
    params_model['encoder_hidden_size']=encoder_hidden_size
    params_model['decoder_hidden_size']=decoder_hidden_size

    # Model Sequence
    params_model['embeddings_batch_norm'] = embeddings_batch_norm
    params_model['rec_loss'] = rec_loss
    params_model['transformer_average_features'] = transformer_average_features
    params_model['transformer_use_embedding_net'] = transformer_use_embedding_net
    params_model['transformer_dim'] = transformer_dim
    params_model['transformer_depth'] = transformer_depth
    params_model['transformer_heads'] = transformer_heads
    params_model['transformer_dim_head'] = transformer_dim_head
    params_model['transformer_mlp_dim'] = transformer_mlp_dim
    params_model['transformer_dropout'] = transformer_dropout
    params_model['transformer_embedding_dim'] = transformer_embedding_dim
    params_model['transformer_embedding_time_len'] = transformer_embedding_time_len
    params_model['transformer_embedding_dropout'] = transformer_embedding_dropout
    params_model['transformer_embedding_time_embed_type'] = transformer_embedding_time_embed_type
    params_model['transformer_embedding_fourier_scale'] = transformer_embedding_fourier_scale
    params_model['transformer_embedding_embed_augment_position'] = transformer_embedding_embed_augment_position
    params_model['transformer_embedding_modality'] = transformer_embedding_modality
    params_model['transformer_attention_use_self_attention']=use_self_attention
    params_model['transformer_attention_use_cross_attention']=use_cross_attention
    params_model['audio_only'] = audio_only
    params_model['video_only'] = video_only
    params_model['transformer_use_class_token'] = transformer_use_class_token

    params_model['lr_scheduler'] = lr_scheduler


    params_model['modality'] = modality
    params_model['word_embeddings'] = word_embeddings
    params_model['av_fusion_module'] = av_fusion_module
    params_model['av_fusion_dropout'] = av_fusion_dropout
    params_model['semantic_consensus_routing'] = semantic_consensus_routing
    params_model['semantic_consensus_weight'] = semantic_consensus_weight
    params_model['semantic_consensus_temperature'] = semantic_consensus_temperature
    params_model['semantic_distill_loss'] = semantic_distill_loss
    params_model['semantic_distill_weight'] = semantic_distill_weight
    params_model['semantic_distribution_weight'] = semantic_distribution_weight
    params_model['semantic_relation_weight'] = semantic_relation_weight
    params_model['semantic_distill_mode'] = semantic_distill_mode
    params_model['semantic_distill_temperature'] = semantic_distill_temperature
    params_model['semantic_hyperspherical'] = semantic_hyperspherical
    params_model['semantic_hyperspherical_temperature'] = semantic_hyperspherical_temperature
    params_model['semantic_radial_mode'] = semantic_radial_mode
    params_model['semantic_radial_alpha'] = semantic_radial_alpha
    params_model['semantic_hup_loss'] = semantic_hup_loss
    params_model['semantic_hup_weight'] = semantic_hup_weight
    params_model['semantic_hup_mode'] = semantic_hup_mode
    params_model['semantic_hup_temperature'] = semantic_hup_temperature
    params_model['semantic_hup_margin'] = semantic_hup_margin
    params_model['semantic_hup_cross_class_only'] = semantic_hup_cross_class_only
    return params_model
