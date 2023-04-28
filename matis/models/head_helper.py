#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""ResNe(X)t Head helper."""

import torch
import torch.nn as nn

class TransformerBasicHead(nn.Module):
    """
    Instrument presence recognition head
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        cfg,
        dropout_rate=0.0,
        act_func="softmax",
    ):
        """
        Perform linear projection and activation as head for tranformers.
        Args:
            dim_in (int): the channel dimension of the input to the head.
            num_classes (int): the channel dimensions of the output to the head.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(TransformerBasicHead, self).__init__()
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

        self.projection = nn.Sequential(nn.Linear(dim_in, dim_in, bias=True),
                                        nn.ReLU(),
                                        nn.Linear(dim_in, num_classes, bias=True))

        self.always = cfg.TASKS.LOSS_FUNC[0]=='bce'
        self.cfg = cfg

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, x):
        
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.projection(x)

        if self.always or not self.training:
            x = self.act(x)
        return x


class TransformerRoIHead(nn.Module):
    """
    Instrument region classification head.
    """

    def __init__(
        self,
        cfg,
        num_classes,
        dropout_rate=0.0,
        act_func="softmax"
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        ResNetRoIHead takes p pathways as input where p in [1, infty].

        Args:
            num_classes (int): the channel dimensions of the p outputs to the
                ResNetHead.
            resolution (list): the list of spatial output size from the ROIAlign.
            scale_factor (list): the list of ratio to the input boxes by this
                number.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
            aligned (bool): if False, use the legacy implementation. If True,
                align the results more perfectly.
        """

        super(TransformerRoIHead, self).__init__()
        self.cfg = cfg
        
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)      
        
        dim_faster = 1024 
        input_dim = 256

        if cfg.MASKFORMER.TRANS:
            self.trans_project = nn.Sequential(nn.Linear(768,dim_faster,bias=True),
                                               nn.ReLU(),
                                               nn.Linear(dim_faster,dim_faster,bias=True))
            
            self.mlp = nn.Sequential(nn.Linear(input_dim, dim_faster, bias=True),
                                     nn.ReLU(),
                                     nn.Linear(dim_faster, dim_faster, bias=True),
                                     )

            dim_final = 2048
        
        else:
            self.mlp = nn.Sequential(nn.Linear(256, dim_faster, bias=False),
                                    nn.BatchNorm1d(dim_faster))

            dim_final = dim_faster + 768
        
        self.projection = nn.Sequential(nn.Linear(dim_final, num_classes, bias=True),)
                                        
        self.always = cfg.TASKS.LOSS_FUNC[-1]=='bce'

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation function.".format(act_func)
            )

    def forward(self, inputs, boxes_mask, bboxes, features=None):
        x = inputs
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        if self.cfg.MASKFORMER.TRANS:
            x = self.trans_project(x)

        x_boxes = x.unsqueeze(1).repeat(1,self.cfg.DATA.MAX_BBOXES,1)[boxes_mask]

        features = features[boxes_mask]
        features = self.mlp(features)
        if hasattr(self, "dropout"):
            features = self.dropout(features)

        x = torch.cat([x_boxes, features], dim=1)
        
        x = self.projection(x)
        
        if self.always or not self.training:
            x = self.act(x)

        return x