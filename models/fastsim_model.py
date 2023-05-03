import dgl
import dgl.function as fn
import torch.nn.functional as F
from dgl import DGLGraph

import numpy as np
import torch
import torch.nn as nn

from models.mpnn import MPNN
from models.tspn import TSPN

import json

class FastSimModel(nn.Module):

    def __init__(self,config):
        super().__init__()

        self.config = config

        self.class_embedding_net = nn.Embedding(5, config['class_embedding_size'])

        self.encoder   = MPNN(config)
        self.predict_set_size = self.config['predict_set_size']

        if config['model_type'] == 'tspn':
            self.outputnet = TSPN(config)


    def forward(self, g):
        self.init_feat(g)
        self.encoder(g)
        self.outputnet(g)

        return g


    def init_feat(self, g):

        g.nodes['truth_particles'].data['features_0'] = torch.cat([
            g.nodes['truth_particles'].data['pt'].view(-1,1),
            g.nodes['truth_particles'].data['eta'].view(-1,1),
            g.nodes['truth_particles'].data['phi'].view(-1,1),
            self.class_embedding_net(g.nodes['truth_particles'].data['class'])
        ], dim=1)
        g.nodes['truth_particles'].data['features'] = g.nodes['truth_particles'].data['features_0']
        
        return g


    def infer(self,g):
        
        self.init_feat(g)
        self.encoder(g)
        g = self.outputnet.infer(g)
        
        return g 
