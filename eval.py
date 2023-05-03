import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')
from lightning import FastSimLightning
import dgl
import dgl.function as fn
from lightning import FastSimLightning
import sys
import os
import json
import torch 
import numpy as np 
from datasetloader import FastSimDataset, collate_graphs
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import os 
import uproot

import awkward as ak

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

config_path = sys.argv[1]
checkpoint_path = sys.argv[2]

with open(config_path, 'r') as fp:
    config = json.load(fp)

REDUCE_DS  = config['reduce_ds_test']

eval_path = 'pred_'+config['name']+'.root'


net = FastSimLightning(config)
checkpoint = torch.load(checkpoint_path,map_location=torch.device('cpu'))
net.load_state_dict(checkpoint['state_dict'])
net.eval()

dataset = FastSimDataset(config['path_test'],config,reduce_ds=config['reduce_ds_test'])

loader = DataLoader(dataset, batch_size=config['batchsize'],num_workers=config['num_workers'],shuffle=False, collate_fn=collate_graphs)

net.net.cuda()
net.cuda()
device = torch.device('cuda')

l_pt_tr, l_pt_pf, l_pt_fs = [], [], []
l_eta_tr, l_eta_pf, l_eta_fs = [], [], []
l_phi_tr, l_phi_pf, l_phi_fs = [], [], []

for g in tqdm(loader):

    g = g.to(device)
    g =  net.net.infer(g)

    g_list = dgl.unbatch(g)

    for i in range(len(g_list)):

        pt_tr = g_list[i].nodes['truth_particles'].data['features_0'][:,0]
        pt_tr = torch.exp(pt_tr * config['var_transform']['particle_pt']['std'] + config['var_transform']['particle_pt']['mean'])

        eta_tr = g_list[i].nodes['truth_particles'].data['features_0'][:,1]
        eta_tr = eta_tr * config['var_transform']['particle_eta']['std'] + config['var_transform']['particle_eta']['mean']

        phi_tr = g_list[i].nodes['truth_particles'].data['features_0'][:,2]
        phi_tr = phi_tr * config['var_transform']['particle_phi']['std'] + config['var_transform']['particle_phi']['mean']

        pt_fs = g_list[i].nodes['fastsim_particles'].data['pt_eta_phi_pred'][:,0]
        pt_fs = torch.exp(pt_fs * config['var_transform']['pflow_pt']['std'] + config['var_transform']['pflow_pt']['mean'])

        eta_fs = g_list[i].nodes['fastsim_particles'].data['pt_eta_phi_pred'][:,1]
        eta_fs = eta_fs * config['var_transform']['pflow_eta']['std'] + config['var_transform']['pflow_eta']['mean']

        phi_fs = g_list[i].nodes['fastsim_particles'].data['pt_eta_phi_pred'][:,2]
        phi_fs = phi_fs * config['var_transform']['pflow_phi']['std'] + config['var_transform']['pflow_phi']['mean']

        pt_pf = g_list[i].nodes['pflow_particles'].data['pt']
        pt_pf = torch.exp(pt_pf * config['var_transform']['pflow_pt']['std'] + config['var_transform']['pflow_pt']['mean'])

        eta_pf = g_list[i].nodes['pflow_particles'].data['eta']
        eta_pf = eta_pf * config['var_transform']['pflow_eta']['std'] + config['var_transform']['pflow_eta']['mean']

        phi_pf = g_list[i].nodes['pflow_particles'].data['phi']
        phi_pf = phi_pf * config['var_transform']['pflow_phi']['std'] + config['var_transform']['pflow_phi']['mean']

        l_pt_tr.extend(pt_tr.unsqueeze(0).cpu().tolist())
        l_eta_tr.extend(eta_tr.unsqueeze(0).cpu().tolist())
        l_phi_tr.extend(phi_tr.unsqueeze(0).cpu().tolist())
        l_pt_pf.extend(pt_pf.unsqueeze(0).cpu().tolist())
        l_eta_pf.extend(eta_pf.unsqueeze(0).cpu().tolist())
        l_phi_pf.extend(phi_pf.unsqueeze(0).cpu().tolist())
        l_pt_fs.extend(pt_fs.unsqueeze(0).cpu().tolist())
        l_eta_fs.extend(eta_fs.unsqueeze(0).cpu().tolist())
        l_phi_fs.extend(phi_fs.unsqueeze(0).cpu().tolist())

with uproot.recreate(eval_path) as file:

    pf_dict = {'pt': l_pt_pf, 'eta': l_eta_pf, 'phi': l_phi_pf}
    fs_dict = {'pt': l_pt_fs, 'eta': l_eta_fs, 'phi': l_phi_fs}
    tr_dict = {'pt': l_pt_tr, 'eta': l_eta_tr, 'phi': l_phi_tr}

    file['truth_tree'] = {
        'tr': ak.zip(tr_dict)
    }
    file['pflow_tree'] = {
        'pf': ak.zip(pf_dict)
    }

    file['fastsim_tree'] = {
        'fs': ak.zip(fs_dict)
    }
