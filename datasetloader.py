import uproot
import numpy as np
import pandas as pd
from tqdm import tqdm

import gc

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, Sampler
import itertools
import math

import dgl
import dgl.function as fn
import torch.nn.functional as F


def collate_graphs(samples):
    batched_graphs = dgl.batch(samples)
    return batched_graphs

class FastsimSampler(Sampler):
    def __init__(self, nevents, batch_size):

        super().__init__(nevents)

        self.batch_size = batch_size

        self.index_to_batch = {}
        self.n_replica = 100

        for i in range(nevents//self.n_replica):
            self.index_to_batch[i] = np.arange(i*self.n_replica,i*self.n_replica+batch_size)
        
        self.n_batches = nevents//self.n_replica

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        batch_order = np.random.permutation(np.arange(self.n_batches))
        for i in batch_order:
            yield self.index_to_batch[i]

class FastSimDataset(Dataset):
    def __init__(self, filename, config=None, reduce_ds=1.0):
        
        self.config = config

        self.f = uproot.open(filename)
        self.tree = self.f['fs_tree'] 

        self.var_transform = self.config['var_transform']

        self.nevents = min(self.tree.num_entries, self.tree.num_entries)

        if reduce_ds < 1.0 and reduce_ds > 0:
            self.nevents = int(self.nevents*reduce_ds)
        if reduce_ds >= 1.0:
            self.nevents = reduce_ds
        print(' we have ',self.nevents, ' events')
                

        self.init_label_dicts()
        self.init_variables_list()


        self.full_data_array = {}


        pdgids = self.tree['particle_pdgid'].array(library='np',entry_stop=self.nevents)
        mask = np.concatenate(pdgids)
        mask = np.array([self.class_labels[x] for x in  mask])
        mask = (mask == 2) + (mask == 3) + (mask == 4) # charged mask truth

        self.n_truth_particles = [] # charged
        for pdgid_single_event in pdgids:
            class_labels_single_event = np.array([self.class_labels[x] for x in  pdgid_single_event])
            n = (class_labels_single_event == 2) + (class_labels_single_event == 3) + (class_labels_single_event == 4)
            self.n_truth_particles.append(n.sum())

        for var in tqdm(self.truth_variables):
            self.full_data_array[var] = self.tree[var].array(library='np',entry_stop=self.nevents) 
            self.full_data_array[var] = np.concatenate(self.full_data_array[var])[mask]

            if var=='particle_pdgid':
                self.full_data_array['particle_class'] = [self.class_labels[x] for x in  self.full_data_array[var]]
       

        pflow_dummy = self.tree['pflow_px'].array(library='np',entry_stop=self.nevents)
        
        self.n_pflow_particles = [] # charged
        
        for pflow_single_event in pflow_dummy:
            self.n_pflow_particles.append(len(pflow_single_event))

        for var in tqdm(self.pflow_variables):
            self.full_data_array[var] = self.tree[var].array(library='np',entry_stop=self.nevents)
            self.full_data_array[var] = np.concatenate( self.full_data_array[var])

        self.full_data_array['pflow_pt'] = np.sqrt(self.full_data_array['pflow_px']**2+self.full_data_array['pflow_py']**2)
    
        # truth particle properties
        particle_phi   = np.arctan2(self.full_data_array['particle_py'], self.full_data_array['particle_px'])
        particle_p     = np.linalg.norm(np.column_stack([self.full_data_array['particle_px'],self.full_data_array['particle_py'],self.full_data_array['particle_pz']]),axis=1)
        particle_theta = np.arccos( self.full_data_array['particle_pz']/particle_p)
        particle_eta   =  -np.log( np.tan( particle_theta/2 )) 
        particle_pt = particle_p*np.sin(particle_theta)
        particle_pt = np.log(particle_pt)

        self.full_data_array['particle_phi']   = particle_phi
        self.full_data_array['particle_pt']    = particle_pt
        self.full_data_array['particle_theta'] = particle_theta
        self.full_data_array['particle_eta']   = particle_eta

        # transform variables and transform to tensors
        for var in tqdm(self.full_data_array.keys()):
            if var=='pflow_pt':
                self.full_data_array[var]=np.log(self.full_data_array[var])
 
            if ((var in self.var_transform)):
                self.full_data_array[var] = (self.full_data_array[var] - self.var_transform[var]['mean']) / self.var_transform[var]['std']
                
            self.full_data_array[var] = torch.tensor(self.full_data_array[var])
        
        self.truth_cumsum = np.cumsum([0]+self.n_truth_particles)
        self.pflow_cumsum = np.cumsum([0]+self.n_pflow_particles)


        del self.tree, particle_phi, particle_p, particle_theta, particle_eta, particle_pt
        gc.collect()

        print('done loading data')



    def get_single_item(self, idx):

        n_truth_particles = self.n_truth_particles[idx]
        n_pflow_particles = self.n_pflow_particles[idx]
        n_fastsim_particles = n_pflow_particles 

        truth_start, truth_end = self.truth_cumsum[idx], self.truth_cumsum[idx+1]
        pflow_start, pflow_end = self.pflow_cumsum[idx], self.pflow_cumsum[idx+1]
       
        truth_pt    = self.full_data_array['particle_pt'][truth_start:truth_end]
        truth_eta   = self.full_data_array['particle_eta'][truth_start:truth_end]
        truth_phi   = self.full_data_array['particle_phi'][truth_start:truth_end]
        truth_e     = self.full_data_array['particle_e'][truth_start:truth_end]
        truth_class = self.full_data_array['particle_class'][truth_start:truth_end].long()
        
        pflow_pt    = self.full_data_array['pflow_pt'][pflow_start:pflow_end]
        pflow_eta   = self.full_data_array['pflow_eta'][pflow_start:pflow_end]
        pflow_phi   = self.full_data_array['pflow_phi'][pflow_start:pflow_end]
        pflow_class = torch.zeros(len(pflow_phi)).long()

        num_nodes_dict = {
            'truth_particles' : n_truth_particles,
            'pflow_particles' : n_pflow_particles,
            'fastsim_particles' : n_fastsim_particles,
            'global_node' : 1
        }
        truth_to_truth_edge_start = torch.arange(n_truth_particles).repeat(n_truth_particles)
        truth_to_truth_edge_end   = torch.repeat_interleave(torch.arange(n_truth_particles), n_truth_particles) 

        truth_to_fastsim_edge_start = torch.arange(n_truth_particles).repeat(n_fastsim_particles)
        truth_to_fastsim_edge_end   = torch.repeat_interleave(torch.arange(n_fastsim_particles), n_truth_particles) 

        fastsim_to_truth_edge_start = torch.arange(n_fastsim_particles).repeat(n_truth_particles)
        fastsim_to_truth_edge_end   = torch.repeat_interleave(torch.arange(n_truth_particles), n_fastsim_particles) 

        pflow_to_fastsim_edge_start = torch.arange(n_pflow_particles).repeat(n_fastsim_particles)
        pflow_to_fastsim_edge_end   = torch.repeat_interleave(torch.arange(n_fastsim_particles), n_pflow_particles) 


        data_dict = {
            ('truth_particles','truth_to_truth','truth_particles') : (truth_to_truth_edge_start, truth_to_truth_edge_end),
            ('truth_particles','truth_to_global','global_node'): (torch.arange(n_truth_particles).int(),torch.zeros(n_truth_particles).int()),

            ('truth_particles','truth_to_fastsim','fastsim_particles') : (truth_to_fastsim_edge_start, truth_to_fastsim_edge_end),
            ('fastsim_particles','fastsim_to_truth','truth_particles') : (fastsim_to_truth_edge_start, fastsim_to_truth_edge_end),

            ('pflow_particles','pflow_to_fastsim','fastsim_particles') : (pflow_to_fastsim_edge_start, pflow_to_fastsim_edge_end)
        }
        g = dgl.heterograph(data_dict, num_nodes_dict)
        g.nodes['truth_particles'].data['idx']   = torch.arange(n_truth_particles) 
        g.nodes['truth_particles'].data['class'] = truth_class
        g.nodes['truth_particles'].data['pt']    = truth_pt.float()
        g.nodes['truth_particles'].data['eta']   = truth_eta.float()
        g.nodes['truth_particles'].data['phi']   = truth_phi.float()
        g.nodes['truth_particles'].data['e']     = truth_e.float()

        g.nodes['pflow_particles'].data['idx']   = torch.arange(n_pflow_particles) 
        g.nodes['pflow_particles'].data['class'] = pflow_class
        g.nodes['pflow_particles'].data['pt']    = pflow_pt.float()
        g.nodes['pflow_particles'].data['eta']   = pflow_eta.float()
        g.nodes['pflow_particles'].data['phi']   = pflow_phi.float()

        return g

        
    def __len__(self):
        return self.nevents 

    def __getitem__(self, idx):
        return self.get_single_item(idx)
        

    def init_label_dicts(self):        
        # photon : 0
        # neutral hadron: n, pion0, K0, Xi0, lambda: 1
        # charged hadron: p+-, K+-, pion+-, Xi+, Omega, Sigma : 2
        # electron : 3
        # muon : 4

        self.class_labels = {
            -3112 : 2,
            3112  : 2,
            3222  : 2,
            -3222 : 2,
            -3334 : 2,
            3334  : 2,
            -3122 : 1,
            3122  : 1,
            310   : 1,
            3312  : 2,
            -3312 : 2,
            3322  : 1,
            -3322 : 1,
            2112  : 1,
            321   : 2,
            130   : 1,
            -2112 : 1,
            2212  : 2,
            11    : 3,
            -211  : 2,
            13    : 4,
            211   : 2,
            -13   : 4,
            -11   : 3,
            22    : 0,
            -2212 : 2,
            -321  : 2
        } 


    def init_variables_list(self):

        self.truth_variables = [
            'particle_px', 'particle_py', 'particle_pz', 'particle_e', 'particle_pdgid'
        ]
            
        self.pflow_variables = [
            'pflow_px','pflow_py', 'pflow_eta', 'pflow_phi'
        ]