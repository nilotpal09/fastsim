import torch
import numpy as np
import torch.nn as nn

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.optimize import linear_sum_assignment


from copy import deepcopy

class Set2SetLoss(nn.Module):

    def __init__(self, config=None):
    
        super().__init__()
        self.config = config
        self.var_transform = self.config["var_transform"]

        self.regression_loss = nn.MSELoss(reduction='none')
        self.num_loss        = nn.CrossEntropyLoss(reduction='mean')


    def forward(self, g, scatter=True):

        # matching 
        matrix = [[0]*g.batch_size for i in range(g.batch_size)]
        matrix_pt = np.zeros((g.batch_size,g.batch_size))
        matrix_eta = np.zeros((g.batch_size,g.batch_size))
        matrix_phi = np.zeros((g.batch_size,g.batch_size))

        n_pflow_particles = g.num_nodes(ntype='pflow_particles')//g.batch_size
        n_fastsim_particles = g.num_nodes(ntype='fastsim_particles')//g.batch_size
        
        pred_pt_list = []

        for i in range(g.batch_size):
            pred_pt_tmp = []

            for j in range(g.batch_size):

                target_pt = g.nodes['pflow_particles'].data['pt'][i*n_pflow_particles:(i+1)*n_pflow_particles]
                input_pt = g.nodes['fastsim_particles'].data['pt_eta_phi_pred'][j*n_pflow_particles:(j+1)*n_pflow_particles,0]

                target_eta = g.nodes['pflow_particles'].data['eta'][i*n_pflow_particles:(i+1)*n_pflow_particles]
                input_eta = g.nodes['fastsim_particles'].data['pt_eta_phi_pred'][j*n_pflow_particles:(j+1)*n_pflow_particles,1]

                target_phi = g.nodes['pflow_particles'].data['phi'][i*n_pflow_particles:(i+1)*n_pflow_particles] * self.var_transform["pflow_phi"]["std"]  + self.var_transform["pflow_phi"]["mean"]
                input_phi = g.nodes['fastsim_particles'].data['pt_eta_phi_pred'][j*n_pflow_particles:(j+1)*n_pflow_particles,2] * self.var_transform["pflow_phi"]["std"]  + self.var_transform["pflow_phi"]["mean"]

                pt_loss = self.regression_loss(
                    input_pt.unsqueeze(1).unsqueeze(0).unsqueeze(1).expand(-1,target_pt.size(0),-1,-1),
                    target_pt.unsqueeze(1).unsqueeze(0).unsqueeze(2).expand(-1,-1,input_pt.size(0),-1)).mean(3)

                eta_loss = self.regression_loss(
                    input_eta.unsqueeze(1).unsqueeze(0).unsqueeze(1).expand(-1,target_pt.size(0),-1,-1),
                    target_eta.unsqueeze(1).unsqueeze(0).unsqueeze(2).expand(-1,-1,input_pt.size(0),-1)).mean(3)

                phi_loss = 1-torch.cos(target_phi.unsqueeze(1).unsqueeze(0).unsqueeze(2).expand(-1,-1,input_pt.size(0),-1)
                                        - input_phi.unsqueeze(1).unsqueeze(0).unsqueeze(1).expand(-1,target_pt.size(0),-1,-1)).mean(3)

                loss = pt_loss + eta_loss + phi_loss

                loss_ = loss.detach().cpu().numpy()

                indices = np.array([linear_sum_assignment(p) for p in loss_])

                pred_pt_tmp.append(input_pt[indices[:,1][0]])

                indices = indices.shape[2] * indices[:,0] + indices[:,1]
                
                fin_loss     = torch.gather(loss.flatten(1,2),1,torch.from_numpy(indices).to(device=loss.device)).mean(1)
                pt_loss_fin  = torch.gather(pt_loss.flatten(1,2),1,torch.from_numpy(indices).to(device=loss.device)).mean(1)
                eta_loss_fin = torch.gather(eta_loss.flatten(1,2),1,torch.from_numpy(indices).to(device=loss.device)).mean(1)
                phi_loss_fin = torch.gather(phi_loss.flatten(1,2),1,torch.from_numpy(indices).to(device=loss.device)).mean(1)

                matrix[i][j] = fin_loss
                

                matrix_pt[i,j] = pt_loss_fin
                matrix_eta[i,j] = eta_loss_fin
                matrix_phi[i,j] = phi_loss_fin

                
            pred_pt_list.append(pred_pt_tmp)


        matrix_matching = csr_matrix(torch.tensor(matrix))
       
        pflow_idx, fastsim_idx = min_weight_full_bipartite_matching(matrix_matching)

        
        
        pt_loss = 0
        eta_loss = 0
        phi_loss = 0
        kin_loss = 0

        for i in range(g.batch_size):

            kin_loss += matrix[pflow_idx[i]][fastsim_idx[i]]/g.batch_size

            pt_loss += matrix_pt[pflow_idx[i],fastsim_idx[i]]/g.batch_size
            eta_loss += matrix_eta[pflow_idx[i],fastsim_idx[i]]/g.batch_size
            phi_loss += matrix_phi[pflow_idx[i],fastsim_idx[i]]/g.batch_size

        # set size loss
        num_loss = self.num_loss(g.nodes['global_node'].data['set_size_pred'], g.batch_num_nodes('pflow_particles')).mean()
        total_loss = kin_loss + num_loss 

        ##### mmd+ha loss for metric tracking ---------->

        # pred_pt = g.nodes['fastsim_particles'].data['pt_eta_phi_pred'][:,0].reshape(g.batch_size,n_fastsim_particles)
        # pred_eta = g.nodes['fastsim_particles'].data['pt_eta_phi_pred'][:,1].reshape(g.batch_size,n_fastsim_particles)
        # pred_phi = g.nodes['fastsim_particles'].data['pt_eta_phi_pred'][:,2].reshape(g.batch_size,n_fastsim_particles) * self.var_transform["pflow_phi"]["std"]  + self.var_transform["pflow_phi"]["mean"]
        # targ_pt = g.nodes['pflow_particles'].data['pt'].reshape(g.batch_size,n_pflow_particles)
        # targ_eta = g.nodes['pflow_particles'].data['eta'].reshape(g.batch_size,n_pflow_particles)
        # targ_phi = g.nodes['pflow_particles'].data['phi'].reshape(g.batch_size,n_pflow_particles) * self.var_transform["pflow_phi"]["std"]  + self.var_transform["pflow_phi"]["mean"]

        # predic = torch.stack([pred_pt,pred_eta,pred_phi],axis=1)
        # target = torch.stack([targ_pt,targ_eta,targ_phi],axis=1)

        # mmd_HA_loss = self.torch_new_MMD(target,predic,HA=True)
        # mmd_PA_loss = self.torch_new_MMD(target,predic,HA=False)

        ############# <------------------



        
        if scatter == True:
            pred_pteaphi = g.nodes['fastsim_particles'].data['pt_eta_phi_pred']
            target_ptephi = torch.stack([
                        g.nodes['pflow_particles'].data['pt'], 
                        g.nodes['pflow_particles'].data['eta'], 
                        g.nodes['pflow_particles'].data['phi']
                    ],dim=1)
            target_ptetaphi_copy, pred_ptetaphi_copy = deepcopy(target_ptephi.cpu().data), deepcopy(pred_pteaphi.cpu().data)
            target_ptetaphi_copy, pred_ptetaphi_copy = self.undo_scalling(target_ptetaphi_copy), self.undo_scalling(pred_ptetaphi_copy)
            ptetaphi = [target_ptetaphi_copy, pred_ptetaphi_copy]

            pred_set_size = np.argmax(g.nodes['global_node'].data['set_size_pred'].detach().cpu().numpy(), axis=1)
            set_sizes = [g.batch_num_nodes('pflow_particles').detach().cpu().numpy(), pred_set_size]

        return  {
            'total_loss':total_loss , 'kin_loss': kin_loss,
            'pt_loss': pt_loss,
            'eta_loss': eta_loss,
            'phi_loss': phi_loss,
            'num_loss': num_loss.detach(),           
            'ptetaphi': ptetaphi, 'set_size': set_sizes, 
            # 'mmd_HA_loss': mmd_HA_loss,
            # 'mmd_PA_loss': mmd_PA_loss,
        }


    def undo_scalling(self, inp):
        pt_mean,  pt_std  = self.var_transform['pflow_pt']['mean'], self.var_transform['pflow_pt']['std']
        eta_mean, eta_std = self.var_transform['pflow_eta']['mean'], self.var_transform['pflow_eta']['std']
        phi_mean, phi_std = self.var_transform['pflow_phi']['mean'], self.var_transform['pflow_phi']['std']

        inp[:,0] = inp[:,0]*pt_std + pt_mean
        inp[:,1] = inp[:,1]*eta_std + eta_mean
        inp[:,2] = inp[:,2]*phi_std + phi_mean

        for i in range(len(inp[:,2])):
            while (inp[i,2]>np.pi):
                inp[i,2]=inp[i,2] - 2*np.pi
            while (inp[i,2]<-np.pi):
                inp[i,2]=inp[i,2] + 2*np.pi

        return inp


    # mmd loss as metric
    def torch_compute_kernel_matrix3(self,A,B):
        N = A.shape[-1]
        Ab = torch.einsum('bai,j->bija',A,torch.ones(N).to(A.device))
        Bb = torch.einsum('i,baj->bija',torch.ones(N).to(B.device),B)
        mse_two_feats = torch.exp(-torch.square(Ab[...,:2]-Bb[...,:2]).sum(axis=-1))
        cos_deltaphi = torch.exp(-(1-torch.cos(Ab[...,2]-Bb[...,2])))
        return mse_two_feats + cos_deltaphi
    

    def torch_new_set_kernel(self,A,B,HA):
        if HA == True:
            M = self.torch_compute_kernel_matrix3(A,B)
            M_ = M.detach().cpu().numpy()
            
            indices = np.array([linear_sum_assignment(p) for p in M_])
            indices = indices.shape[2] * indices[:, 0] + indices[:, 1]
            
            f_loss = torch.gather(M.flatten(1,2),1,torch.from_numpy(indices).to(device=M.device)).mean(1)            
            return f_loss

        else:
            ATA = self.torch_compute_kernel_matrix3(A,A)
            BTB = self.torch_compute_kernel_matrix3(B,B)
            DA,UA = torch.linalg.eig(ATA)
            DB,UB = torch.linalg.eig(BTB)
            DA = DA.float()
            UA = UA.float()
            DB = DB.float()
            UB = UB.float()
            M = self.torch_compute_kernel_matrix3(A,B)
            X = torch.einsum('bi,bji,bjk,bkl,bl->bil',torch.sqrt(1/DA),UA,M,UB,torch.sqrt(1/DB))
            _,S,_ = torch.linalg.svd(X)
            return torch.square(S).prod(axis=-1)

    def torch_prod_kernels(self,A,B,HA):
        n_part = A.size()[2]
        Ab = torch.tile(A.clone().detach().requires_grad_(True).unsqueeze(0),(len(A),1,1,1))
        Bb = torch.tile(B.clone().detach().requires_grad_(True).unsqueeze(1),(1,len(A),1,1))
        return self.torch_new_set_kernel(Ab.reshape(-1,3,n_part),Bb.reshape(-1,3,n_part),HA).reshape(len(A),-1)

    def torch_new_MMD(self,D1,D2,HA=True):
        mask = (1-torch.eye(len(D1))).bool()
        XX = self.torch_prod_kernels(D1,D1,HA)[mask]
        YY = self.torch_prod_kernels(D2,D2,HA)[mask]
        XY = self.torch_prod_kernels(D1,D2,HA)
        return torch.mean(XX) + torch.mean(YY) - 2*torch.mean(XY)
