import comet_ml
from pytorch_lightning.core.module import LightningModule
import torch
from torch.utils.data import DataLoader

import sys
sys.path.append('./models/')

from fastsim_model import FastSimModel
from loss_set2set import Set2SetLoss

from datasetloader import FastsimSampler, FastSimDataset, collate_graphs
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import gaussian_kde


class FastSimLightning(LightningModule):

    def __init__(self, config, comet_exp=None):
        super().__init__()
        torch.manual_seed(1)
        self.config = config
        self.net = FastSimModel(self.config)

        if config['model_type'] == 'tspn':
            self.loss = Set2SetLoss(config)

        self.comet_exp = comet_exp


    def set_comet_exp(self, comet_exp):
        self.comet_exp = comet_exp


    def forward(self, g):
        return self.net(g)

    
    def training_step(self, batch, batch_idx):
        
        g = batch
        self(g)
        
        losses = self.loss(g)

        for loss_type in self.config['loss_types']:
            self.log('train/'+loss_type, losses[loss_type],batch_size=g.batch_size)
 
        return losses['total_loss']
        


    def validation_step(self, batch, batch_idx):  
        g = batch
        self(g)
        losses = self.loss(g)
        return_dict = {
            'val_loss'     : losses['total_loss'],
            'plot_data': {
                'ptetaphi': losses['ptetaphi'],
                'set_size': losses['set_size'],
            },
        }

        for loss_type in self.config['loss_types']:
            self.log('val/'+loss_type, losses[loss_type],batch_size=g.batch_size)
            return_dict[loss_type] = losses[loss_type]
 
        return return_dict

    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config['learningrate'])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,100], gamma=0.5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def train_dataloader(self):
        if 'reduce_ds_train' in self.config:
            reduce_ds = self.config['reduce_ds_train']
        else:
            reduce_ds = 1

        dataset = FastSimDataset(self.config['path_train'], self.config, reduce_ds=reduce_ds)
        batch_sampler = FastsimSampler(len(dataset), batch_size=self.config['batchsize'])
        loader = DataLoader(dataset, num_workers=self.config['num_workers'], 
                             batch_sampler=batch_sampler, collate_fn=collate_graphs, pin_memory=False)
        return loader

    
    def val_dataloader(self):
        if 'reduce_ds_val' in self.config:
            reduce_ds = self.config['reduce_ds_val']
        else:
            reduce_ds = 1
        
        dataset = FastSimDataset(self.config['path_valid'], self.config, reduce_ds=reduce_ds)
        batch_sampler = FastsimSampler(len(dataset), batch_size=self.config['batchsize'])
        loader = DataLoader(dataset, num_workers=self.config['num_workers'], 
                             batch_sampler=batch_sampler, collate_fn=collate_graphs, pin_memory=False)
        return loader


    def validation_epoch_end(self, outputs):
        
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val/avg-loss epoch', avg_loss)

        if plt.get_fignums():
            plt.clf()
            fig = plt.gcf()
        else:
            fig = plt.figure(figsize=(15, 10), dpi=100, tight_layout=True)
        canvas = FigureCanvas(fig) 


        # cardinality plot
        set_size_target = np.hstack([x['plot_data']['set_size'][0] for x in outputs])
        set_size_pred   = np.hstack([x['plot_data']['set_size'][1] for x in outputs])

        ax1 = fig.add_subplot(2, 3, 1)
        self.fancy_scatter(fig, ax1, set_size_target, set_size_pred)
        ax1.set_title('Cardinality')
        ax1.set_xlabel('PFlow'); ax1.set_ylabel('FastSim')

        # pt eta phi plots
        ptetaphi_target = np.vstack([np.vstack(x['plot_data']['ptetaphi'][0]) for x in outputs])
        ptetaphi_pred   = np.vstack([np.vstack(x['plot_data']['ptetaphi'][1]) for x in outputs])

        ax2 = fig.add_subplot(2, 3, 2)
        self.fancy_scatter(fig, ax2, ptetaphi_target[:,0], ptetaphi_pred[:,0])
        ax2.set_title('log(pT)')
        ax2.set_xlabel('PFlow'); ax2.set_ylabel('FastSim')

        ax3 = fig.add_subplot(2, 3, 3)
        self.fancy_scatter(fig, ax3, ptetaphi_target[:,1], ptetaphi_pred[:,1])
        ax3.set_title('eta')
        ax3.set_xlabel('PFlow'); ax3.set_ylabel('FastSim')

        ax4 = fig.add_subplot(2, 3, 4)
        self.fancy_scatter(fig, ax4, ptetaphi_target[:,2], ptetaphi_pred[:,2])
        ax4.set_title('phi')
        ax4.set_xlabel('PFlow'); ax4.set_ylabel('FastSim')

        canvas.draw()
        w, h = fig.get_size_inches() * fig.get_dpi()
        image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(h), int(w), 3)

        if self.comet_exp is not None:
            self.comet_exp.log_image(
                image_data=image,
                name='truth vs reco scatter',
                overwrite=False, 
                image_format="png",
            )
        else:
            plt.savefig('scatter.png')


    def fancy_scatter(self, fig, ax, x, y):
        xy = np.vstack([x, y])
        try:
            z = gaussian_kde(xy)(xy)
            idx = z.argsort()
            x, y, z = x[idx], y[idx], z[idx]
            im = ax.scatter(x, y, s=3, c=z, cmap="cool")
        except:
            im = ax.scatter(x, y, s=3)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')
