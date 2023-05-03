import comet_ml
import sys
paths = sys.path
for p in paths:
     if '.local' in p:
             paths.remove(p)

from lightning import FastSimLightning
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning import Trainer

import json
import glob
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
num_dev = 1

if __name__ == "__main__":
    config_path = sys.argv[1]

    if len(sys.argv) == 3:
        debug_mode = sys.argv[2]
    else:
        debug_mode = '0'

    with open(config_path, 'r') as fp:
         config = json.load(fp)

    net = FastSimLightning(config)

    if debug_mode == '1':
        trainer = Trainer(
            max_epochs=config['num_epochs'],
            accelerator='gpu',
            devices = num_dev,
            default_root_dir='checkpoint_dir'
        )

    else:
        comet_logger = CometLogger(
            api_key='your_apikey',
            save_dir='checkpoint_dir',
            project_name="fastsim", 
            experiment_name=config['name']
        )

        net.set_comet_exp(comet_logger.experiment)
        comet_logger.experiment.log_asset(config_path,file_name='config')

        all_files = glob.glob('./*.py')+glob.glob('models/*.py')
        for fpath in all_files:
            comet_logger.experiment.log_asset(fpath)

        trainer = Trainer(
            max_epochs = config['num_epochs'],
            accelerator='gpu',
            devices = num_dev,
            default_root_dir = 'checkpoint_dir',
            logger = comet_logger
            )
    
    trainer.fit(net)



