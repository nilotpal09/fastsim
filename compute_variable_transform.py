import uproot
import numpy as np
import sys
import json
from tqdm import tqdm

import matplotlib.pyplot as plt

def transform_phi(phi):
    for i in range(len(phi)):
        while (phi[i]>np.pi):
            phi[i]=phi[i] - 2*np.pi
        while (phi[i]<-np.pi):
            phi[i]=phi[i] + 2*np.pi
    return phi

config_path = sys.argv[1]
with open(config_path, 'r') as fp:
     config = json.load(fp)


var_to_transform = [
    "particle_pt", "particle_eta", "particle_phi",
    "pflow_pt", "pflow_eta", "pflow_phi"
]

truth_variables = ['particle_px', 'particle_py', 'particle_pz', 'particle_e', 'particle_pdgid']
pflow_variables = ['pflow_px','pflow_py', 'pflow_eta', 'pflow_phi']

class_labels = {
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

filename = config['path_train']

f = uproot.open(filename)
tree = f['fs_tree']

n_events = tree.num_entries

print(' we have', n_events)
        
full_data_array = {}

pdgids = tree['particle_pdgid'].array(library='np',entry_stop=n_events)
mask = np.concatenate(pdgids)
mask = np.array([class_labels[x] for x in  mask])
mask = (mask == 2) + (mask == 3) + (mask == 4) # charged mask truth

for var in tqdm(truth_variables):
    full_data_array[var] = tree[var].array(library='np',entry_stop=n_events)
    full_data_array[var] = np.concatenate( full_data_array[var] )[mask]

for var in tqdm(pflow_variables):
    full_data_array[var] = tree[var].array(library='np',entry_stop=n_events)
    full_data_array[var] = np.concatenate( full_data_array[var] )

full_data_array['pflow_pt']=np.log(np.sqrt(full_data_array['pflow_px']**2+full_data_array['pflow_py']**2))
full_data_array['pflow_phi']=transform_phi(full_data_array['pflow_phi'])

# truth particle properties
particle_phi   = np.arctan2(full_data_array['particle_py'], full_data_array['particle_px'])
particle_p     = np.linalg.norm(np.column_stack([full_data_array['particle_px'],full_data_array['particle_py'],full_data_array['particle_pz']]),axis=1)
particle_theta = np.arccos( full_data_array['particle_pz']/particle_p)
particle_eta   =  -np.log( np.tan( particle_theta/2 )) 
particle_pt = particle_p*np.sin(particle_theta)
particle_pt = np.log(particle_pt)

full_data_array['particle_phi']   = transform_phi(particle_phi)
full_data_array['particle_pt']    = particle_pt
full_data_array['particle_theta'] = particle_theta
full_data_array['particle_eta']   = particle_eta

for var in var_to_transform:

    print('"{}": {{"mean": {:.5f}, "std": {:.5f}}},'.format(
        var, full_data_array[var].mean(), full_data_array[var].std())
    )

print('"{}": {{"mean": {:.5f}, "std": {:.5f}}},'.format(
        'pt', (full_data_array['pflow_pt'].mean()+full_data_array['particle_pt'].mean())/2, (full_data_array['pflow_pt'].std()+full_data_array['particle_pt'].std())/2)
    )
print('"{}": {{"mean": {:.5f}, "std": {:.5f}}},'.format(
        'eta', (full_data_array['pflow_eta'].mean()+full_data_array['particle_eta'].mean())/2, (full_data_array['pflow_eta'].std()+full_data_array['particle_eta'].std())/2)
    )
print('"{}": {{"mean": {:.5f}, "std": {:.5f}}},'.format(
        'phi', (full_data_array['pflow_phi'].mean()+full_data_array['particle_phi'].mean())/2, (full_data_array['pflow_phi'].std()+full_data_array['particle_phi'].std())/2)
    )
