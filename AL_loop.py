import os
import sys
sys.path.append('/home/ir-zani1/rds/rds-ukaea-ap001/ir-zani1/qualikiz/black_box_AL/bmdal_reg/')
from bmdal_reg.bmdal.feature_data import TensorFeatureData
from bmdal_reg.bmdal.algorithms import select_batch
from data_handling.preparation import get_data
from data_handling.datautils import train_keys_store
from models import Regressor
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import pickle as pkl


device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)

cfg = {'train_size':5000, 
       'valid_size':5000,
       'test_size': 5000,
       'fluxes':['efiitg_gb'], 
       'denormalise':False,
       'gkmodel':'QLK15D',
       'use_all_outputs':True,
       'use_classifier':False,
       'labels_unavailable':False}

cfg.update({'data':
            {
            'train': '/home/ir-zani1/rds/rds-ukaea-ap001/ir-zani1/qualikiz/Active_Continual_Learning/data/QLK15D/train_data_raw.pkl',
            'validation' : '/home/ir-zani1/rds/rds-ukaea-ap001/ir-zani1/qualikiz/Active_Continual_Learning/data/QLK15D/valid_data_raw.pkl',
            'test' : '/home/ir-zani1/rds/rds-ukaea-ap001/ir-zani1/qualikiz/Active_Continual_Learning/data/QLK15D/test_data_raw.pkl',
            'pool' : '/home/ir-zani1/rds/rds-ukaea-ap001/ir-zani1/qualikiz/Active_Continual_Learning/data/QLK15D/pool_data_raw.pkl'
            }
}
)

MAX_N = 10_000
BATCH = 5_000
NUM_MODELS = 2

print('Loading data...')
train, valid, test, pool, scaler = get_data(cfg)

input_keys = train_keys_store("QLK15D")

# train = train.data
# valid = valid.data
# test = test.data
# pool = pool.data

pool_idxs = pool.data.index
train_idxs = train.data.index

model_sizes = [2,4,6,8]
NTRAIN = np.linspace(BATCH, MAX_N, BATCH)
results_dict = {str(model_size):[] for model_size in model_sizes}
for model_size in model_sizes:
    print(f"model size={model_size}")
    while len(train)<MAX_N:
        print(f'Training with {len(train)} data points')
        losses = []
        # --- loop over number of models to get more statistics
        for _ in range(NUM_MODELS):
        # feature_data = {'train': X_train,
        #                 'pool': X_pool}
        # y_train = train['efiitg_gb']    
            n_models = 1
            n_features = 15

            model = Regressor(flux='efiitg_gb', inputs=n_features, model_size=model_size, scaler=scaler, device=device, dropout=0)
            train.set_output(flux='efiitg_gb')
            train_loader = DataLoader(
                train, batch_size=512, shuffle=True, pin_memory=True
            )
            valid.set_output(flux='efiitg_gb')
            valid_loader = DataLoader(
                valid, batch_size=len(valid), shuffle=False, pin_memory=True
            )
            test.set_output(flux='efiitg_gb')
            test_loader = DataLoader(
                test, batch_size=len(test), shuffle=False, pin_memory=True
            )

            model.fit(train_loader=train_loader, valid_loader=valid_loader, epochs=400, patience=1000, weight_decay=0)
            pred, average_loss, unscaled_avg_loss = model.predict(test_loader)
            losses.append(average_loss)

        results_dict[str(model_size)].append(np.mean(losses))

        # --- TensorFeatureData is needed for AL later on, keep it here even if it's unused so far
        train_data = TensorFeatureData(torch.tensor(train.data[input_keys].values, device=device))
        pool_data = TensorFeatureData(torch.tensor(pool.data[input_keys].values, device=device))
        y_train = torch.tensor(train.data['efiitg_gb'].values, device=device)
        #new_idxs = np.random.choice(pool_idxs, size=BATCH) # select_batch
        new_idxs, _ = select_batch(batch_size=BATCH, models=[model], 
                                data={'train': train_data, 'pool': pool_data}, y_train=y_train,
                                selection_method='random', sel_with_train=True,
                                base_kernel='grad', kernel_transforms=[('rp', [512])])      
        new_idxs = new_idxs.cpu().numpy()  
        to_append = pool.data.iloc[new_idxs,:]
        train.data = pd.concat([train.data, to_append], axis=0)
        pool.remove(new_idxs)
        # logical_new_idxs = torch.zeros(pool_idxs.shape[-1], dtype=torch.bool)
        # logical_new_idxs[new_idxs] = True
        # train_idxs = torch.cat([train_idxs, pool_idxs[logical_new_idxs]], dim=-1)
        # pool_idxs = pool_idxs[~logical_new_idxs]

with open('/home/ir-zani1/rds/rds-ukaea-ap001/ir-zani1/qualikiz/black_box_AL/results/random_sampling.pkl','wb') as f:
    pkl.dump(results_dict)


