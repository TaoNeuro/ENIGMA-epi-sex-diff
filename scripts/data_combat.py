import os
import numpy as np
import pandas as pd

import sys
sys.path.append('..')

from helper.func import data_combat

project_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), '..', '..'))
data_dir = os.path.join(project_dir, '...')
results_dir = os.path.join(project_dir, '...')

demo = pd.read_csv(os.path.join(data_dir, 'demo_combined.csv'), index_col=0)
ctx = pd.read_csv(os.path.join(data_dir, 'ctx_combined.csv'), index_col=0)
sv = pd.read_csv(os.path.join(data_dir, 'sv_combined.csv'), index_col=0)

sites_tle = np.load(os.path.join(data_dir, 'sites_tle.npy'), allow_pickle=True)
sites_ige = np.load(os.path.join(data_dir, 'sites_ige.npy'), allow_pickle=True)

## Combat harmonization
group_types = ['tle', 'ige']
metric_dfs = {'ctx': ctx, 'sv': sv}

for group in group_types:

    combat_model = data_combat(demo, group_type=group, sites_tle=sites_tle, sites_ige=sites_ige)

    for key, df in metric_dfs.items():
        
        covs = combat_model.prepare_covariates(site='Site', age='Age', sex='Sex', group='Dx')
        combat_result = combat_model.harmonize_data(df, covs)

        # save the data
        output_dir = os.path.join(results_dir, '...')
        os.makedirs(output_dir, exist_ok=True)

        combat_result.to_csv(os.path.join(output_dir, f"{key}_{group}.csv"), index=True)

    print(f"{group} combat files are all saved: ctx, sv")




