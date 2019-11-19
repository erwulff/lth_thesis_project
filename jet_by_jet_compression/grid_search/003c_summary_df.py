import numpy as np
import pandas as pd

module_name = 'AE_bn_LeakyReLU'
grid_search_folder = module_name + '_AOD_grid_search_custom_normalization_1500epochs/'

summary_df = pd.read_pickle(grid_search_folder + 'summary_df.pkl')
summary_df = summary_df.astype(dtype={'Batch size': int, 'Learning rate': float, 'Weight decay': float, 'Epoch': int, 'Validation loss': float})

best_summary = pd.DataFrame(columns=summary_df.columns)
for nodes in summary_df['Nodes'].unique():
    curr_df = summary_df[summary_df['Nodes'] == nodes]
    curr_best = curr_df.loc[curr_df['Validation loss'].idxmin()]
    best_summary = best_summary.append(curr_best)

best_summary.to_pickle(grid_search_folder + 'summary_best_param_df.pkl')

best_summary.pop('Module')
best_summary.pop('Epoch')

with open(grid_search_folder + module_name + '_best_table.tex', 'w') as tf:
    tf.write(best_summary.to_latex(index=False, formatters={'Validation loss': lambda x: '%.3e' % np.float(x),
                                                            'Learning rate': lambda x: '%.0e' % np.float(x),
                                                            'Weight decay': lambda x: '0' if np.float(x) == 0. else '%.0e' % np.float(x)}))
