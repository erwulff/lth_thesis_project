import numpy as np
import pandas as pd

module_name = 'AE_bn_LeakyReLU'
# grid_search_folder = module_name + '_AOD_grid_search_custom_normalization_1500epochs/'
grid_search_folder = module_name + '_AOD_grid_search_custom_normalization_1500epochs_12D10D8D/'

summary_df = pd.read_pickle(grid_search_folder + 'summary_df.pkl')
summary_df = summary_df.astype(dtype={'Batch size': int, 'Learning rate': float, 'Weight decay': float, 'Epoch': int, 'Validation loss': float})

best_summary = pd.DataFrame(columns=summary_df.columns)
worst_summary = pd.DataFrame(columns=summary_df.columns)
median_summary = pd.DataFrame(columns=summary_df.columns)

for nodes in summary_df['Nodes'].unique():
    curr_df = summary_df[summary_df['Nodes'] == nodes]

    curr_df = curr_df.sort_values(by='Validation loss')
    # uppermedian = curr_df[curr_df['Validation loss'] > curr_df['Validation loss'].median()].iloc[0]
    curr_lower_median = curr_df[curr_df['Validation loss'] < curr_df['Validation loss'].median()].iloc[-1]

    curr_best = curr_df.loc[curr_df['Validation loss'].idxmin()]
    curr_worst = curr_df.loc[curr_df['Validation loss'].idxmax()]
    best_summary = best_summary.append(curr_best)
    worst_summary = worst_summary.append(curr_worst)
    median_summary = median_summary.append(curr_lower_median)

best_summary.to_pickle(grid_search_folder + 'summary_best_param_df.pkl')
worst_summary.to_pickle(grid_search_folder + 'summary_worst_param_df.pkl')
median_summary.to_pickle(grid_search_folder + 'summary_median_param_df.pkl')

best_summary.pop('Module')
best_summary.pop('Epoch')
worst_summary.pop('Module')
worst_summary.pop('Epoch')
median_summary.pop('Module')
median_summary.pop('Epoch')

with open(grid_search_folder + module_name + '_best_table.tex', 'w') as tf:
    tf.write(best_summary.to_latex(index=False, formatters={'Validation loss': lambda x: '%.3e' % np.float(x),
                                                            'Learning rate': lambda x: '%.0e' % np.float(x),
                                                            'Weight decay': lambda x: '0' if np.float(x) == 0. else '%.0e' % np.float(x)}))

with open(grid_search_folder + module_name + '_worst_table.tex', 'w') as tf:
    tf.write(worst_summary.to_latex(index=False, formatters={'Validation loss': lambda x: '%.3e' % np.float(x),
                                                             'Learning rate': lambda x: '%.0e' % np.float(x),
                                                             'Weight decay': lambda x: '0' if np.float(x) == 0. else '%.0e' % np.float(x)}))

with open(grid_search_folder + module_name + '_median_table.tex', 'w') as tf:
    tf.write(median_summary.to_latex(index=False, formatters={'Validation loss': lambda x: '%.3e' % np.float(x),
                                                              'Learning rate': lambda x: '%.0e' % np.float(x),
                                                              'Weight decay': lambda x: '0' if np.float(x) == 0. else '%.0e' % np.float(x)}))
