import sys
import os
BIN = '../../../'
sys.path.append(BIN)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import my_matplotlib_style as ms

mpl.rc_file(BIN + 'my_matplotlib_rcparams')

# Validation loss from repeated trainings of same AE with same hyperparameters
best_losses_18 = np.array([1.015527e-05, 1.025252e-05, 2.331196e-05, 9.812227e-06, 1.019148e-05])
best_losses_20 = np.array([5.937740e-06, 4.288729e-06, 5.196329e-06, 5.161926e-06, 4.946615e-06])

plt.close('all')

# Choose same color as in previously made grid search plots
color = ms.colorprog(2, 3)

# Plot figures
xx = np.arange(len(best_losses_18)) + 1

plt.figure()
plt.plot(xx, best_losses_18, marker='*', color=color, linestyle='', label='wd=0.01\nbs=4096\nlr=0.01')
# Choose ylim to match previously made grid search plots to make comparison by
# eye easier
plt.ylim(top=2e-4, bottom=3e-6)
plt.yscale('log')
plt.suptitle('18-dimensional latent space')
plt.xticks(xx)
plt.xlabel('Training instance')
plt.ylabel('Validation loss')
plt.legend()
plt.savefig('reruns_18D.png')

plt.figure()
plt.plot(xx, best_losses_20, marker='*', color=color, linestyle='', label='wd=0.01\nbs=4096\nlr=0.01')
# Choose ylim to match previously made grid search plots to make comparison by
# eye easier
plt.ylim(top=2e-4, bottom=3e-6)
plt.yscale('log')
plt.suptitle('20-dimensional latent space')
plt.xticks(xx)
plt.xlabel('Training instance')
plt.ylabel('Validation loss')
plt.legend()
plt.savefig('reruns_20D.png')

plt.show()
