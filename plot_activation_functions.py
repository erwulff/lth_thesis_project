import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

mpl.rc_file('my_matplotlib_rcparams')
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18

plt.close('all')

xx = np.linspace(start=-10, stop=10, num=1000)

logistic = 1. / (1 + np.exp(-xx))
tanh = (1 - np.exp(-xx)) / (1 + np.exp(-xx))
relu = np.maximum(np.zeros_like(xx), xx)
leakyrelu = np.maximum(.1 * xx, xx)

fig, axs = plt.subplots(2, 2, gridspec_kw={'hspace': 0.3, 'wspace': 0.2})
axs[0, 0].plot(xx, logistic)
axs[0, 0].set_title('Logistic')
axs[0, 1].plot(xx, tanh)
axs[0, 1].set_title('Tanh')
axs[1, 0].plot(xx, relu)
axs[1, 0].set_title('ReLU')
axs[1, 1].plot(xx, leakyrelu)
axs[1, 1].set_title('Leaky ReLU')
axs[0, 0].set_ylim(bottom=-1.1, top=1.1)
axs[0, 1].set_ylim(bottom=-1.1, top=1.1)
plt.show()
