import matplotlib.pyplot as plt
import pandas as pd

import matplotlib as mpl

import my_matplotlib_style as ms

mpl.rc_file('my_matplotlib_rcparams')

BIN = '../data/split_data/'

# train0 = pd.read_pickle(BIN + 'sub_train_0')
# test0 = pd.read_pickle(BIN + 'sub_test_0')
#
# train1 = pd.read_pickle(BIN + 'sub_train_1')
# test1 = pd.read_pickle(BIN + 'sub_test_1')
#
# train2 = pd.read_pickle(BIN + 'sub_train_2')
# test2 = pd.read_pickle(BIN + 'sub_test_2')
#
# train3 = pd.read_pickle(BIN + 'sub_train_3')
# test3 = pd.read_pickle(BIN + 'sub_test_3')
#
# train4 = pd.read_pickle(BIN + 'sub_train_4')
# test4 = pd.read_pickle(BIN + 'sub_test_4')
#
# train5 = pd.read_pickle(BIN + 'sub_train_5')
# test5 = pd.read_pickle(BIN + 'sub_test_5')
#
# train6 = pd.read_pickle(BIN + 'sub_train_6')
# test6 = pd.read_pickle(BIN + 'sub_test_6')
#
# train7 = pd.read_pickle(BIN + 'sub_train_7')
# test7 = pd.read_pickle(BIN + 'sub_test_7')
#
# train8 = pd.read_pickle(BIN + 'sub_train_8')
# test8 = pd.read_pickle(BIN + 'sub_test_8')
#
# train9 = pd.read_pickle(BIN + 'sub_train_9')
# test9 = pd.read_pickle(BIN + 'sub_test_9')

plt.close('all')
N_plots = 10
for i in range(N_plots):
    thiscol = ms.colorprog(i, N_plots)
    train = pd.read_pickle(BIN + 'sub_train_%d' % i)
    test = pd.read_pickle(BIN + 'sub_test_%d' % i)
    plt.figure(1)
    plt.hist(train['pT'], color=thiscol, density=True)

plt.show()
