import pandas as pd

# Load data
train = pd.read_pickle('processed_data/train.pkl')
test = pd.read_pickle('processed_data/test.pkl')

# Sort according to pT
train = train.sort_values(by='pT')
test = test.sort_values(by='pT')

columns = train.columns

indx_train = len(train) // 10
indx_test = len(test) // 10

path_to_save_folder = '../data/split_data/'
for i in range(10):
    sub_train = train.values[i * indx_train:(i + 1) * indx_train]
    sub_test = test.values[i * indx_test:(i + 1) * indx_test]

    sub_train = pd.DataFrame(sub_train, columns=columns)
    sub_test = pd.DataFrame(sub_test, columns=columns)

    sub_train.to_pickle(path_to_save_folder + 'sub_train_%d' % i)
    sub_test.to_pickle(path_to_save_folder + 'sub_test_%d' % i)
