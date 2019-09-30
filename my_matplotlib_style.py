import matplotlib as mpl
import matplotlib.pyplot as plt


def set_my_style():

    # lines
    mpl.rcParams['lines.linewidth'] = 2
    mpl.rcParams['lines.color'] = 'r'

    # axis
    mpl.rcParams[axes.titlesize] = 26
    mpl.rcParams[axes.grid] = True


def sciy():
    plt.gca().ticklabel_format(style='sci', scilimits=(0, 0), axis='y')


def scix():
    plt.gca().ticklabel_format(style='sci', scilimits=(0, 0), axis='x')
