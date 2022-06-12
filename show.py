import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def show_graph(model, sm, ge):
    pre_matrix = model.matrix_predict_graph(ge)
    ori_matrix = sm.matrix_graph(ge)
    plt.figure(figsize=(7, 5))
    sns.heatmap(pre_matrix)
    plt.title('prediction')
    plt.show()
    plt.figure(figsize=(7, 5))
    sns.heatmap(ori_matrix)
    plt.title('origin')
    plt.show()


def my_plt(models, train):
    labels = [r'one-$L_{1^+}$', r'one-$L_2$']
    origin = [item[2] for item in train]
    origin = np.array(origin)
    plt.figure()
    colors = ['g', 'r', 'y']
    alphas = [1, 1]
    linewidths = [1.5, 1]
    plt.scatter(range(len(origin)), origin, s=20, c='b', marker='x', label='truth', linewidth=0.5 )
    for i, md in enumerate(models):
        acc, one, pred = md.test_now(train)
        pred = np.array(pred)
        plt.plot(range(len(origin)), pred, label=labels[i],linewidth=linewidths[i], c=colors[i], alpha=alphas[i])
    plt.legend()
    plt.savefig('sensitivity2.eps', format='eps')
    plt.show()


def two_plt(xs):
    plt.figure()
    plt.plot(xs[0][:, 0, 0], xs[0][:, 0, 1], c='blue', label='user0 embed')
    plt.plot(xs[0][:, 1, 0], xs[0][:, 1, 1], c='red', label='user2 embed')
    plt.text(xs[0][-1, 0, 0], xs[0][-1, 0, 1], 0)
    plt.text(xs[0][-1, 1, 0], xs[0][-1, 1, 1], 0)
    plt.legend()
    for i, x in enumerate(xs[1:]):
        plt.plot(x[:, 0, 0], x[:, 0, 1], c='blue')
        plt.plot(x[:, 1, 0], x[:, 1, 1], c='red')
        plt.text(x[-1, 0, 0], x[-1, 0, 1], i+1)
        plt.text(x[-1, 1, 0], x[-1, 1, 1], i+1)
    # plt.show()
    plt.savefig('meet.eps')

def plt_tri(x, y, a, b, c1, label, color='b', styl=None):
    a = abs(a)
    b = abs(b)
    c1 = abs(c1)
    h = abs(b + c1 - a) / 2
    ax = b * a / (b + c1)
    # px = [x, x + a + b, x + a, x]
    px = [x, x + a, x + ax, x]
    # py = [y, y, y + c1, y]
    py = [y, y, y + h, y]
    if styl is not None:
        plt.plot(px, py, c=color, linestyle='--')
    else:
        plt.plot(px, py, c=color)
    plt.ylim(0, 1)
    plt.text(x + ax/2, y + h, label)

def plt_tris(ue, y=1.0, label='user',color='b'):
    ars = np.array(ue.data)
    for i in range(4):
        if i == 3:
            la = label + '_new'
            styl = '--'
        else:
            la = label + str(i)
            styl = None
        plt_tri(x=i, y=y, a=ars[i][0], b=ars[i][1], c1=ars[i][2], label=la, color=color, styl=styl)

def plt_s_c(ue,ce):
    plt.figure()
    plt_tris(ue, y=0.5, label='user',color='b')
    plt_tris(ce, y=0.1, label='item',color='r')
    plt.savefig('triangle.eps')

