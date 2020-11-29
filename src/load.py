import random
import scipy.sparse as sp
import scipy.io

def load_adj(filename):
    a = []
    b = []
    c = []
    edges = []
    with open(filename + '.edges') as f:
        for line in f:
            edge = [int(x) for x in line.split()]
            edges.append((edge[0], edge[1]))
            a.append(edge[0])
            b.append(edge[1])
            c.append(1)

    k = 2 + len(set(a))//10
    samples = random.sample(a, k)
    for x in samples:
        for y in samples:
            if(x!=y and (x,y) not in edges):
                a.append(x)
                b.append(y)
                c.append(1)

    adj = sp.coo_matrix((c, (a, b))).toarray()
    return sp.lil_matrix(adj)

def load_attr(filename):
    a = []
    b = []
    c = []
    with open(filename + '.feat') as f:
        for line in f:
            edge = [int(x) for x in line.split()]
            for attr_number, attr_value in enumerate(edge[1:]):
                a.append(edge[0])
                b.append(attr_number)
                c.append(attr_value)
    return sp.csr_matrix((c, (a, b)))

datapath = '../data/'
platform = 'twitter/'
filename = '66804457'
# filename = '19948202'
platform = 'facebook/'
filename = '107'

filename = datapath + platform + filename

adj = load_adj(filename)
features = load_attr(filename)

print(type(adj), type(features))
print(adj.shape, features.shape)

