import numpy as np
from numpy.linalg import inv

def split(s, delim=" "):
    words = []
    word = []
    for c in s:
        if c not in delim:
            word.append(c)
        else:
            if word:
                words.append(''.join(word))
                word = []
    if word:
        words.append(''.join(word))
    return words

def loadfile(filename):
    file = open(filename, "r")
    first = True
    rows = list()
    for line in file:
       if(first) == True:
           dims = split(line)
           print(dims[0], dims[1])
           first = False
       else:
           vals = split(line, [' ','\t'])
           for val in vals:
               print(val)
           rows.append(vals)
       
    return dims, rows


def vari(i, j, c):
    sum = 0
    for r in range(row_n):
        val1 = c_attars[r,c]-means[c][i]
        val2 = c_attars[r,c]-means[c][i]
        val = val1*val2
        sum += val
     
    sig = sum/row_n
    print(sig)
    return sig

def cov(c):
    co = list()
    for i in range(col_n):
        row = list()
        for j in range(col_n):
            row.append(vari(i, j, c))
            
        co.append(row)
    return co

def multiv(x, E, c):
    x = np.array(x)
    v1 = (x - means[:, c])
    v2 = np.matmul(v1.T, inv(E))
    v3 = np.matmul(v2, v1)
    v4= numpy.exp(0.5*v3[0,0])
    E_d = np.linalg.det(E)
    val = (2*np.pi*E_d*v4)^0.5
    
    print(val)
    return val


dims, rows = loadfile('Train.txt')
row_n = int(dims[2])
col_n = int(dims[0])

data = np.array(rows)
attrs = data[:,0:int(dims[0])]
y = data[:,int(dims[0]):]
y = y.astype('int')
classes = set(y.flat)

attrs = attrs.astype('float')
c_attars = list()
for c in classes:
    c_att = list()
    for r in range(row_n):
        if int(data[r,int(dims[0])]) == c:
            c_att.append(c)
    c_attars.append(c_att)

c_attars = np.array(c_attars)
c_attars = attrs.astype('float')


means = list()
varis = list()

for c in classes:
    m = list()
    v = list()
    for i in range(int(dims[0])):
        attr = attrs[:,i]
        print(attr.mean())
        print(attr.var())
        m.append(attr.mean())
        v.append(attr.var())
        
    means.append(m)
    varis.append(v)
 

means = np.array(means)
means = attrs.astype('float')

varis = np.array(varis)
varis = attrs.astype('float')








