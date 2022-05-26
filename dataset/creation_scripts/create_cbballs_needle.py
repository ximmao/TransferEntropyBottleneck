"""
modified from the original RTRBM code by Ilya Sutskever from
http://www.cs.utoronto.ca/~ilya/code/2008/RTRBM.tar

"""


import numpy as np
import os
import random

import matplotlib
import matplotlib.pyplot as plt
import functools

random.seed(0)
np.random.seed(0)

shape_std=np.shape
def shape(A):
    if isinstance(A, np.ndarray):
        return shape_std(A)
    else:
        return A.shape()

size_std = np.size
def size(A):
    if isinstance(A, np.ndarray):
        return size_std(A)
    else:
        return A.size()

det = np.linalg.det

def new_speeds(m1, m2, v1, v2):
    new_v2 = (2*m1*v1 + v2*(m2-m1))/(m1+m2)
    new_v1 = new_v2 + (v2 - v1)
    return new_v1, new_v2


def norm(x):
    return np.sqrt((x**2).sum())
def sigmoid(x):
    return 1./(1.+np.exp(-x))

SIZE=10
# size of bounding box: SIZE X SIZE.

def bounce_n(T=128, n=2, r=None, m=None):
    if m==None: m=np.array([1]*n)
    # r is to be rather small.
    X=np.zeros((T, n, 2), dtype='float')
    v = np.random.randn(n,2)
    v = v / norm(v)*.5
    good_config=False
    while not good_config:
        x = 2+np.random.rand(n,2)*8
        good_config=True
        for i in range(n):
            for z in range(2):
                if x[i][z]-r[i]<0:
                    good_config=False
                if x[i][z]+r[i]>SIZE:
                    good_config=False

        # that's the main part.
        for i in range(n):
            for j in range(i):
                if norm(x[i]-x[j])<r[i]+r[j]:
                    good_config=False


    eps = .5
    for t in range(T):
        # for how long do we show small simulation

        for i in range(n):
            X[t,i]=x[i]

        for mu in range(int(1/eps)):

            for i in range(n):
                x[i]+=eps*v[i]

            for i in range(n):
                for z in range(2):
                    if x[i][z]-r[i]<0:  v[i][z]= abs(v[i][z]) # want positive
                    if x[i][z]+r[i]>SIZE: v[i][z]=-abs(v[i][z]) # want negative


            for i in range(n):
                for j in range(i):
                    if norm(x[i]-x[j])<r[i]+r[j]:
                        # the bouncing off part:
                        w    = x[i]-x[j]
                        w    = w / norm(w)

                        v_i  = np.dot(w.transpose(),v[i])
                        v_j  = np.dot(w.transpose(),v[j])

                        new_v_i, new_v_j = new_speeds(m[i], m[j], v_i, v_j)

                        v[i]+= w*(new_v_i - v_i)
                        v[j]+= w*(new_v_j - v_j)

    return X

def ar(x,y,z):
    return z/2+np.arange(x,y,z,dtype='float')

def matricize_grey(X,res,r=None):

    T, n= shape(X)[0:2]

    A=np.zeros((T,1,res,res), dtype='float')
    L=np.ones(T, dtype='float')*(-1.)

    [I, J]=np.meshgrid(ar(0,1,1./res)*SIZE, ar(0,1,1./res)*SIZE)

    for t in range(T):
        for i in range(n):
            A[t,0]+= np.exp(-(  ((I-X[t,i,0])**2+(J-X[t,i,1])**2)/(r[i]**2)  )**4    )

        A[t][A[t]>1]=1
        L[t] = 0.
    return A, L

color_list=[(1.,1.,1.), #white
            (1.,0.,0.), #red
            (0.,1.,0.), #green
            (0.,0.,1.), #blue
            (1.,1.,0.), #yellow
            (1.,0.,1.), #magenta
            (0.,1.,1.)] #cyan
color_name=['white','red','green','blue','yellow','magenta','cyan']

def matricize_color(X,res,r=None,c=None,change_color=True, change_p=0.5):

    T, n= shape(X)[0:2]
    # if r==None: r=array([1.2]*n)
    
    # if not initialized, pick randomly
    if c == None: 
        bc = np.random.randint(len(color_list))
        c = color_list[bc]
    curr_color_cnt = 0

    A=np.zeros((T,3,res,res), dtype='float')
    L=np.ones(T, dtype='float')*(-1.)

    [I, J]=np.meshgrid(ar(0,1,1./res)*SIZE, ar(0,1,1./res)*SIZE)

    for t in range(T):
        for i in range(n):
            for k in range(3):
                A[t,k]+= np.exp(-(  ((I-X[t,i,0])**2+(J-X[t,i,1])**2)/(r[i]**2)  )**4    )

        A[t][A[t]>1]=1
        if change_color:
            if curr_color_cnt > 3:
                if np.random.rand() < change_p:
                    new_bc = np.random.randint(len(color_list))
                    if bc == new_bc:
                        curr_color_cnt += 1
                    else:
                        curr_color_cnt = 0
                        bc = new_bc
                        c = color_list[bc]
                else:
                    curr_color_cnt += 1
            else:
                curr_color_cnt += 1
        for k in range(3):
            A[t,k] *= c[k]
        # curr_color_cnt += 1
        L[t] = bc
    return A, L

def matricize_color_noswitch(X,res,r=None,c=None, num_class=7):

    T, n= shape(X)[0:2]
    # num_class = 7
    # if r==None: r=array([1.2]*n)
    
    # if not initialized, pick randomly
    if c == None: 
        bc = np.random.randint(len(color_list))
        c = color_list[bc]
    curr_color_cnt = 1

    A=np.zeros((num_class, T,3,res,res), dtype='float')
    L=np.ones((num_class, T), dtype='float')*(-1.)

    [I, J]=np.meshgrid(ar(0,1,1./res)*SIZE, ar(0,1,1./res)*SIZE)
    have_switch = False
    for t in range(T):
        for i in range(n):
            for k in range(3):
                for b_idx in range(num_class):
                    A[b_idx,t,k]+= np.exp(-(  ((I-X[t,i,0])**2+(J-X[t,i,1])**2)/(r[i]**2)  )**4    )

        A[:,t][A[:,t]>1]=1
        for b_idx in range(num_class):
            c = color_list[int(b_idx)]
            for k in range(3):
                A[b_idx,t,k,:,:] *= (2*c[k] + 1.)/3. # additional whitening
            L[b_idx,t] = int(b_idx)
    return A, L

def bounce_vec_noswitch(res, n=2, T=128, r =None, m =None, matricize=None):
    if r==None:
        r=np.array([1.2]*n)
    x = bounce_n(T,n,r,m)
    V, L = matricize(x,res,r)
    return V.reshape(7, T, -1, res**2), x, L

def bounce_vec(res, n=2, T=128, r =None, m =None, matricize=None):
    if r==None:
        r=np.array([1.2]*n)
    x = bounce_n(T,n,r,m)
    V, L = matricize(x,res,r)
    return V.reshape(T, -1, res**2), x, L

# make sure you have this folder
logdir = './bball_samples'
def show_sample(V):
    T   = len(V)
    res = int(np.sqrt(shape(V)[1]))
    for t in range(T):
        plt.imshow(V[t].reshape(res,res),cmap=matplotlib.cm.Greys_r)
        # Save it
        fname = logdir+'/'+str(t)+'.png'
        plt.savefig(fname)

def save_npy(cfilename, data):
    with open(cfilename, mode='wb') as f:
        np.save(f, data)


def create_training_data(T, N, num_ball):
    num_class = 7
    print('T={:d}, N={:d}'.format(T,N))
    res=32
    dump_dir='./data/bball_data_noswitch_g1_3/'
    change_p = 0.4
    curr_color_cnt = 0

    matricize_func = functools.partial(matricize_color_noswitch, num_class=num_class) # be default white ball
    
    train_dat=np.zeros((N, num_class,T,3,res*res), dtype=np.float32)
    train_x = np.zeros((N,T,num_ball,2),dtype=np.float32) # ball locations
    train_color = np.ones((N,num_class,T), dtype=np.float32)*(-1.)
    print(train_dat.shape)
    print('create train set')
    for i in range(N):
        # print(i)
        train_dat[i,:,:,:,:], train_x[i,:,:,:], train_color[i,:,:] =bounce_vec_noswitch(res=res, n=num_ball, T=T, matricize=matricize_func)
     # save train dataset
    train_dump_dir = dump_dir + 'train'
    if not os.path.exists(train_dump_dir):
        os.makedirs(train_dump_dir)
    save_npy(os.path.join(train_dump_dir, 'images_data.npy'), train_dat)
    save_npy(os.path.join(train_dump_dir, 'balls_loc.npy'), train_x)
    save_npy(os.path.join(train_dump_dir, 'balls_color.npy'), train_color)

    Nv=int(N/20)
    val_dat=np.zeros((Nv,num_class, T,3,res*res), dtype=np.float32)
    val_x = np.zeros((Nv,T,num_ball,2),dtype=np.float32) # ball locations
    val_color = np.ones((Nv,num_class, T), dtype=np.float32)*(-1.)
    print('create val set')
    for i in range(Nv):
        val_dat[i,:,:,:,:], val_x[i,:,:,:], val_color[i,:,:]=bounce_vec_noswitch(res=res, n=num_ball, T=T, matricize=matricize_func)
        shuffled_idx = np.random.permutation(num_class)
        val_dat[i,:,:,:,:] = val_dat[i][shuffled_idx]
        val_color[i,:,:] = val_color[i][shuffled_idx]
    print(shuffled_idx)
    # save val dataset

    val_dump_dir = dump_dir + 'valid'
    if not os.path.exists(val_dump_dir):
        os.makedirs(val_dump_dir)
    save_npy(os.path.join(val_dump_dir, 'images_data.npy'), val_dat)
    save_npy(os.path.join(val_dump_dir, 'balls_loc.npy'), val_x)
    save_npy(os.path.join(val_dump_dir, 'balls_color.npy'), val_color)

    Nt=int(N/20)
    test_dat=np.zeros((Nt,num_class, T,3,res*res), dtype=np.float32)
    test_x = np.zeros((Nt,T,num_ball,2),dtype=np.float32) # ball locations
    test_color = np.ones((Nt,num_class,T), dtype=np.float32)*(-1.)
    print('create test set')
    for i in range(Nt):
        test_dat[i,:,:,:,:], test_x[i,:,:,:], test_color[i,:,:]=bounce_vec_noswitch(res=res, n=num_ball, T=T, matricize=matricize_func)
        shuffled_idx = np.random.permutation(num_class)
        test_dat[i,:,:,:,:] = test_dat[i][shuffled_idx]
        test_color[i,:,:] = test_color[i][shuffled_idx]
    print(shuffled_idx)
    # save test dataset
    test_dump_dir = dump_dir + 'test'
    if not os.path.exists(test_dump_dir):
        os.makedirs(test_dump_dir)
    save_npy(os.path.join(test_dump_dir, 'images_data.npy'), test_dat)
    save_npy(os.path.join(test_dump_dir, 'balls_loc.npy'), test_x)
    save_npy(os.path.join(test_dump_dir, 'balls_color.npy'), test_color)
    return (train_dat, train_x, train_color), (val_dat, val_x, val_color), (test_dat, test_x, test_color)

# generate dataset
train_p, val_p, test_p = create_training_data(T=6, N=1500, num_ball=2)
print('dataset size')
dat, x, color = val_p
print('valid', dat.shape, color.shape)
dat, x, color = test_p
print('test', dat.shape, color.shape)

print('finished generating, now plotting visualizations for you')

import matplotlib.pyplot as plt
T   = dat.shape[2]
res = int(np.sqrt(dat.shape[4]))
print(T, res)
for dat_p_c, color_p_c in zip(dat, color):
    for b_idx in range(7):
        dat_p = dat_p_c[b_idx]
        color_p = color_p_c[b_idx]
        expanded_img = dat_p[0].reshape(-1,res,res)
        # print(np.histogram(expanded_img))
        for m in range(1,T):
            # print(np.histogram(dat_p[m]))
            expanded_img = np.concatenate((expanded_img, dat_p[m].reshape(-1,res,res)), axis=2)
        print(expanded_img.shape)
        print([color_name[int(k)] for k in color_p])
        plt.figure(figsize=(25,150))
        print(np.histogram(expanded_img[:, :]))
        plt.imshow(np.transpose(expanded_img[:, :],(1,2,0)), vmin=0., vmax=1.)
        plt.show()
        input('press')