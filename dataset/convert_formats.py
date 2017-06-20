import numpy as np
import pandas as pd
import os, shutil, time, h5py
import pickle
from random import seed, randint
import time


datapath  = 'mix0.data'
labelpath = 'mix0.label'
hdfpath   = 'test-domain2.h5'

def shuffle(filename1, filename2, hdfname):
    start_time = time.time()

    # Data #
    X = [line.rstrip('\n') for line in open(filename1)]
    # Label #
    Y = [line.rstrip('\n') for line in open(filename2)]

    # Reading into matrices #
    n_rows = len(X)
    n_cols = 128
    data = np.zeros((n_rows, 4, n_cols/4, 1), dtype=int)
    label = np.zeros((n_rows, 1, 1, 1), dtype=int)
    d = 0
    for i in X:
        i += '\x00' * (n_cols - len(i))
        l = [ord(c) for c in i]
        data[d, 0, :, 0] = l[0:n_cols/4]
        data[d, 1, :, 0] = l[n_cols/4:2*n_cols/4]
        data[d, 2, :, 0] = l[2*n_cols/4:3*n_cols/4]
        data[d, 3, :, 0] = l[3*n_cols/4:n_cols]
        label[d, 0, 0, 0] = int(Y[d])
        d += 1

    # Shuffle #
    seed(time.time())
    tmp  = np.zeros_like(data[0, :, :, 0])
    for i in xrange(10000):
        index1 = randint(0, n_rows)
        index2 = randint(0, n_rows)
        tmp = data[index1, :, :, 0]
        data[index1, :, :, 0] = data[index2, :, :, 0] 
        data[index2, :, :, 0] = tmp
        tmp2 = label[index1, 0, 0, 0]
        label[index1, 0, 0, 0] = label[index2, 0, 0, 0]
        label[index2, 0, 0, 0] = tmp2

    # To HDF5 #
    h5f = h5py.File(hdfname, 'a') # append mode #
    h5f.create_dataset('data', data=data)
    h5f.create_dataset('label', data=label)
    h5f.close()
    print('%s seconds' % (time.time() - start_time))

def convert4():
    # Convert the many csv files into a single hdf file
    start_time = time.time()

    h5f = h5py.File(hdfpath, 'a') # append mode #

    # Data #
    X = [line.rstrip('\n') for line in open('dga_predict/malware_dns_53_uniq')]
    n_rows = len(X)
    n_cols = 128
    data = np.zeros((n_rows, 4, n_cols/4, 1), dtype=int)
    label = np.zeros((n_rows, 1, 1, 1), dtype=int)
    d = 0
    for i in X:
        i += '\x00' * (n_cols - len(i))
        l = [ord(c) for c in i]
        data[d, 0, :, 0] = l[0:n_cols/4]
        data[d, 1, :, 0] = l[n_cols/4:2*n_cols/4]
        data[d, 2, :, 0] = l[2*n_cols/4:3*n_cols/4]
        data[d, 3, :, 0] = l[3*n_cols/4:n_cols]
        label[d, 0, 0, 0] = 1
        d += 1

    print label
    print data
    h5f.create_dataset('data', data=data)
    h5f.create_dataset('label', data=label)
    h5f.close()
    print('%s seconds' % (time.time() - start_time))

def convert3():
    # Convert the many csv files into a single hdf file
    start_time = time.time()

    h5f = h5py.File(hdfpath, 'a') # append mode #

    # Data #
    X = pickle.load(open('dga_predict/traindata.pkl'))
    n_rows = len(X)
    n_cols = 128
    data  = np.zeros((n_rows, 4, n_cols/4, 1), dtype=int)
    label = np.zeros((n_rows, 1, 1, 1), dtype=int)
    d = 0
    for i in X:
        if i[0] == "benign": # legit
            label[d, 0, 0, 0] = 0
        else: # dga
            label[d, 0, 0, 0] = 1
        s = i[1] + '\x00' * (n_cols - len(i[1]))
        l = [ord(c) for c in s]
        data[d, 0, :, 0] = l[0:n_cols/4]
        data[d, 1, :, 0] = l[n_cols/4:2*n_cols/4]
        data[d, 2, :, 0] = l[2*n_cols/4:3*n_cols/4]
        data[d, 3, :, 0] = l[3*n_cols/4:n_cols]
        d += 1

    h5f.create_dataset('data', data=data)
    h5f.create_dataset('label', data=label)
    h5f.close()
    print('%s seconds' % (time.time() - start_time))

def convert2():
    # Convert the many csv files into a single hdf file
    start_time = time.time()

    h5f = h5py.File(hdfpath, 'a') # append mode #

    # Data #
    X = [line.rstrip('\n') for line in open(datapath)]
    n_rows = len(X)
    n_cols = 128
    data = np.zeros((n_rows, 4, n_cols/4, 1), dtype=int)
    d = 0
    for i in X:
        i += '\x00' * (n_cols - len(i))
        l = [ord(c) for c in i]
        data[d, 0, :, 0] = l[0:n_cols/4]
        data[d, 1, :, 0] = l[n_cols/4:2*n_cols/4]
        data[d, 2, :, 0] = l[2*n_cols/4:3*n_cols/4]
        data[d, 3, :, 0] = l[3*n_cols/4:n_cols]
        d += 1

    h5f.create_dataset('data', data=data)

    # Label #
    X = [line.rstrip('\n') for line in open(labelpath)]
    n_rows = len(X)
    n_cols = 1
    data = np.zeros((n_rows, n_cols, 1, 1), dtype=int)
    data[:, 0, 0, 0] = X
    h5f.create_dataset('label', data=data)
     
    h5f.close()
    print('%s seconds' % (time.time() - start_time))

def convert():
    # Convert the many csv files into a single hdf file
    start_time = time.time()

    h5f = h5py.File(hdfpath, 'a') # append mode #

    # Data #
    X = [line.rstrip('\n') for line in open(datapath)]
    n_rows = len(X)
    n_cols = 128
    data = np.zeros((n_rows, n_cols, 1, 1), dtype=int)
    d = 0
    for i in X:
        i += '\x00' * (n_cols - len(i))
        data[d, :, 0, 0] = [ord(c) for c in i]
        d += 1

    h5f.create_dataset('data', data=data)

    # Label #
    X = [line.rstrip('\n') for line in open(labelpath)]
    n_rows = len(X)
    n_cols = 1
    data = np.zeros((n_rows, n_cols, 1, 1), dtype=int)
    data[:, 0, 0, 0] = X
    h5f.create_dataset('label', data=data)
     
    h5f.close()
    print('%s seconds' % (time.time() - start_time))

def test():
    h5f = h5py.File(hdfpath, 'r')  # 15.3 MB on disk

    arr = h5f['data'][:]
    print(arr.shape)
    print(arr.nbytes / 1e6, 'MB in memory')

    arr = h5f['label'][:]
    print(arr.shape)
    print(arr.nbytes / 1e6, 'MB in memory')

def test2():
    X = pd.read_csv(datapath, index_col=None, header=None)
    print type(X)
    print type(X.values)
    return X
