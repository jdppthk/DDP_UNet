import h5py
import numpy as np
import pathlib

def gaussFilter(u, sigmax, sigmay):

    fu = np.fft.fftshift(np.fft.fft2(u))
    [nrows, ncols] = u.shape
    cy, cx = nrows/2, ncols/2
    y = np.linspace(0, nrows, nrows)
    x = np.linspace(0, ncols, ncols)
    X, Y = np.meshgrid(x, y)
    gmask = np.exp(-(((X-cx)/sigmax)**2 + ((Y-cy)/sigmay)**2))
    gfu = np.multiply(gmask, fu)
    gfu = np.fft.ifftshift(gfu)
    gu = np.fft.ifft2(gfu)

    return np.real(gu)


def getResidualTensor(truth, sigma, ds_factor):
    u_true = truth[:,:,0]
    v_true = truth[:,:,1]
    T_true = truth[:,:,2]


    T_filter = gaussFilter(T_true, sigma, sigma)
    u_filter = gaussFilter(u_true, sigma, sigma)
    v_filter = gaussFilter(v_true, sigma, sigma)

    uu_filter = gaussFilter(np.multiply(u_true, u_true), sigma, sigma)
    vv_filter = gaussFilter(np.multiply(v_true, v_true), sigma, sigma)
    uv_filter = gaussFilter(np.multiply(u_true, v_true), sigma, sigma)
    uT_filter = gaussFilter(np.multiply(u_true, T_true), sigma, sigma)
    vT_filter = gaussFilter(np.multiply(v_true, T_true), sigma, sigma)

    Rxx = uu_filter - np.multiply(u_filter, u_filter)
    Rxz = uv_filter - np.multiply(u_filter, v_filter)
    Rzz = vv_filter - np.multiply(v_filter, v_filter)
    Rtx = uT_filter - np.multiply(u_filter, T_filter)
    Rtz = vT_filter - np.multiply(v_filter, T_filter)
        
    residual = np.stack((Rxx, Rzz, Rxz, Rtx, Rtz), axis = -1) 
        
    return residual[::ds_factor,::ds_factor,:]


ds_factor = 8 
g = h5py.File('rbcTestSet.h5', 'w')
input_group = g.create_group('input_data')
target_group = g.create_group('target_data')
g.flush()

base_path = '/global/cscratch1/sd/jpathak/rbc2d/data/'

base_file_name = 'snapshots_Ra7Pr0xres512zres512seed'

initialize = True
for seed in np.arange(31,33):
    print(seed)
    for sbin in np.arange(2,11):
        print(sbin)
        filename = base_path + base_file_name + str(seed) + '_s' + str(sbin) + '.h5'
        print(filename)
        if pathlib.Path(filename).exists():
          with h5py.File(filename, 'r') as f:
              u = f['tasks']['u']
              v = f['tasks']['w']
              T = f['tasks']['b']
              p = f['tasks']['p']
              field = np.stack((u,v,T,p), axis = -1)   
              if initialize == True:

                  data_shape = field[:,::ds_factor,::ds_factor,:].shape
                  input_group.create_dataset("fields", data = field[:,::ds_factor,::ds_factor,:], maxshape = (None, data_shape[1], data_shape[2], data_shape[3]))
                  residual_tensor = target_group.create_dataset("residual_tensor", (data_shape[0], data_shape[1], data_shape[2],5), maxshape = (None, data_shape[1], data_shape[2], 5)) 

                  for image_iter in range(data_shape[0]):
                      residual_tensor[image_iter, :,:,:] = getResidualTensor(field[image_iter, :,:,0:3], 100, ds_factor)


                  g.flush()
                  initialize = False
              else:
                  current = g['input_data']['fields']
                  current_shape = current.shape
                  extend_shape = field.shape
                  current.resize(current_shape[0]+extend_shape[0],axis = 0)
                  current[current_shape[0]:current_shape[0]+extend_shape[0],:,:,:] = field[:,::ds_factor,::ds_factor,:]
                  residual_tensor = g['target_data']['residual_tensor']
                  residual_tensor.resize(current_shape[0] + extend_shape[0], axis = 0)


                  for image_iter in range(extend_shape[0]):

                      residual_tensor[image_iter+current_shape[0],:,:,:] = getResidualTensor(field[image_iter, : ,:, 0:3],100, ds_factor)

                  g.flush()

g.close()  
  
