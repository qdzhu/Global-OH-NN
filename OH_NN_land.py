from utils_vae import read_field
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Model
import tensorflow
# import keras 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse
import sys, importlib

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.utils import plot_model
import tensorflow as tf

from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Dense,Flatten 
from VAES import encoder_gen , dense_gen,encoder_dense,cloud_model,kl,schedule,decoder_dense
import matplotlib.pyplot as plt
import json
from utils_vae import train_test_data
import pandas as pd
import argparse
import pickle
import math

output_filepath = '/net/fs06/d2/qdzhu/process/CESM2/spc/continent/'
mask_data = pd.read_csv(output_filepath+'mask.csv')
indx = mask_data['lon']<0
mask_data.loc[indx, 'lon']= mask_data.loc[indx, 'lon']+360
mask_data = mask_data.sort_values(by=['lat','lon'])
lat = np.array(mask_data['lat'].values).reshape((192,288))
lat_np = np.repeat(lat[ np.newaxis, :, :], 65*12, axis=0)
month = np.sin(np.array(list(range(1,13))*65)/12*2*math.pi)
month_np = np.repeat(month[:, np.newaxis], 192*288, axis=1).reshape((65*12,192,288))
mask = np.array(mask_data['mask']).reshape((192,288))


class Loss_log_like( keras.layers.Layer ):
    def __init__(self,name = None):
        super(Loss_log_like,self).__init__(name= name)
        self.loss_mse = tf.keras.losses.MeanSquaredError()
    
    def call(self,inputs , weights = 1):
        n_dims = 1
        ytrue,ypred = inputs['true'],inputs['pred']
        mu = ypred[:, 0:n_dims]
        logsigma = ypred[:, n_dims:]

        mse = -0.5*K.sum(K.square((ytrue-mu)/K.exp(logsigma)),axis=1)
        sigma_trace = -K.sum(logsigma, axis=1)
        log2pi = -0.5*n_dims*np.log(2*np.pi)

        log_likelihood = mse+sigma_trace+log2pi

    
        loss = K.mean(-log_likelihood)
        self.add_loss(weights*loss)
        
        self.add_metric(loss,name = self.name)
        self.add_metric(self.loss_mse(mu,ytrue),name = 'mse_'+self.name)## only for comparison
              
#         qs = [0.975, 0.995]
#         q = tf.constant(np.array([qs]), dtype=tf.float32)
#         error = ytrue - mu
#         val = tf.maximum(q*error, (q-1)*error)
#         lossq = K.mean(val)
        
#         self.add_loss(2*lossq)
#         self.add_metric(lossq,name = self.name+'quntile')

        return ypred
    

    
class Loss_Simple( keras.layers.Layer ):
    def __init__(self,name = None):
        super(Loss_Simple,self).__init__(name= name)
        self.loss_mse = tf.keras.losses.MeanSquaredError()
    
    def call(self,inputs , weights = 1):
        n_dims = 1
        ytrue,ypred = inputs['true'],inputs['pred']
        # yp = ypred[:,0]
        loss = self.loss_mse(ytrue,ypred)
        self.add_loss(weights*loss)
        self.add_metric(self.loss_mse(ytrue,ypred),name = self.name+'mse')
        

        
        return ypred
    

class Loss_rotation_invariant( keras.layers.Layer ):
    
    def __init__(self,name = None):
        super(Loss_rotation_invariant,self).__init__(name= name)
        self.loss_mse = tf.keras.losses.MeanAbsoluteError()
        self.lambda_weight = 1
 
    def call (self,inputs, weights = 1):
        # loss: reconstruction
        y_true,y_pred = inputs['true'],inputs['pred']
        
        
        loss = self.loss_mse(y_true,y_pred)
        mean_org = K.mean(K.abs(y_true))
        self.add_loss(weights * loss/mean_org)
        self.add_metric(loss/mean_org,name = self.name)
        self.add_metric(mean_org,name = 'mean_org')
        return loss
    
    
def train_nn_models(cesm2_model, total_input_vars, feature_sel, opt, config_file):    
    output_path = '/net/fs06/d2/qdzhu/process/CESM_recalc/'+cesm2_model+'/'
    data = np.load(output_path + "trop_noemis.npy")
    
    with open('/net/fs06/d2/qdzhu/process/CESM_recalc/'+cesm2_model+'/feature_pool', "rb") as fp:   # Unpickling
        train_vars = pickle.load(fp)
    #train_vars = ['FLASHFRQ','CLDTOT','SFCH2O','SFCO','SFNO','MEG_ISOP'] + ['Q','T','strat_O3','jo3_b','O3','NO','NO2','CH2O','CO','ISOP','CH4','jno2','OH']

    print(total_input_vars)
    input_vars = total_input_vars + ['OH']

    #normalize
    data_norm = np.zeros((65*12, 192,288, len(input_vars)))
    means = []
    stds = []
    for i in range(len(input_vars)):
        var = input_vars[i]
        if (var == 'month') :
            this_mean = np.mean(month_np)
            this_std = np.std(month_np)
            data_norm[:,:,:,i] = (month_np-this_mean)/this_std
        elif (var == 'lat') :
            this_mean = np.mean(lat_np)
            this_std = np.std(lat_np)
            data_norm[:,:,:,i] = (lat_np-this_mean)/this_std
        else:
            if ('weight' in opt) and (var == 'OH'):
                i_var = train_vars.index(var)
                this_mean = np.mean(data[:,:,:,i_var]*data[:,:,:,'MASS'])
                this_std = np.std(data[:,:,:,i_var]*data[:,:,:,'MASS'])
                data_norm[:,:,:,i] = (data[:,:,:,i_var]*data[:,:,:,'MASS']-this_mean)/this_std
            else:
                i_var = train_vars.index(var)
                this_mean = np.mean(data[:,:,:,i_var])
                this_std = np.std(data[:,:,:,i_var])
                data_norm[:,:,:,i] = (data[:,:,:,i_var]-this_mean)/this_std
        means.append(this_mean)
        stds.append(this_std)
        
    x_train = data_norm[:50*12,:,:,:-1]
    y_train = data_norm[:50*12,:,:,-1]

    x_test = data_norm[50*12:,:,:,:-1]
    y_test = data_norm[50*12:,:,:,-1]
    
    if opt == 'train':

        # # Opening JSON file
        with open('./configs/config_32_RI_decoder_4nodes.json') as json_file:
        #with open('./configs/' + config_file) as json_file:
            model_config = json.load(json_file)
        outputs = []
        coef = model_config["coef"]
        dim = model_config["dim"]
        inshape = x_train.shape[3]
        rotation_invariant = model_config['rotation_invariant']
        train_decoder = model_config['train_decoder']
        inshape_cloud = (inshape+model_config["configs_encoder"]['latent_dim'],)
        inshape_precip = (inshape+model_config["configs_encoder"]['latent_dim'],)

        oh_model = dense_gen(inshape,model_config["config_dense"])

        inputdense = tf.keras.layers.Input(shape=(inshape,))
        oh_true = tf.keras.layers.Input(shape=(1,))

        oh =oh_model.dense_nn(inputdense)
        oh_pred  = Loss_Simple(name = 'oh_loss')({'true':oh_true,'pred':oh}, weights = 1)
        outputs.append(oh_pred)

        vae = tf.keras.Model(inputs=[inputdense,oh_true],
                             outputs=outputs)

        optimizer = tensorflow.keras.optimizers.Adam(learning_rate=0.0004)
        callback_lr=LearningRateScheduler(schedule,verbose=1)
        earlyStopping=tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')

        vae.compile(optimizer = optimizer)

        x_train = x_train.reshape((50*12*192*288, len(total_input_vars)))
        y_train = y_train.reshape((50*12*192*288, 1))
        
        mask_np = np.repeat(mask[np.newaxis, :,:],50*12,axis=0).reshape(50*12*192*288)
        x_train = x_train[~np.isnan(mask_np),:]
        y_train = y_train[~np.isnan(mask_np),:]
        
        hist = vae.fit(
                x=[x_train,y_train],
                epochs=20,
                batch_size=128,
                validation_split = 0.2,
                callbacks= [earlyStopping,callback_lr],
                shuffle = True
            )



        path = '/net/fs06/d2/qdzhu/oh_models/nn_land/{}/'.format(feature_sel)
        #encoder_result.vae_encoder.save(path + 'encoder_2_32_mse')
        #decode_zz.save(path + 'decoder_2_32_mse')
        vae.save(path + 'org_2_32_mse_6in')
        #vae.save(path + config_file[7:-5]) #'org_2_32_mse_6in')
        #oh_model.dense_nn.save(path + 'oh_dense_' + config_file[7:-5])
    
    if opt == 'test':
        x_test = x_test.reshape((15*12*192*288, len(total_input_vars)))
        y_test = y_test.reshape((15*12*192*288, 1))
        
        mask_np = np.repeat(mask[np.newaxis, :,:],15*12,axis=0).reshape(15*12*192*288)
        x_test = x_test[~np.isnan(mask_np),:]
        y_test = y_test[~np.isnan(mask_np),:]
        
        path = '/net/fs06/d2/qdzhu/oh_models/nn_land/{}/'.format(feature_sel)
        vae =  tf.keras.models.load_model(path + 'org_2_32_mse_6in')
        #vae =  tf.keras.models.load_model(path + config_file[7:-5])#'org_2_32_mse_6in')
        pred_test= vae.predict([x_test,y_test])
        np.save(path + 'pred_{}_{}.npy'.format('org_2_32_mse_6in', cesm2_model), pred_test)
        np.save(path + 'test_{}_{}.npy'.format('org_2_32_mse_6in', cesm2_model), y_test)
        np.save(path + 'x_test_{}_{}.npy'.format('org_2_32_mse_6in', cesm2_model), x_test)
        #np.save(path + 'pred_{}_{}.npy'.format(config_file[7:-5], cesm2_model), pred_test)
        #np.save(path + 'test_{}_{}.npy'.format(config_file[7:-5], cesm2_model), y_test)


def pred_sat_nn_models(cesm2_model, total_input_vars, feature_sel, opt):    
    output_path = '/net/fs06/d2/qdzhu/process/CESM_recalc/'+cesm2_model+'/'
    data = np.load(output_path + "trop_noemis.npy")
    
    with open('/net/fs06/d2/qdzhu/process/CESM_recalc/'+cesm2_model+'/feature_pool', "rb") as fp:   # Unpickling
        train_vars = pickle.load(fp)
    #train_vars = ['FLASHFRQ','CLDTOT','SFCH2O','SFCO','SFNO','MEG_ISOP'] + ['Q','T','strat_O3','jo3_b','O3','NO','NO2','CH2O','CO','ISOP','CH4','jno2','OH']

    print(total_input_vars)
    input_vars = total_input_vars + ['OH']

    sat_no2 = np.load("/net/fs06/d2/qdzhu/process/CESM_recalc/sat/no2_new_col_qa4ecv_nofilter_scale.npy")
    sat_no2 = sat_no2.reshape((10*12,192,288))
    sat_hcho = np.load("/net/fs06/d2/qdzhu/process/CESM_recalc/sat/hcho_new_col_qa4ecv_nofilter_scale.npy")
    sat_hcho = sat_hcho.reshape((10*12,192,288))
    sat_co = np.load("/net/fs06/d2/qdzhu/process/CESM_recalc/sat/co_new_col.npy")
    sat_co = sat_co.reshape((10*12,192,288))
    #normalize
    data_norm = np.zeros((65*12, 192,288, len(input_vars)))
    means = []
    stds = []
    for i in range(len(input_vars)):
        var = input_vars[i]
        if (var == 'month') :
            this_mean = np.mean(month_np)
            this_std = np.std(month_np)
            data_norm[:,:,:,i] = (month_np-this_mean)/this_std
        elif (var == 'lat') :
            this_mean = np.mean(lat_np)
            this_std = np.std(lat_np)
            data_norm[:,:,:,i] = (lat_np-this_mean)/this_std
        else:
            i_var = train_vars.index(var)
            this_mean = np.mean(data[:,:,:,i_var])
            this_std = np.std(data[:,:,:,i_var])
            data_norm[:,:,:,i] = (data[:,:,:,i_var]-this_mean)/this_std
            if (var == 'NO2col_trop') and ('no2' in opt):
                data_norm[55*12:,:,:,i] = (sat_no2-this_mean)/this_std
            if (var == 'CH2Ocol_trop') and ('hcho' in opt):
                data_norm[55*12:,:,:,i] = (sat_hcho-this_mean)/this_std
            if (var == 'COcol_trop') and ('co' in opt):
                data_norm[55*12:,:,:,i] = (sat_co-this_mean)/this_std
                
        means.append(this_mean)
        stds.append(this_std)
        
    x_test = data_norm[55*12:,:,:,:-1]
    y_test = data_norm[55*12:,:,:,-1]
   
 
    x_test = x_test.reshape((10*12*192*288, len(total_input_vars)))
    y_test = y_test.reshape((10*12*192*288, 1))
    mask_np = np.repeat(mask[np.newaxis, :,:],10*12,axis=0).reshape(10*12*192*288)
    x_test = x_test[~np.isnan(mask_np),:]
    y_test = y_test[~np.isnan(mask_np),:]
    path = '/net/fs06/d2/qdzhu/oh_models/nn_land/{}/'.format(feature_sel)
    vae =  tf.keras.models.load_model(path + 'org_2_32_mse_6in')
    #vae =  tf.keras.models.load_model(path + config_file[7:-5])#'org_2_32_mse_6in')
    pred_test= vae.predict([x_test,y_test])
    np.save(path + 'sat_pred_{}_{}_{}.npy'.format(opt,'org_2_32_mse_6in', cesm2_model), pred_test)
        #np.save(path + 'test_{}_{}.npy'.format('org_2_32_mse_6in', cesm2_model), y_test)
        #np.save(path + 'pred_{}_{}.npy'.format(config_file[7:-5], cesm2_model), pred_test)
        #np.save(path + 'test_{}_{}.npy'.format(config_file[7:-5], cesm2_model), y_test)
        
def pred_ind_feature_nn_models(cesm2_model, total_input_vars, feature_sel, opt):    
    output_path = '/net/fs06/d2/qdzhu/process/CESM_recalc/'+cesm2_model+'/'
    data = np.load(output_path + "trop_noemis.npy")
    
    with open('/net/fs06/d2/qdzhu/process/CESM_recalc/'+cesm2_model+'/feature_pool', "rb") as fp:   # Unpickling
        train_vars = pickle.load(fp)
    #train_vars = ['FLASHFRQ','CLDTOT','SFCH2O','SFCO','SFNO','MEG_ISOP'] + ['Q','T','strat_O3','jo3_b','O3','NO','NO2','CH2O','CO','ISOP','CH4','jno2','OH']

    print(total_input_vars)
    input_vars = total_input_vars + ['OH']
    
    #normalize
    data_norm = np.zeros((65*12, 192,288, len(input_vars)))
    means = []
    stds = []
    for i in range(len(input_vars)):
        var = input_vars[i]
        if (var == 'month') :
            this_mean = np.mean(month_np)
            this_std = np.std(month_np)
            data_norm[:,:,:,i] = (month_np-this_mean)/this_std
        elif (var == 'lat') :
            this_mean = np.mean(lat_np)
            this_std = np.std(lat_np)
            data_norm[:,:,:,i] = (lat_np-this_mean)/this_std
        else:
            i_var = train_vars.index(var)
            this_mean = np.mean(data[:,:,:,i_var])
            this_std = np.std(data[:,:,:,i_var])
            data_norm[:,:,:,i] = (data[:,:,:,i_var]-this_mean)/this_std
            #replace the input feature with annual mean
            if var in opt:
                this_data = data[:,:,:,i_var].reshape((65,12,192,288))
                data_annual_const = np.nanmean(this_data[55:,:,:,:],axis=0)
                data_annual_const_new = np.repeat(data_annual_const[np.newaxis,:,:,:], 65, axis=0)
                data_annual_const_new = data_annual_const_new.reshape((65*12,192,288))
                data_norm[:,:,:,i] = (data_annual_const_new-this_mean)/this_std
                
        means.append(this_mean)
        stds.append(this_std)
        
    x_test = data_norm[55*12:,:,:,:-1]
    y_test = data_norm[55*12:,:,:,-1]
   
 
    x_test = x_test.reshape((10*12*192*288, len(total_input_vars)))
    y_test = y_test.reshape((10*12*192*288, 1))
    mask_np = np.repeat(mask[np.newaxis, :,:],10*12,axis=0).reshape(10*12*192*288)
    x_test = x_test[~np.isnan(mask_np),:]
    y_test = y_test[~np.isnan(mask_np),:]
    
    path = '/net/fs06/d2/qdzhu/oh_models/nn_land/{}/'.format(feature_sel)
    vae =  tf.keras.models.load_model(path + 'org_2_32_mse_6in')
    #vae =  tf.keras.models.load_model(path + config_file[7:-5])#'org_2_32_mse_6in')
    pred_test= vae.predict([x_test,y_test])
    np.save(path + 'const_pred_{}_{}_{}.npy'.format("_".join(opt),'org_2_32_mse_6in', cesm2_model), pred_test)
        #np.save(path + 'test_{}_{}.npy'.format('org_2_32_mse_6in', cesm2_model), y_test)
        #np.save(path + 'pred_{}_{}.npy'.format(config_file[7:-5], cesm2_model), pred_test)
        #np.save(path + 'test_{}_{}.npy'.format(config_file[7:-5], cesm2_model), y_test)
        
def main(args):
    opt = args.s
    cesm2_model = args.m
    feature_sel = args.f
    
    match feature_sel:
        case "v1":
            total_input_vars = ['FLASHFRQ','CLDTOT','Q','T','SFCH2O','SFCO','SFNO','MEG_ISOP','jo3_b','CH4','jno2']
        case "v2":
            vars_2d = ['FLASHFRQ','CLDTOT']
            vars_3d = ['Q','T','jo3_b','O3','NO2','CH2O','CO','ISOP','CH4','jno2']
            total_input_vars = vars_2d + vars_3d
        case "v3":
            total_input_vars = ['FLASHFRQ','CLDTOT','Q','T','jo3_b','jno2','NO2','CH2O','CO','CH4']
        case "v4":
            total_input_vars = ['FLASHFRQ','CLDTOT','Q','T','jo3_b']
        case "v5":
            total_input_vars = ['jo3_b','NO2','CH2O','CO','ISOP','CH4']
        case "v6":
            total_input_vars = ['FLASHFRQ','CLDTOT','Q','T','jo3_b','jno2','CO','CH4']
        case "sat":
            total_input_vars = ['FLASHFRQ','CLDTOT','Q','T','jo3_b','jno2','NO2col_trop','CH2Ocol_trop','COcol_trop']
        case "sat_v1":
            total_input_vars = ['FLASHFRQ','CLDTOT','Q','T','jo3_b','jno2','NO2col_trop','CH2Ocol_trop','COcol_trop','O3col_tot','lat','month']
        case "sat_v2":
            total_input_vars = ['FLASHFRQ','CLDTOT','Q','T','jo3_b','jno2','NO2col_trop','CH2Ocol_trop','COcol_trop','O3','lat','month']
        case "sat_v3":
            total_input_vars = ['FLASHFRQ','CLDTOT','Q','T','jo3_b','jno2','NO2col_trop','CH2Ocol_trop','COcol_trop','month']
        case "sat_v4":
            total_input_vars = ['FLASHFRQ','CLDTOT','Q','T','jo3_b','jno2','NO2col_trop','CH2Ocol_trop','COcol_trop','lat','month','MASS','TROP_P']
        case "sat_v5":
            total_input_vars = ['FLASHFRQ','CLDTOT','Q','T','jo3_b','jno2','NO2col_trop','CH2Ocol_trop','COcol_trop','lat','month','MASS']
        case "sat_v6":
            total_input_vars = ['FLASHFRQ','CLDTOT','Q','T','jo3_b','jno2','NO2col_trop','CH2Ocol_trop','COcol_trop','lat','month']
        case "sat_v7":
            total_input_vars = ['FLASHFRQ','CLDTOT','Q','T','jo3_a','jno2','NO2col_trop','CH2Ocol_trop','COcol_trop','lat','month']
        case "sat_v7_weight":
            total_input_vars = ['FLASHFRQ','CLDTOT','Q','T','jo3_a','jno2','NO2col_trop','CH2Ocol_trop','COcol_trop','lat','month']
        case "v7":
            total_input_vars = ['FLASHFRQ','CLDTOT','Q','T','jo3_a','jno2','NO2','CH2O','CO','lat','month']
        case "sat_v8":
            total_input_vars = ['FLASHFRQ','CLDTOT','T','jo3_a_Q','jno2','NO2col_trop','CH2Ocol_trop','COcol_trop','lat','month']
        case "sat_v9":
            total_input_vars = ['FLASHFRQ','CLDTOT','Q','T','jo3_a','jno2','NO2col_trop','CH2Ocol_trop','COcol_trop','lat','month','MASS']
        case "sat_v10":
            total_input_vars = ['FLASHFRQ','CLDTOT','Q','T','jo3_a','jno2','NO2col_trop','CH2Ocol_trop','COcol_trop','lat','month','AODVIS']
        
    
    print(total_input_vars)
    if opt == 'sat':
        pred_sat_nn_models(cesm2_model, total_input_vars, feature_sel, 'co')
        pred_sat_nn_models(cesm2_model, total_input_vars, feature_sel, 'no2')
        pred_sat_nn_models(cesm2_model, total_input_vars, feature_sel, 'hcho')
        pred_sat_nn_models(cesm2_model, total_input_vars, feature_sel, 'no2_hcho_co')
        
        #pred_sat_nn_models(cesm2_model, total_input_vars, feature_sel, 'no2_hcho')
    elif opt == 'const':
        pred_ind_feature_nn_models(cesm2_model, total_input_vars, feature_sel, ['NO2col_trop'])
        pred_ind_feature_nn_models(cesm2_model, total_input_vars, feature_sel, ['CH2Ocol_trop'])
        pred_ind_feature_nn_models(cesm2_model, total_input_vars, feature_sel, ['COcol_trop'])
        pred_ind_feature_nn_models(cesm2_model, total_input_vars, feature_sel, ['NO2col_trop','CH2Ocol_trop','COcol_trop'])
        pred_ind_feature_nn_models(cesm2_model, total_input_vars, feature_sel, ['FLASHFRQ'])
        pred_ind_feature_nn_models(cesm2_model, total_input_vars, feature_sel, ['Q','T'])
    else:
        config_file = 'config_dense_3layers.json'
        train_nn_models(cesm2_model, total_input_vars, feature_sel, opt, config_file)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Step')
    parser.add_argument('-s', type=str, help='Train or perdict')
    parser.add_argument('-m', type=str, help='Specify the CESM2 model')
    parser.add_argument('-f', type=str, help='Specify the feature sets')
    args = parser.parse_args()
    main(args)
