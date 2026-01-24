import numpy as np
import numpy as np
import sys
import os
import  argparse
import datetime
import numpy as np
import pandas as pd
import h5py
import matplotlib as mpl 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature

import warnings
warnings.simplefilter("ignore")

from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='multi_input_combination_models_new_fusion_encoder_temporal_attention_dynamic_loss',
                    help='the path for saving models')
args = parser.parse_args()

ncolors = 256
color_array = plt.get_cmap('jet')(range(ncolors))
map_object = LinearSegmentedColormap.from_list(name='jet_alpha',colors=color_array)
plt.register_cmap(cmap=map_object)
plnbis = mcolors.Normalize(vmin=0,vmax=0.000008)
def DenseScatter_insitu(Part2Y,Part2X, title, path):
    xy = np.vstack([Part2X,Part2Y])
    z = gaussian_kde(xy)(xy)
    
    idx = z.argsort()
    Part2X, Part2Y, z = Part2X[idx], Part2Y[idx], z[idx]
    
    r2=np.corrcoef(Part2X,Part2Y)[1,0]**2
    bias = np.abs(np.mean(Part2Y-Part2X))
    mad = np.mean(np.absolute(Part2X-Part2Y))
    rmse = np.sqrt(np.mean((Part2X-Part2Y)**2))
    nrmse=rmse/np.mean(Part2X)
    
    img = plt.scatter(Part2X, Part2Y, c=z, s=15, cmap="jet_alpha",norm=plnbis)
    plt.xlim((0, 1200))
    plt.ylim((0, 1200))
    plt.xlabel('Measured (W/$m^2$)',size=18)
    plt.ylabel('Predicted (W/$m^2$)',size=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.plot(np.arange(1250),'r')
    plt.axis([0,1210,0,1210])
    t = plt.text(100,870,'$R^2$={:.3f}\nRMSE={:.2f}'.format(r2,rmse),fontsize=18)
    t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='white'))
    if title.startswith('SolarFormer'):
        plt.title(title, size=20, c='r')
    else:
        plt.title(title, size=20)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(path + title + '-instantaneous.png',bbox_inches = 'tight', pad_inches=0.02,dpi=600)
    plt.clf()
    with open(path + 'output.txt', 'a') as f:
    return r2,bias,rmse,nrmse

def denseScatterHist1200(cp,tp,title, path):
    r2=np.corrcoef(cp,tp)[1,0]**2
    bias = np.abs(np.mean(cp-tp))
    rmse = np.sqrt(np.mean((cp-tp)**2))
    nrmse=rmse/np.mean(tp)
    n=cp.shape[0]
    
    hist, xbins, ybins = np.histogram2d(tp,cp,(60, 60))
    extent = [xbins.min(),xbins.max(),ybins.min(),ybins.max()]
    fig, ax = plt.subplots()
    fig.set_size_inches(6, 6)
    plt.imshow(np.sqrt(np.ma.masked_where(hist == 0, hist).T),cmap='jet', origin='lower', extent=extent)
    ax.tick_params(labelsize=16)
    plt.xlabel('Measured DSR',fontsize=18)
    plt.ylabel('Estimated DSR',fontsize=18)
    plt.yticks(np.arange(0,1201,100))
    plt.plot(np.arange(1200),'r')
    plt.axis([0,1210,0,1210])
    plt.text(100,800,'$R^2$={:.3f}\nRMSE={:.2f}'.format(r2,rmse),fontsize=18, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
    plt.title(title,fontsize=18)
    plt.savefig(path + title + '-instantaneous.png',bbox_inches = 'tight', pad_inches=0.02,dpi=600)
    plt.clf()

    with open(path + 'output.txt', 'a') as f:
        print(r2,bias,rmse,nrmse, file=f)
    return r2,bias,rmse,nrmse,n

states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none')
DEFAULT_DATA_HOME = '/data/geonex_sat'
DEFAULT_TILELIST   = DEFAULT_DATA_HOME + '/solarcube_sitelist.csv'
DEFAULT_INSITU = DEFAULT_DATA_HOME+'/solarcube_insitu.csv'
TYPES   = ['vis047','vis086','ir133','ssr','sza','cm','insitu']

def MSE(pred, true):
    return np.mean((pred-true)**2)
    
def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MBD(pred, true):
    return np.mean((pred - true))

def compute(predictions, gts):
    rRMSE_scores = 0
    rMBD_scores = 0
    b, t, c = predictions.shape
    average = np.mean(gts)
    for i_batch in range(b):
        for temporal_index in range(t):
            rRMSE_scores += RMSE(predictions[i_batch][temporal_index], gts[i_batch][temporal_index]) / average
            rMBD_scores += MBD(predictions[i_batch][temporal_index], gts[i_batch][temporal_index]) / average
    return rRMSE_scores /(b*t), rMBD_scores /(b*t)

colors = [(0, 0, 1), (1, 1, 0), (1, 0.5, 0)] 
n_bins = 100  
cmap_name = 'three_color_cmap'
ir_cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

pre_new_combination = np.load(args.data_path+'new_combination_pred.npy')*279.49 + 193.44
pre_new_combination_5 = np.load(args.data_path+'new_combination_pred_5.npy')*279.49 + 193.44
pre_new_combination_7 = np.load(args.data_path+'new_combination_pred_7.npy')*279.49 + 193.44
pre_new_combination_9 = np.load(args.data_path+'new_combination_pred_9.npy')*279.49 + 193.44
pre_new_combination_18 = np.load(args.data_path+'new_combination_pred_18.npy')*279.49 + 193.44

gt_new_combination = (np.load(args.data_path+'new_combination_gt.npy')*279.49 + 193.44)
gt_new_combination_5 = (np.load(args.data_path+'new_combination_gt_5.npy')*279.49 + 193.44)
gt_new_combination_7 = (np.load(args.data_path+'new_combination_gt_7.npy')*279.49 + 193.44)
gt_new_combination_9 = (np.load(args.data_path+'new_combination_gt_9.npy')*279.49 + 193.44)
gt_new_combination_18 = (np.load(args.data_path+'new_combination_gt_18.npy')*279.49 + 193.44)

pred_combined = np.concatenate([
    pre_new_combination_5,
    pre_new_combination_7,
    pre_new_combination_9,
    pre_new_combination_18
], axis=0)

gt_combined = np.concatenate([
    gt_new_combination_5,
    gt_new_combination_7,
    gt_new_combination_9,
    gt_new_combination_18
], axis=0)

import numpy as np
np.save(args.data_path + 'pred_combined.npy', pred_combined)
np.save(args.data_path + 'gt_combined.npy', gt_combined)

x = DenseScatter_insitu(pre_new_combination_5.flatten(), gt_new_combination_5.flatten(), 'station_5', args.data_path)
x = DenseScatter_insitu(pre_new_combination_7.flatten(), gt_new_combination_7.flatten(), 'station_7', args.data_path)
x = DenseScatter_insitu(pre_new_combination_9.flatten(), gt_new_combination_9.flatten(), 'station_9', args.data_path)
x = DenseScatter_insitu(pre_new_combination_18.flatten(), gt_new_combination_18.flatten(), 'station_18', args.data_path)

if args.data_path.strip()[-1] == '1':
    x = DenseScatter_insitu(pred_combined.flatten(), gt_combined.flatten(), 'LSTM-att+SatSolarCast', args.data_path)
else:
    x = DenseScatter_insitu(pred_combined.flatten(), gt_combined.flatten(), 'LSTM-att', args.data_path)

with open(args.data_path + 'output.txt', 'a') as f:
        print('\n', file=f)