"""
Created on Thu Jan 26 11:45:02 2023

@author: e.chakhvashvili
"""

from easygui import *


import sys
import pandas as pd
from scipy.stats import linregress
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.plot
import rasterstats
import os
from matplotlib.ticker import FormatStrFormatter
 

#path where you have stored your ortho, shapefile and calibration file
#this is the only direct input required from the user
default_path=r"D:\Kevin\240723 Phenoroam\230601"

#asking user for file locations: shapefile, calibration file, ortho as
#well as the name of the reflectance product that will be saved. you do not
#need to add .tif ending to the name, it will be automatically added
ret_val = msgbox("You are embarking on the great path of empirical line correction. Prepare your panels, shapefiles and orthomosaic")
if ret_val is None: # User closed msgbox
    sys.exit(0)
    
shapefile = fileopenbox("Please select the shapefile (only the file with .shp extension)", default=default_path)
if shapefile is None:
    raise ValueError("Please select a valid shapefile.")
    
excel_file = fileopenbox("Please select the calibration file in excel format", default=default_path)
if excel_file is None:
    raise ValueError("Please select a valid excel file.")

ortho_file = fileopenbox("Please select the orthomosaic", default=default_path)
if ortho_file is None:
    raise ValueError("Please select a valid orthomosaic.")
if not ortho_file.endswith(".tif"):
    raise ValueError("Please select a valid orthomosaic. Should have .tif extension")
    
save_location = filesavebox("Choose the save location and type the name of the orthomosaic", default=default_path, filetypes=["*.tif"])
if save_location is None:
    raise ValueError("Please select a valid save location.")

if not save_location.endswith(".tif"):
    save_location += ".tif"

#selecting MicaSense camera model (5 or 10 channel)
cameramodel = choicebox('Select nubmer of bands', 'Selecting band numbers', [5,10])
cameramodel = int(cameramodel)



#this part plots the mean radiance extracted from each panel in each band
#and outputs a plot showing the mean and standard deviation of the extraction
#means and standard deviations are saved as variables for further processing
def plot_radiance(shapefile, ortholoc, cameramodel):
    bands = list(range(1, 6 if cameramodel == 5 else 11)) 
    n_panels = len(rasterstats.zonal_stats(shapefile, ortholoc, band=1, stats=['mean']))
    x = list(range(1, n_panels+1))
    global zoStamean, zoStastdv
    zoStamean, zoStastdv = [[f['mean'] for f in rasterstats.zonal_stats(shapefile, ortholoc, band=b, stats=['mean','std'])] 
                            for b in bands], [[f['std'] for f in rasterstats.zonal_stats(shapefile, ortholoc, band=b, stats=['mean','std'])] 
                                          for b in bands]
                                             
    if cameramodel == 5:                                              
        fig, axs = plt.subplots(1,5, sharey=True, sharex=True, figsize=(18,8))
    else:
        fig, axs = plt.subplots(2,5, sharey=True, sharex=True, figsize=(18,8))
    fig.add_subplot(111, frameon=False)

    for ax, bn in zip (axs.flat, range(cameramodel)):
        
        ax.plot(x, zoStamean[bn], 
                marker='.',
                linestyle='None',
                markersize = 15 , 
                markeredgecolor = 'black',
                markerfacecolor='w') 
        
        ax.errorbar(x, zoStamean[bn], 
                    zoStastdv[bn], 
                    linestyle='None', 
                    marker='^',ms=0.1, 
                    ecolor='red')
        
        ax.set_xticks(x)
        ax.set_title('Band ' + str(bn+1)) 
        
    plt.tick_params(labelcolor='none', 
                    which='both', 
                    top=False, 
                    bottom=False, 
                    left=False, 
                    right=False)
    
    plt.ylabel('Radiance', labelpad=15, fontsize=20)
    plt.xlabel('Panels', fontsize=20)  
    path = os.path.join(os.path.dirname(ortholoc), 'Plots')
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(os.path.join(path, "ZonalStatistics_Panels.png"), dpi=300)
    return zoStamean, zoStastdv

plot_radiance(shapefile, ortho_file, cameramodel)


#if standard deviation is smaller than 0.0001, remove the panels; this value
#can be changed
for i in range(len(zoStastdv)):
    for j in range(len(zoStastdv[i])):
        if zoStastdv[i][j] < 0.0001:  #change if needed
            zoStamean[i][j] = np.nan


#calibration file should contain the exact number of panels that was used
#during the overflight, and same excel template should be used every time (wavelength+bands columns)
def process_excel(excel_file, zoStamean):
    global result, dfsize, wavelengths
    df_excel = pd.read_excel(excel_file)
    df = pd.DataFrame(zoStamean, columns=df_excel.columns[1:])
    df1 = pd.read_excel(excel_file, index_col="Wavelength")
    result = pd.concat([df, df1], axis=0, sort=False)
    dfsize = int(result.shape[0] / 2)
    wavelengths = list(df1.index.values)
    return result, dfsize, wavelengths

process_excel(excel_file,zoStamean)

#plot title for empirical line correction
if "PlotTitle" in globals():
    pass
else:
    PlotTitle = enterbox("Please enter the empirical line correction plot title")

slope_list = []
intercept_list = []


def max_value(inputlist):
    return np.nanmax([sublist[-1] for sublist in inputlist])

#plotting linear regression and deriving slope, intercept for further
#processing
def plotregress(dataf):
    if cameramodel==5:       
        fig, axs = plt.subplots(nrows=1, ncols=5, sharey=True, sharex=True, figsize=(18,8))
    else:
        fig, axs = plt.subplots(nrows=2, ncols=5, sharey=True, sharex=True, figsize=(18,8))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    axs = axs.flatten()  

    for i, wave in zip(range(int(dataf.shape[0]/2)),wavelengths):
        b1 = dataf.iloc[i,:]
        b1nex = dataf.iloc[i+dfsize,:]
        mask = ~np.isnan(b1) & ~np.isnan(b1nex)
        slope, intercept, r_value, p_value, std_err = linregress(b1[mask],b1nex[mask])
        slope_list.append(slope)
        intercept_list.append(intercept)
        dataf.loc[b1.name, 'slope'] = slope
        dataf.loc[b1.name, 'intercept'] = intercept
        dataf.loc[b1.name, 'r_value'] = r_value
        dataf.loc[b1.name, 'p_value'] = p_value
        dataf.loc[b1.name, 'std_err'] = std_err
        axs[i].plot(b1, b1nex, "o", color="blue", markersize=7, 
                    markeredgewidth=1, 
                    markeredgecolor="blue",
                    markerfacecolor="None", zorder=2)
        
        axs[i].plot(b1, intercept + slope*b1, 'red', label='fitted line',zorder=1)
        axs[i].errorbar(b1,b1nex,std_err,alpha=0.0)
        x0 = b1.mean()
        y0 = slope*x0+intercept
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        axs[i].text(0.05,0.95,"y=%1.3fx%+1.3f\nr$^2$=%1.4f"%(slope,intercept,r_value**2), 
           size=9,
           verticalalignment='top', 
           horizontalalignment='left', 
           transform=axs[i].transAxes,
           bbox=props)
 
        plt.ylim(0,0.7)
        plt.xlim(0,max_value(zoStamean)+0.01)
        plt.locator_params(nbins=3)
        axs[i].set_title (wave)
        axs[i].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        fig.suptitle(PlotTitle, fontsize=15,weight='bold')
        fig.text(0.5, 0.07, "Radiance", ha='center',fontsize=12, weight='bold')
        fig.text(0.07, 0.5, "Reflectance factor", va='center', rotation='vertical', fontsize=12,weight='bold')
        path = os.path.join(os.path.dirname(ortho_file), 'Plots')
        if not os.path.exists(path):
                os.makedirs(path)
        plt.savefig(os.path.join(path, "ELC.png"),dpi=300)

plotregress(result)

 
#this snippet converts slope and intercept values into numpy array
sr = pd.DataFrame({'slope': slope_list, 'intercept': intercept_list})
arr = sr.values


#import orthoimage and assign nodata values
with rasterio.open(ortho_file) as src:
    cka=src.read()
    cka[cka==src.nodata]=np.nan
    cka[cka==1]=np.nan
    
#create empty array with same shape as the original 
corr=np.zeros((cka.shape[0],cka.shape[1],cka.shape[2]),dtype=np.float32)

#applying empirical line correction. every value below zero will be converted
#to 0.0001 and above 1 to 1
for k in range(0,cka.shape[0]):
    corr[k,:,:]=cka[k]
    corr[k,:,:]=(cka[k]*arr[k,0])+arr[k,1]
    corr[corr<0]=0.0001
    corr[corr>1]=1

#writing raster    
with rasterio.open(save_location, 'w', 
                   driver='GTiff',count=cameramodel,
                   width=cka.shape[2],
                   height=cka.shape[1],
                   dtype=rasterio.float32,
                   crs=src.crs,
                   transform=src.transform) as outf:
    #outf.write_mask(True)
    outf.write(corr)
    
src.close()
