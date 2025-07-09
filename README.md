# MuST-C: The Multi-Sensor and Multi-Temporal Dataset of Multiple Crops for In-Field Phenotyping and Monitoring


[**Website**](https://www.ipb.uni-bonn.de/data/MuST-C/) **|** [**Data Repo**](https://bonndata.uni-bonn.de/previewurl.xhtml?token=86b0cc03-24d8-4129-ac31-3ecbdadd60fd)

![MuST-C](https://github.com/user-attachments/assets/29f64697-294b-4087-9851-642c491566c6)
This repo contains code pertaining to the data set MuST-C, a mutli-sensor, multi-temporal and multiple crop dataset, consisting of data from sensors:   
* High resolution RGB camera
* Multispectral cameras (10 bands)
* 20x instantaneously-triggered cameras 
* RIEGL miniVUX-SYS LiDAR
* Ouster OS1 multi-beam LiDAR
* LMI laser triangulation scanners, 

and of multiple crops:
* sugar beets
* maize
* potato
* soy beans
* wheat
* wheat - faba bean intercrop

## Quick Start
Our dataset comprise data from multiple sensors over multiple days and multiple crops, and is useful for multiple tasks.
In this example, we show how you can quickly start by performing a simple data processing: sorting all data into plot-level data for a given plot.

1. Download sample data
We provide a sample of our dataset for you to quickly download, check, and develop with our dataset.  
[Click here to download the sample data.](https://bonndata.uni-bonn.de/api/access/datafile/:persistentId?persistentId=doi:10.60507/FK2/OX9XTM/0J1Y8F&key=86b0cc03-24d8-4129-ac31-3ecbdadd60fd)  
[Metadata about what is included in the sample data is here.](#sample-data)  
2. Uncompress the downloaded ``sample.zip'' to where you want your dataset to be extracted.
  You should get a directory structure like (which is the same the structure of the complete dataset):
```
MuST-C
│   sample.zip
└───images
│   └───UAV1-RGB
│       │   230614
│       └───calibrations
└───point_clouds
└───raster_data
    │   file021.txt
    │   file022.txt

```
3. Run the script to extract the data for the plot of id 198 (sugar beet plot)
```bash
python3 ...
```


## Dataset Download
You can download parts of the dataset using the project [website](https://www.ipb.uni-bonn.de/data/MuST-C/)
or the full dataset (~4TB) from the [data repo](https://bonndata.uni-bonn.de/previewurl.xhtml?token=86b0cc03-24d8-4129-ac31-3ecbdadd60fd).
To use this code base, download the dataset into your desired $PARENT_DIR, while maintain the directory structure from the downloaded files:

![dir_struct](https://github.com/user-attachments/assets/f868e8c7-c971-4882-9475-f7b44d3a1e99)


## Developers Kit
To use our data set, we provide a [developer's kit here](dev_kit),
where we share the scripts used to extract the data shown in our motivating figure (above) and other useful functions.

## Plotting Graphs
To reproduce the graphs in our paper, we provide the relevant [scripts here](plot_graphs_from_paper).

## Code Release
We also provide some scripts we used in the development of our data set:
* [scripts to extract LAI from a sequence of images here](md_Destructive_LAI)
* [script(s) to obtain multispectral reflectance from **UAV3-MS**](UAV3-MS)

## Sample Data
