# MuST-C: The Multi-Sensor and Multi-Temporal Dataset of Multiple Crops for In-Field Phenotyping and Monitoring


[**Website**](https://www.ipb.uni-bonn.de/data/MUST-C/) **|** [**Data Repo**](https://bonndata.uni-bonn.de/previewurl.xhtml?token=86b0cc03-24d8-4129-ac31-3ecbdadd60fd)

![MuST-C](assets/motivation2.png "MuST-C")
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

## Dataset Download
You can download parts of the dataset using the project [website](https://www.ipb.uni-bonn.de/data/MUST-C/)
or the full dataset (~4TB) from the [data repo](https://bonndata.uni-bonn.de/previewurl.xhtml?token=86b0cc03-24d8-4129-ac31-3ecbdadd60fd).
To use this code base, download the dataset into your desired $PARENT_DIR, while maintain the directory structure from the downloaded files:

![dir_struct](assets/folder_structure.png)


## Developers Kit
To use our data set, we provide a [developer's kit here](dev_kit),
where we share the scripts used to extract the data shown in our motivating figure (above) and other useful functions.

## Plotting Graphs
To reproduce the graphs in our paper, we provide the relevant [scripts here](plot_graphs_from_paper).

## Code Release
We also provide some scripts we used in the development of our data set:
* [scripts to extract LAI from a sequence of images here](md_Destructive_LAI)
* [script(s) to obtain multispectral reflectance from **UAV3-MS**](UAV-MS)


