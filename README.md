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
Our dataset comprises data from multiple sensors over multiple days and multiple crops, and is useful for multiple tasks.
In this example, we show how you can quickly start by performing a simple data processing: sorting all data into plot-level data for a given plot.

1. Download sample data
We provide a sample of our dataset for you to quickly download, check, and develop with our dataset.  
[Click here to download the sample data.](https://bonndata.uni-bonn.de/api/access/datafile/:persistentId?persistentId=doi:10.60507/FK2/OX9XTM/YDODS9&key=86b0cc03-24d8-4129-ac31-3ecbdadd60fd)  
[Metadata about what is included in the sample data is here.](#sample-data)  
2. Uncompress the downloaded ``sample.zip'' to where you want your dataset to be extracted.
  You should get a directory structure like (which is the same the structure of the complete dataset):
```
MuST-C
└───images
└───point_clouds
└───raster_data
└───LAI_biomass_and_metadata
```
3. Clone this repo and install the dev kit. We recommend using a virtual environment or docker for this.
```bash
git clone https://github.com/PRBonn/MuST-C.git
cd MuST-C/dev_kit
pip install -r requirements.txt
pip install .
```
4. Run the script to extract the data for the plot of id 198 (this is a sugar beet plot) to `output_dir`. This will process all the sensors present in the parent_dir:
```bash
python3 get_plot_data.py \
        --parent_dir <path to downloaded MuST-C> \
        --output_dir <path to extracted plot-wise> \
        --plot_id 198
```
OR: if you are looking for a specific sensor, you can specify the sensor like this:
```bash
python3 get_plot_data.py \
        --parent_dir <path to downloaded MuST-C> \
        --output_dir <path to extracted plot-wise> \
        --plot_id 198 \
        --uav1-rgb \
        --uav2-rgb \
        --uav3-rgb \
        --uav3-ms \
        --ugv-rgb \
        --uav2-lidar \
        --ugv-lmi \
        --ugv-ouster
```
This will process all data from the specified sensor. For example, using the flag `uav1-rgb` the script will process all the images, pointclouds, and raster data, present in `parent_dir`, and will skip any missing files.

## Dataset Download
You can download parts of the dataset using the project [website](https://www.ipb.uni-bonn.de/data/MuST-C/)
or the full dataset (~4TB) from the [data repo](https://bonndata.uni-bonn.de/previewurl.xhtml?token=86b0cc03-24d8-4129-ac31-3ecbdadd60fd).
To use this code base, download the dataset into your desired $PARENT_DIR, while maintain the directory structure from the downloaded files:

![folder structure](./assets/folder_structure.svg)

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
We provide a sample of our dataset [here](https://bonndata.uni-bonn.de/api/access/datafile/:persistentId?persistentId=doi:10.60507/FK2/OX9XTM/0J1Y8F&key=86b0cc03-24d8-4129-ac31-3ecbdadd60fd).
The sample focuses on data from mid-June (around 14.06.2023) for the plot 198 of sugar beets.
To keep the filesize reasonaly small, we only extracted data of only the plot 198 with the exceptions of the point cloud from  **UAV2-Lidar** and all raster data which comprises the whole field for one date.
If you decide to subsequently download more of the dataset, you can seamlessly extract the new data into the same parent directory because this sample data follows the same directory structure as the complete dataset.

Specifically, this sample contains:
+ images of the plot 198 from **UAV1-RGB** (14.06.2023), **UAV2-RGB** (15.06.2023), **UAV3-RGB** (15.06.2023), **UAV3-MS** (15.06.2023), and **UGV-RGB** (13.06.2023) and their calibration files
+ point clouds of the plot 198 from **UAV1-RGB** (14.06.2023), **UAV2-Lidar** (15.06.2023), **UGV-LMI** (13.06.2023), and **UGV-Ouster** (13.06.2023)
+ raster data from **UAV2-RGB** (15.06.2023), **UAV3-RGB** (15.06.2023), and **UAV3-MS** (15.06.2023)
+ LAI from destructive measurements, LAI from SunScan, and Biomass combined into a single .csv file (LAI\_biomass\_and\_metadata/LAI\_biomass\_combined.csv) for the whole trial period
+ shapefile of the field trial (**md_FieldSHP**)

The sample.zip file is about 5 GB compressed. The uncompressed size is about 7 GB on disc. 

