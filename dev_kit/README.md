## Developer's kit for MuST-C data set

In here, we provide Python scripts to perform some simple tasks we think are useful for our data set.
We also include some useful utils which you can install via pip:
```sh
pip install -r requirements.txt
pip install .
```

### Extracting data of a specific location
This script can crop out a specific region of interest centred around given locations (UTM32N) 
and of some specified size from (RGB and multispectral) raster data and point clouds.

1. Download whichever parts of the [dataset](https://www.ipb.uni-bonn.de/data/MUST-C/) that you need to $PARENT_DIR.
You will also need to also download the metadata ($PARENT_DIR/metadata) for the full functionality of the script. 
2. List the locations of interest in the csv, see example poi.csv for formatting.
3. List the types of data you want for all the points in another csv, see example data.csv for formatting.
4. Run the script. This might take some time depending on the number of locations and packages.
```sh
python extract_data.py \
       -p $PARENT_DIR \
       -o <output dir> \
       -i <path to poi.csv> \
       -d <path to data.csv> \
       -c <roi_size>
```

### Calculation of vegetation indices from multi-spectral data
Get cropped vegetation indices (ndvi, ndre, evi, osavi) for a specific location in UTM coordinates (the images will be saved to the current working path):
```sh
python get_vegetation_indices_ms.py \
       -p <$PARENT_DIR/raster_data/UAV3-MS/XX.tif> \
       -x <UTM x> \
       -y <UTM y> \
       -c <clip size>
```

### Other useful functions
#### Cultivar of plots

```python
from utils.id_mapper import IdMapper

IdMapper.set_df()
cult = IdMapper.shp_id2cult(shp_id)  # get the cultivar as a string from the plot id 
```

#### Sowing dates of cultivars
Get sowing dates for each cultivar 
```python
from utils.sowing_dates import sowing_dates

date_str_yyymmdd = sowing_dates["Sugar Beet"]
```
