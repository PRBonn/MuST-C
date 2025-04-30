## Plotting graphs from the paper

To run these scripts, you must first: 

1. Download the [metadata]( https://bonndata.uni-bonn.de/api/access/datafile/:persistentId?persistentId=doi:10.60507/FK2/OX9XTM/K5JOHB) and unzip the file into $PARENT_DIR
1. Install the requirements and the dev kit (I tested on Python 3.8) 
    ```sh
    pip3 install -r requirements.txt 
    pip3 install ../dev_kit
    ```

### SunScan LAI against days sown
```sh
python plot_sunscan.py -p $PARENT_DIR -o <output_dir>
```

## Biomass against days sown
```sh
python plot_biomass.py -p $PARENT_DIR -o <output_dir>
```

## Destructive LAI per plant
```sh
python plot_des_lai.py -p $PARENT_DIR -o <output_dir>
```

## SunScan vs Destructive LAI R<sup>2</sup>
```sh
python plot_r2.py -p $PARENT_DIR -o <output_dir>
```


