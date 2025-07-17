## Downloading data

Here we provide the URLs to download files based on the sensor data package.
+ On Linux systems we recommend you use the command ```wget --content-disposition --trust-server-names -P <path to download to> -i <modality>_<data_package>.txt``` to automatically download our data.

+ If you wish to download the whole dataset, you can use the bash for loop:
```
for fn in *.txt; do wget --content-disposition --trust-server-names -P <path to download to> -i $fn; done
```

+ You can use the glob to download specific data e.g., ```images*.txt``` for all the images or ```*UAV1*.txt``` for all the data from UAV1.
