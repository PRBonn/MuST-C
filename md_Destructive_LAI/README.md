# Code for destructive LAI measurement

0. install necessary libraries and download sample data
```sh
pip install -r requirements.txt
wget https://www.ipb.uni-bonn.de/data/MuST-C/data/sample_data.zip
unzip sample_data

```
1. Print the calibration file and scan before each leaf scanning
You can modify the markers_list.json to customise the cultivars scanned.
```sh
cd markers
python generate_markers.py -o <output dir> -j <json filepath>
```
2. Prepare a directory with the leaf scans, and a separate directory for the calibration scans. See ./data/leaves and ./data/calibration for an example.
2. Set the area to crop out in data/calibration.yaml
```sh
python crop_and_threshold.py -d ./data/leaves -o ./data
```
3. Run calibration with the calibration dir to create calib.txt
```sh
python calibration_board_one_dir.py \
       -c ./data/calibration \
       -o ./data \
       --calib_fp data/calibration.yaml
```
4. (optional) Run vegetation masks. You can manually correct this mask if required.
```sh
mkdir ./data/vis_vm
python generate_veg_mask.py \
      --img_dir ./data/leaves \
      --calib_fp ./data/calibration.yaml \
      --out_dir ./data/vis_vm
```
5. Process leaves images to obtain leaf areas
```sh
python process_images.py -d data/leaves --is_vis --vm_dir ./data/vis_vm/labeller/semantics
```
You may need to adjust the tracker configuration in ./data/tracker_hyps.cfg to match your setup.
By adding the `--is_vis` flag, you can check the visualisation of the tracker in ./data/test_dbg and ./data/vis_frame.

6. Check the leaf area csv (la.csv)
We encode further information in the csv as follows:
- stddev=-1 means long leaf, so no repeated measurements for stddev calculations
- stddev=-2 means the leaf could not be fully seen (e.g. was placed too high or low in camera FOV)
