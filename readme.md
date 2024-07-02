# Car Object Detection - YOLOV8 Simple Pipeline

Original Datasets from: TJHSST/Car Object Detection Datasets

## Dataset Preparation

The dataset preparation script is the jupyter notbook file: <ins>**data.ipynb**</ins>.\
You can pretty much refer to this data pipeline for all YOLOv8 dataset conversion.

## Metrics
Baseline Model: YOLOV8n\
Epochs Ran: 300

**The first 200 Epochs:**
![results](https://github.com/csf233csf/car-detection-project/assets/56235101/5d854fe0-0bc8-4c82-ada6-db526cb4f661)

**The last 100 Epochs (Resumed Training):**
![results](https://github.com/csf233csf/car-detection-project/assets/56235101/bffa51c6-9440-423c-b688-146e17297808)

## Stats and Speed:
**Speed:**
![image](https://github.com/csf233csf/car-detection-project/assets/56235101/b1879dff-9947-4a72-8137-2b61f1d1e76e)

**Pred Result:**
![image](https://github.com/csf233csf/car-detection-project/assets/56235101/e1c93cd7-16ba-479d-8981-f9ccd964cd67)

## Talk is cheap, show me the weights

Model is avaliable here: **https://huggingface.co/kanoml/car-detection-yolo**

# API USAGE

**To Start the Flask API:**
```sh
python3 server2.py
```
**Example Usage:**
```sh
python3 api_test.py
```


## How to use this Train script:

1. Verify your CUDA and environment (If needed)
```sh
python3 cudaverification.py
```

3. Install the requirements
```sh
pip install ultralytics
```

5. Run the training script
```sh
python3 train.py
```

6. Resume the training from a checkpoint
```sh
python3 resume.py
```



