
# Human Activity Recognition with tree-based models

This uses an approach based on the paper
[Are Microcontrollers Ready for Deep Learning-Based Human Activity Recognition?](https://www.mdpi.com/2079-9292/10/21/2640).

For each time-window, time-based statistical features are computed,
and then classified with a RandomForest model.
The window size used is 128 samples, 

It is tested on the PAMAP2 and UCI-HAR datasets,
using a wrist-mounted accelerometer.

## Status
**Work in Progress**

Feature extraction in MicroPython has been implemented.

Feature extraction takes 

## TODO

- Setup conversion and export of emlearn RF model
- Setup evaluation on a small dataset. Run on device
- Setup end-2-end demo for a hardware device. M5StickC
- Run the training + test/evaluation in CI
- Add instructions how to run to this README


## Run on device

`FIXME: not implemented`

## Run training

Install requirements
```
pip install -r requirements.txt
```

Download training data
```
python -m leaf_clustering.data.har.uci
```

Run training process
```
python har_train.py
```


