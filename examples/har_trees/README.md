
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

- Setup train+eval pipeline. Use tools from LeafClustering repo to download data
- Setup evaluation on a small dataset. Export data as .npy file. Runnable on host and device
- Setup end-2-end demo for a hardware device.
- Run the training + test/evaluation in CI
- Add instructions how to run to this README


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


