

# User journey

- Level 0a. Run a pretrained example/demo in the browser
- Level 0b. Run a pretrained example/demo on a board
- Level 1. Train on-device
- Level 2. Collect a dataset, do training on PC, deploy back to microcontroller
- Level 3. Bake the custom model into the firmware

# Milestones

- Can run example in browser
- Can run on-device training example
- First demo video published
- First test by other users
- First course held

# TODO

emtrees cleanup

- Add some basic automated testing

emlneighbors cleanup

- Add a release step.
- Make API be OOP
- Implement teardown/destroy
- Add automated test
- MAYBE. Add a `load_model` function to module


sequence. On-device training demo

- Add logic for entering training mode. Long-press. Use a fixed number of samples. 5?
- Update dataset. Sanity check performance
- Implement the prediction mode on host
- Test it out on host
- Test it out on device
- Make demo video
- Add some documentation / README


Benchmarks

- Add FLASH and RAM usage
- Test gzip compression of .csv model
- Add a couple of different sized models to benchmark?
- Add another application/dataset for benchmark

In-browser demo

- Test MicroPython build for WASM/browser
