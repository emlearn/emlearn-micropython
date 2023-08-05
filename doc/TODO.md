

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

On-device training demo

- Implement time-between-event preprocessor. Try on PC, test on device
- Implement kNN in emlearn. int16_t first
- Test it out on device

In-browser demo

- Test MicroPython build for WASM/browser

Benchmarks

- Add FLASH and RAM usage
- Add a couple of different sized models to benchmark?
- Add another application/dataset for benchmark
