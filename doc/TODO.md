

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

sequence. On-device training demo

- Make core logic into a class
- Add some tests
- Set LED 
- Set a threshold after training
- Check threshold
- Set output 
- Return to checking state after unlock
- Set an out pin based on detection 
- Test it out on device
- Make demo video
- Add some documentation / README

Examples. Add a novelty detection example?

Benchmarks

- Add FLASH and RAM usage
- Test gzip compression of .csv model
- Add a couple of different sized models to benchmark?
- Add another application/dataset for benchmark

In-browser demo

- Test MicroPython build for WASM/browser
