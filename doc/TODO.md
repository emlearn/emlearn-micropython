

# User journey

- Level 0a. Run a pretrained example/demo in the browser
- Level 0b. Run a pretrained example/demo on PC/host
- Level 0c. Run a pretrained example/demo on a board
- Level 1. Train custom model on-device
- Level 2. Collect a dataset, do training on PC, deploy back to microcontroller
- Level 3. Bake the custom model into the firmware

# Milestones

- First demo video published. DONE, toothbrush
- First test by other users. DONE, Cornell university
- Complete example for full flow. Data collect, train, deploy. DONE, HAR
- First course held
- Can run on-device training example
- Can run example in browser


# TODO

sequence. On-device training demo

- Use accelerometer instead of piezo. On M5StickC, for example
- Compute impulsive-ness feature. Magnitude, RMS, exponential smooth, then Delta * times level ?
- Alternative: Use IIR for knock detection
- Maybe blink during unlocked state
- Add a blink to each event. For user feedback
- Make demo video
- Add some documentation / README
- Make state diagram
- Make timing diagram. Highlight distances/features

#### Examples

- Add a novelty detection example?

#### Benchmarks

- Add FLASH and RAM usage
- Test gzip compression of .csv model for trees
- Add a couple of different sized models to benchmark?
- Add another application/dataset for benchmark

#### In-browser demo

- Test MicroPython build for WASM/browser.
Requires user/external C module build support.
https://github.com/emlearn/emlearn-micropython/issues/18
- Test getting audio input into MicroPython Webassembly
- Test getting IMU data (ie on phone), in browser
