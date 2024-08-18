

# User journey

- Level 0a. Run a pretrained example/demo in the browser
- Level 0b. Run a pretrained example/demo on a board
- Level 1. Train custom model on-device
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

- Record piezo data with ADC. 100 Hz?
Typical taps. Slower pushes. Handling noises.
- Setup event detection for piezo.
In its own module.
Threshold on delta and level?
- Create emliir module, use for piezo detection
- Maybe blink during unlocked state
- Add a blink to each event. For user feedback
- Make demo video
- Make state diagram
- Make timing diagram. Highlight distances/features
- Add some documentation / README

Learnings.

- Putting piezo on small thin plate worked well.
On table not working, no response.
Hitting direct not so good either, rise of finger causes change. Double-trigger. Also tricky to hit in right place.
- LEDs as protection diodes worked well. Both red and green can be used. Lights up on direct hits, if placed by piezo.
- Analog RC filter is beneficial for piezo connections. Using 10k+100nF, has 160 Hz cutoff. Should maybe move it to 80Hz? Since only sampling at 100 Hz. 
- Only direct hits can reach trigger levels on 3.3V I/O. Need ADC for other cases. But am seing some 100mV when placed on small plate

Examples

- Add a novelty detection example?

Benchmarks

- Add FLASH and RAM usage
- Test gzip compression of .csv model for trees
- Add a couple of different sized models to benchmark?
- Add another application/dataset for benchmark

In-browser demo

- Test MicroPython build for WASM/browser
