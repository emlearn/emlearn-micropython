# NOTE: MicroPython must be flashed before-hand
# and emlfft.mpy built

MPREMOTE='mpremote'

${MPREMOTE} cp src/emlfft/emlfft.mpy :
${MPREMOTE} cp benchmarks/fft/fft_python.py :

${MPREMOTE} run benchmarks/fft/fft_benchmark.py

