# NOTE: MicroPython must be flashed before-hand
# and emlearn_fft.mpy built

MPREMOTE='mpremote'

${MPREMOTE} cp src/emlearn_fft/emlearn_fft.mpy :
${MPREMOTE} cp benchmarks/fft/fft_python.py :

${MPREMOTE} run benchmarks/fft/fft_benchmark.py

