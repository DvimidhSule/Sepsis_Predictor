# MCU Deployment — Flagship Sepsis Model on BRD2605A (SiWG917)

Runs the **full 40-feature flagship model** on physical Cortex-M4F silicon. Unlike the
gate-bound FPGA edge model (4 features, compiled to combinational Verilog), the M4 has a
hardware FPU and megabytes of flash, so the entire model runs directly in float C.

## Files

| File | Purpose | Generated? |
|---|---|---|
| `sepsis_flagship_model.h/.c` | Float inference compiled from `models/sepsis_booster.json` | yes — `hw/generate_c.py` |
| `test_vectors.h` | 64 embedded test vectors + host-reference margins | yes — `hw/generate_c.py` |
| `app_sepsis_benchmark.c` | On-device timing + serial-print harness (DWT cycle counter) | no |
| `validate_c.c` | Host-side bit-agreement check vs xgboost | no |

Regenerate the model + vectors any time with:

```bash
python hw/generate_c.py
```

## 1. Host bit-check (before flashing)

Confirm the generated C reproduces the Python model:

```bash
gcc -O2 -I hw/mcu hw/mcu/validate_c.c hw/mcu/sepsis_flagship_model.c -lm -o hw/mcu/validate_c
./hw/mcu/validate_c hw/flagship_golden_vectors.csv
```

Expect `max |margin diff|` on the order of 1e-6 and **zero** decision mismatches.

## 2. Build for the board (Simplicity Studio)

1. **New project** → *Empty C Project* (or *Platform - Blink Baremetal*) targeting
   board **BRD2605A** / part **SiWG917M111MGTBA**.
2. **Add sources**: copy `sepsis_flagship_model.c`, `sepsis_flagship_model.h`,
   `test_vectors.h`, and `app_sepsis_benchmark.c` into the project.
3. **Enable serial printf**: in the *Software Components* view install
   **Services → IO Stream → IO Stream: USART** (instance `vcom`) and
   **Third Party → Tiny printf** (or *IO Stream: STDIO*), so `printf` is retargeted to
   VCOM. The Cortex-M4F FPU is enabled by default.
4. **Call the benchmark**: from `app_init()` (or `main()`), call
   ```c
   void sepsis_benchmark_run(void);
   sepsis_benchmark_run();
   ```
5. **Build & flash** (Ctrl+B, then Flash / Run).

## 3. Capture the on-device log

- Open the VCOM serial console (Simplicity Studio's *Console*, or PuTTY/Tera Term) at
  **115200 8N1**.
- Reset the board; the harness prints per-vector margin, cycle count, and µs, then a
  summary (min/avg/max latency, throughput, on-device vs host mismatches).
- Save the serial log and a photo of the board to `figures/` for the README
  (e.g. `figures/mcu_serial_log.txt`, `figures/mcu_board.jpg`).

## Notes

- Timing uses the DWT cycle counter (`DWT_CYCCNT`); latency µs assumes the M4 core at
  180 MHz (PS4). If you run PS3 (90 MHz) etc., set `-DSEPSIS_CORE_MHZ=90`.
- This benchmarks inference on **pre-recorded test vectors** — it is not a
  patient-connected or clinically deployed system.
