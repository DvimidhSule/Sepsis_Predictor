# Physical Hardware Deployment — Silicon Labs BRD2605A (SiWG917)

## Background

Through a Silicon Labs workshop attended during a CFHE internship, I got hands-on
access to a Silicon Labs BRD2605A (SiWG917) evaluation kit (Cortex-M4). This document
covers extending the edge model from FPGA simulation to physical embedded deployment.

## Why a second hardware target

The existing `hw/` pipeline compiles the quantized edge model to Verilog, verified
in RTL simulation (Vivado/Icarus) — proving the model *could* run on an FPGA/ASIC.
This extension adds a second target: the same quantized model compiled to C and
run on physical Cortex-M4 silicon. Two honest, distinct claims:

- **FPGA path:** compiled to Verilog, verified in RTL simulation — no physical board.
- **MCU path:** compiled to C, running on physical BRD2605A (SiWG917) hardware — verified
  via measured on-device inference latency.

Both are generated from the same quantized tree structure, so the two targets are
provably the same model, not two different implementations.

## What's added

- `hw/generate_c.py` — emits a dependency-free C inference function from the same
  quantized edge model used by `generate_verilog.py`.
- Bit-exact validation against `hw/golden_vectors.csv` (the same test vectors used
  in Verilog co-simulation) before any hardware flashing.
- A Simplicity Studio project targeting the BRD2605A (SiWG917), flashing the generated
  inference code and streaming per-sample timing over serial.
- Photos of the board running the deployed model, and a captured serial log, in
  `figures/`.

## Scope and honesty notes

- This demonstrates inference running correctly on target hardware using
  pre-recorded test vectors — it is not a patient-connected or clinically
  deployed system.
- The board is on loan via the Silicon Labs workshop, not permanently owned;
  see acknowledgments.

## Acknowledgments

Board access provided by Silicon Labs via a hardware workshop conducted during
a CFHE internship.
