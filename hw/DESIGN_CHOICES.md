# Hardware Design Choices & Trade-off Analysis

This document details the design decisions and architectural trade-offs made when porting the sepsis early-warning model into Verilog.

---

## 1. Feature vs. Tree Complexity

In hardware design for tree ensembles on FPGAs, the logic footprint is driven by two main factors:
1. **Decision Tree Depth:** Capped at depth $d$, each tree requires up to $2^d - 1$ comparators. Footprint scales exponentially with depth:
   - **Depth 4:** 15 comparators per tree (375 comparators for 25 trees)
   - **Depth 5:** 31 comparators per tree
   - **Depth 6:** 63 comparators per tree
   - **Depth 10:** 1,023 comparators per tree
2. **Feature Complexity:** Calculating rolling features in hardware requires registers and buffers. A larger feature space requires a wider datapath and increases routing complexity.

Our evaluation showed that deep trees on a small feature set quickly lead to overfitting. Capping the edge model at **depth 4** provides optimal generalization performance while keeping the hardware footprint extremely small (375 comparators).

---

## 2. Platform Suitability (FPGA vs. MCU)

When deploying tree ensembles to a microcontroller (e.g. ESP32), evaluation cost scales linearly with the number of trees, as trees are evaluated sequentially in software. Evaluating a 366-tree ensemble requires only a few microseconds, making both our server and edge models computationally feasible on standard MCUs.

Deploying to an FPGA, however, allows **deterministic, parallel, single-cycle inference** and provides a pathway for ultra-low-power, battery-operated clinical sensors. This implementation serves as a scalable template for compilation of tree models directly into combinational Verilog.

---

## 3. Two-Model Architecture & Verification

We trained two separate models to analyze the performance-to-hardware trade-off:

| Parameter | Edge Model (RTL Engine) | Server Model |
| :--- | :--- | :--- |
| **Feature Count** | 4 (Temp, Resp, HR, SBP) | 40 (All vitals & deltas) |
| **Tree Count** | 25 | 366 |
| **Max Tree Depth** | 4 | 5 |
| **Logic Size** | **375 comparators** | **10,581 comparators** |
| **Test ROC-AUC** | **0.674** | **0.722** |

### Key Trade-offs:
1. **Verification Tractability:** Guaranteeing bit-exact correctness between the Python model and the Verilog engine is a primary goal. Modeling 4 simple 3-hour rolling averages in hardware is highly verifiable. Modeling the full 40-feature server model (including 6-hour windows, deltas, missingness indicators, and variable-divisor rolling means) in Verilog is significantly more complex and prone to implementation mismatches.
2. **Logic Utilization:** The edge model is ~28x smaller than the server model, allowing it to fit on the smallest, lowest-cost FPGAs.
3. **Interpretability:** Monitoring 4 key vital trends matches clinical intuition and simplifies the explainability of the generated alarms.

The 0.048 ROC-AUC difference represents the measured accuracy cost of utilizing a simplified, highly verifiable bedside hardware design.
