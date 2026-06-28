# Fixed-Point Quantization Specification

This document defines the integer fixed-point formats and arithmetic specs used to compile the XGBoost edge model into synthesizable Verilog.

---

## 1. Quantization Performance

The edge model was quantized in Python and validated against the sealed test set:

| Model Version | ROC-AUC | Description |
| :--- | :--- | :--- |
| XGBoost Reference (Floating-Point) | **0.6736** | Original model trained in XGBoost |
| Floating-Point Replica | **0.6731** | Replicated inference in pure Python |
| **Quantized Integer Model** | **0.6729** | Fixed-point model using locked widths |

The fixed-point quantization results in a negligible ROC-AUC degradation of **0.0007**, which is well within cross-validation noise.

---

## 2. Number Formats

To ensure a simple and efficient hardware datapath, we use a uniform fixed-point format for the input features and thresholds, and a separate format for the leaf values.

| Signal | Scaling Factor | Range (Observed) | Bit-Width | Format |
| :--- | :--- | :--- | :--- | :--- |
| Input Features (`*_mean3`) | $2^4$ (x16) | 0 to ~282 | **14-bit** | Signed Integer (Q10.4) |
| Split Thresholds | $2^4$ (x16) | 8.17 to 190.67 | 14-bit | Signed Integer (Q10.4) |
| Leaf Values | $2^8$ (x256) | -0.3596 to +0.1569 | **8-bit** | Signed Integer (Q0.8) |
| Margin Accumulator | $2^8$ (x256) | -0.83 to +1.92 | **12-bit** | Signed Integer (Q4.8) |

Constants used in generator script:
- `FRAC_FEAT = 4` (Feature fractional bits)
- `FRAC_LEAF = 8` (Leaf fractional bits)

---

## 3. Decision Logic

The XGBoost model generates risk probabilities using the logistic sigmoid function:
$$p = \frac{1}{1 + e^{-\sum \text{leaves}}}$$

Since the sigmoid function is strictly monotonic, we can perform classification by comparing the raw margin accumulation directly to a threshold:
$$\text{alarm} = (\text{accumulator} \ge \text{ALARM\_THRESHOLD})$$

This design eliminates the need for complex exponential calculations or division units on the FPGA, saving logic resources.

---

## 4. Missing Value Handling

XGBoost handles missing input features natively by learning a default routing direction (left or right) for each split. 

In Verilog, we represent this by carrying a `valid` bit alongside each input feature:
- If `valid` is high, the comparator compares the feature value to the threshold.
- If `valid` is low (indicating a missing measurement), the multiplexer automatically routes the datapath to the learned default branch.
