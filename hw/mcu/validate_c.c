/* Host-side bit-agreement check for the generated flagship C model.
 *
 * Reads hw/flagship_golden_vectors.csv (40 feature columns + xgb_margin + xgb_proba),
 * runs the generated sepsis_predict_margin/proba, and reports the max absolute
 * difference vs the Python/xgboost float model. A float model compiled straight from
 * the same trees should agree to within floating-point rounding (~1e-6), and every
 * thresholded prediction should be identical.
 *
 * Build (from repo root):
 *   gcc -O2 -I hw/mcu hw/mcu/validate_c.c hw/mcu/sepsis_flagship_model.c -lm -o hw/mcu/validate_c
 *   ./hw/mcu/validate_c hw/flagship_golden_vectors.csv
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "sepsis_flagship_model.h"

#define MAXLINE 8192

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1] : "hw/flagship_golden_vectors.csv";
    FILE *fp = fopen(path, "r");
    if (!fp) { fprintf(stderr, "cannot open %s\n", path); return 1; }

    char line[MAXLINE];
    if (!fgets(line, sizeof line, fp)) { fprintf(stderr, "empty file\n"); return 1; }  /* header */

    double max_margin_diff = 0.0, max_proba_diff = 0.0;
    long n = 0, pred_mismatch = 0;

    while (fgets(line, sizeof line, fp)) {
        float x[SEPSIS_N_FEATURES];
        float xgb_margin = 0.f, xgb_proba = 0.f;
        int col = 0;
        char *tok = strtok(line, ",\n");
        while (tok) {
            if (col < SEPSIS_N_FEATURES) {
                x[col] = (tok[0] == '\0') ? NAN : (float)atof(tok);
            } else if (col == SEPSIS_N_FEATURES) {
                xgb_margin = (float)atof(tok);
            } else if (col == SEPSIS_N_FEATURES + 1) {
                xgb_proba = (float)atof(tok);
            }
            col++;
            tok = strtok(NULL, ",\n");
        }
        if (col < SEPSIS_N_FEATURES + 2) continue;

        float m = sepsis_predict_margin(x);
        float p = sepsis_predict_proba(x);
        double md = fabs((double)m - (double)xgb_margin);
        double pd = fabs((double)p - (double)xgb_proba);
        if (md > max_margin_diff) max_margin_diff = md;
        if (pd > max_proba_diff) max_proba_diff = pd;
        /* decision agreement at the natural 0.5-proba / 0-margin boundary */
        if ((m >= 0.f) != (xgb_margin >= 0.f)) pred_mismatch++;
        n++;
    }
    fclose(fp);

    printf("Validated %ld vectors\n", n);
    printf("max |margin diff| = %.3e\n", max_margin_diff);
    printf("max |proba  diff| = %.3e\n", max_proba_diff);
    printf("sign(margin) mismatches = %ld / %ld\n", pred_mismatch, n);
    int ok = (max_margin_diff < 1e-3) && (pred_mismatch == 0);
    printf("%s\n", ok ? "PASS: C float model matches the Python model." : "FAIL");
    return ok ? 0 : 2;
}
