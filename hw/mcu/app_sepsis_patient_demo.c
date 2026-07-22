/* On-device bedside-monitor replay for the Silicon Labs BRD2605A (SiWG917, Cortex-M4F).
 *
 * The MCU counterpart of the Vivado patient-streaming testbench (hw/tb_demo_patients.v)
 * and the Dash dashboard (demo/app.py): replays REAL held-out PhysioNet-2019 test
 * patients hour by hour through the flagship model running on silicon, printing the
 * calibrated sepsis risk and the alarm state each hour.
 *
 *   - Patient "SEPTIC":  the same held-out patient the dashboard replays; the risk
 *     must rise and trip the alarm hours before the sepsis warning label.
 *   - Patient "CONTROL": a non-septic held-out patient; the alarm must stay quiet.
 *
 * Risk = isotonic( sigmoid(margin) ) -- identical calibration + threshold to the
 * dashboard, with the isotonic map embedded as a piecewise-linear table.
 * Device risk is checked against the host-computed reference each hour.
 *
 * Replays pre-recorded vitals -- NOT a patient-connected system.
 */
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "sepsis_flagship_model.h"
#include "patient_stream.h"

/* piecewise-linear isotonic calibration (clips outside the table, like sklearn) */
static float iso_calibrate(float p_raw)
{
  if (p_raw <= ISO_X[0])         return ISO_Y[0];
  if (p_raw >= ISO_X[ISO_N - 1]) return ISO_Y[ISO_N - 1];
  int lo = 0, hi = ISO_N - 1;
  while (hi - lo > 1) {
    int mid = (lo + hi) >> 1;
    if (ISO_X[mid] <= p_raw) lo = mid; else hi = mid;
  }
  float t = (p_raw - ISO_X[lo]) / (ISO_X[hi] - ISO_X[lo]);
  return ISO_Y[lo] + t * (ISO_Y[hi] - ISO_Y[lo]);
}

/* print helpers that avoid %f (newlib-nano safe) */
static void print_vital(float v, int decimals)
{
  if (isnan(v)) { printf("  -- "); return; }
  if (decimals == 0) {
    printf("%3d ", (int)(v + 0.5f));
  } else {
    int scaled = (int)(v * 10.0f + 0.5f);
    printf("%2d.%1d ", scaled / 10, scaled % 10);
  }
}

static void print_risk(float r)
{
  int scaled = (int)(r * 1000.0f + 0.5f);
  printf("0.%03d", scaled);
}

static int stream_patient(const char *tag,
                          const float x[][SEPSIS_N_FEATURES],
                          const float display[][4],
                          const unsigned char *label,
                          const float *risk_ref,
                          int n_hours)
{
  printf("--- %s ---\r\n", tag);
  printf("hour   HR  Temp Resp  SBP |  risk  | ALARM label\r\n");
  int first_alarm = -1, first_label = -1, mismatches = 0;
  for (int h = 0; h < n_hours; ++h) {
    float margin = sepsis_predict_margin(x[h]);
    float p_raw  = 1.0f / (1.0f + expf(-margin));
    float risk   = iso_calibrate(p_raw);
    int alarm    = (risk >= SEPSIS_ALARM_THRESH);

    if (fabsf(risk - risk_ref[h]) > 1e-3f) mismatches++;
    if (alarm && first_alarm < 0)    first_alarm = h;
    if (label[h] && first_label < 0) first_label = h;

    printf(" h%02d  ", h);
    print_vital(display[h][0], 0);   /* HR   */
    print_vital(display[h][1], 1);   /* Temp */
    print_vital(display[h][2], 0);   /* Resp */
    print_vital(display[h][3], 0);   /* SBP  */
    printf("| ");
    print_risk(risk);
    printf(" |   %d     %d", alarm, label[h]);
    if (alarm && h == first_alarm)   printf("  <-- ALARM raised");
    if (label[h] && h == first_label) printf("  <-- sepsis warning label");
    printf("\r\n");
  }
  printf("device-vs-host risk mismatches: %d/%d\r\n", mismatches, n_hours);
  if (first_alarm >= 0 && first_label >= 0) {
    printf("first alarm h%d | first warning label h%d -> %d h early warning on silicon\r\n",
           first_alarm, first_label, first_label - first_alarm);
  } else if (first_alarm < 0) {
    printf("alarm stayed quiet for all %d h (correct: non-septic patient)\r\n", n_hours);
  }
  printf("\r\n");
  return mismatches;
}

void sepsis_patient_demo_run(void)
{
  printf("\r\n=== BEDSIDE MONITOR REPLAY: held-out ICU patients on silicon ===\r\n");
  printf("model: flagship (%d trees, %d features) | calibrated risk, alarm thresh ",
         SEPSIS_N_TREES, SEPSIS_N_FEATURES);
  print_risk(SEPSIS_ALARM_THRESH);
  printf("\r\n\r\n");

  int mm = 0;
  mm += stream_patient("Patient A: SEPTIC (held-out test patient 11575)",
                       SEPTIC_X, SEPTIC_DISPLAY, SEPTIC_LABEL, SEPTIC_RISK_REF,
                       SEPTIC_N_HOURS);
  mm += stream_patient("Patient B: CONTROL, non-septic (held-out test patient 99)",
                       CONTROL_X, CONTROL_DISPLAY, CONTROL_LABEL, CONTROL_RISK_REF,
                       CONTROL_N_HOURS);
  printf("=== replay done | total device-vs-host mismatches: %d ===\r\n", mm);
}
