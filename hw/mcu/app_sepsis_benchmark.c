/* On-device sepsis inference benchmark for the Silicon Labs BRD2605A (SiWG917, Cortex-M4F).
 *
 * Streams the flagship model over pre-recorded test vectors, times each inference with
 * the Cortex-M4 DWT cycle counter, verifies results against the host-reference margins,
 * and prints per-vector + summary timing over the VCOM debug UART (printf).
 *
 * Notes for embedded printf: avoids %f entirely (newlib-nano often ships with float
 * printf disabled) -- floats are printed as scaled integers. Latency is derived from
 * the runtime SystemCoreClock, not a hard-coded frequency.
 *
 * This measures inference on real silicon using pre-recorded vectors -- it is NOT a
 * patient-connected system (see HARDWARE_DEPLOYMENT.md).
 */
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "sepsis_flagship_model.h"
#include "test_vectors.h"

/* CMSIS core clock variable, updated by the SDK clock manager. */
extern uint32_t SystemCoreClock;

/* Cortex-M4 core debug / DWT registers for cycle-accurate timing. */
#define DEMCR      (*(volatile uint32_t *)0xE000EDFC)
#define DWT_CTRL   (*(volatile uint32_t *)0xE0001000)
#define DWT_CYCCNT (*(volatile uint32_t *)0xE0001004)
#define DEMCR_TRCENA       (1u << 24)
#define DWT_CTRL_CYCCNTENA (1u << 0)

static void dwt_init(void)
{
  DEMCR |= DEMCR_TRCENA;
  DWT_CYCCNT = 0;
  DWT_CTRL |= DWT_CTRL_CYCCNTENA;
}

/* print a float as a fixed-point decimal without %f (e.g. -3.14159 -> "-3.14159") */
static void print_f(const char *label, float v, const char *suffix)
{
  int32_t scaled = (int32_t)(v * 100000.0f + (v >= 0 ? 0.5f : -0.5f));
  int32_t ip = scaled / 100000;
  int32_t fp = scaled % 100000;
  if (fp < 0) fp = -fp;
  if (v < 0 && ip == 0) {
    printf("%s-0.%05ld%s", label, (long)fp, suffix);
  } else {
    printf("%s%ld.%05ld%s", label, (long)ip, (long)fp, suffix);
  }
}

void sepsis_benchmark_run(void)
{
  dwt_init();

  uint32_t mhz = SystemCoreClock / 1000000u;
  printf("\r\n=== Sepsis flagship model on BRD2605A (SiWG917, Cortex-M4F) ===\r\n");
  printf("model: %d trees, %d features, %d nodes | core %lu MHz\r\n",
         SEPSIS_N_TREES, SEPSIS_N_FEATURES, SEPSIS_N_NODES, (unsigned long)mhz);

  uint32_t min_c = 0xFFFFFFFFu, max_c = 0, sum_c = 0;
  int mismatches = 0;
  volatile float sink = 0.f; /* keep the optimizer honest */

  for (int i = 0; i < SEPSIS_N_VECTORS; ++i) {
    uint32_t t0 = DWT_CYCCNT;
    float m = sepsis_predict_margin(SEPSIS_TEST_X[i]);
    uint32_t t1 = DWT_CYCCNT;
    uint32_t cyc = t1 - t0;
    sink += m;

    if (cyc < min_c) min_c = cyc;
    if (cyc > max_c) max_c = cyc;
    sum_c += cyc;

    float ref = SEPSIS_TEST_MARGIN[i];
    if (fabsf(m - ref) > 1e-3f) mismatches++;

    /* latency in nanoseconds = cyc * 1000 / MHz  (integer math) */
    uint32_t ns = (uint32_t)(((uint64_t)cyc * 1000u) / mhz);
    printf("vec %2d: ", i);
    print_f("margin=", m, " ");
    print_f("(ref ", ref, ") | ");
    printf("%lu cyc | %lu.%02lu us\r\n",
           (unsigned long)cyc, (unsigned long)(ns / 1000u),
           (unsigned long)((ns % 1000u) / 10u));
  }

  uint32_t avg_c = sum_c / SEPSIS_N_VECTORS;
  uint32_t avg_ns = (uint32_t)(((uint64_t)avg_c * 1000u) / mhz);
  uint32_t min_ns = (uint32_t)(((uint64_t)min_c * 1000u) / mhz);
  uint32_t max_ns = (uint32_t)(((uint64_t)max_c * 1000u) / mhz);

  printf("---------------------------------------------------------------\r\n");
  printf("inferences: %d | on-device vs host-reference mismatches: %d\r\n",
         SEPSIS_N_VECTORS, mismatches);
  printf("cycles  min=%lu  avg=%lu  max=%lu\r\n",
         (unsigned long)min_c, (unsigned long)avg_c, (unsigned long)max_c);
  printf("latency min=%lu.%02lu us  avg=%lu.%02lu us  max=%lu.%02lu us\r\n",
         (unsigned long)(min_ns / 1000u), (unsigned long)((min_ns % 1000u) / 10u),
         (unsigned long)(avg_ns / 1000u), (unsigned long)((avg_ns % 1000u) / 10u),
         (unsigned long)(max_ns / 1000u), (unsigned long)((max_ns % 1000u) / 10u));
  printf("throughput ~%lu inferences/sec\r\n",
         (unsigned long)(SystemCoreClock / (avg_c ? avg_c : 1u)));
  print_f("checksum: ", sink, "\r\n");
  printf("=== done ===\r\n");
}
