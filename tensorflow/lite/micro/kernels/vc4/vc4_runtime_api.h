/* Copyright 2026 The TFLM Authors.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Minimal generic C ABI consumed by the VC4 FullyConnected overlay. The
 * platform runtime supplies strong definitions; if it is not linked, weak
 * references are absent and TFLM uses the reference CPU kernel.
 */
#ifndef TENSORFLOW_LITE_MICRO_KERNELS_VC4_RUNTIME_API_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_VC4_RUNTIME_API_H_
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct vc4_buffer {
  void *cpu;
  uint32_t bus;
  uint32_t handle;
  uint32_t bytes;
} vc4_buffer_t;

enum {
  kVc4StatusOk = 0,
  kVc4StatusUnavailable = -1,
  kVc4StatusInvalidArgument = -2,
  kVc4StatusInvalidShape = -3,
  kVc4StatusNoMemory = -4,
  kVc4StatusTimeout = -5,
  kVc4StatusIoError = -6,
  kVc4StatusNotInitialized = -7,
  kVc4StatusUnsupported = -8,
  kVc4StatusError = -9,
};

int vc4_is_available(void);
int vc4_alloc(uint32_t bytes, uint32_t alignment, uint32_t flags,
              vc4_buffer_t *buffer);
int vc4_copy_to_buffer(vc4_buffer_t *buffer, const void *source,
                       uint32_t bytes);
/* Optional shape-aware upload. An older runtime can omit it and keep raw,
 * row-major vc4_copy_to_buffer semantics. */
int vc4_upload_weights_fp32(vc4_buffer_t *buffer, const float *source,
                            uint32_t output_depth, uint32_t input_depth);
int vc4_free(vc4_buffer_t *buffer);
int vc4_should_offload_fp32(uint32_t output_depth, uint32_t input_depth);
int vc4_matvec_fp32(const float *weights_cpu, uint32_t weights_bus,
                    const float *input, uint32_t output_depth,
                    uint32_t input_depth, uint32_t worker_count,
                    float *output);
/* Release GPU staging when a TFLM interpreter frees the matching FC op. */
int vc4_release_weights(const float *weights_cpu, uint32_t output_depth,
                        uint32_t input_depth);
#ifdef __cplusplus
}
#endif
#endif
