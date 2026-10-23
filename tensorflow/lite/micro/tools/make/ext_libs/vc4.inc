# Copyright 2026 The TensorFlow Authors.
# Licensed under the Apache License, Version 2.0.
#
# VideoCore IV co-processor kernel selection. The specialized FullyConnected
# implementation uses weak platform-runtime symbols and falls back to the
# upstream reference kernel when no VC4 runtime is linked/available. This file
# deliberately does not fetch external code or claim to enable hardware.

# The VC4 kernel sources are selected by the generic CO_PROCESSOR specialization
# mechanism in Makefile, from tensorflow/lite/micro/kernels/vc4/.
VC4_RUNTIME ?=
ifneq ($(VC4_RUNTIME),)
  MICROLITE_CC_SRCS += $(VC4_RUNTIME)
endif
