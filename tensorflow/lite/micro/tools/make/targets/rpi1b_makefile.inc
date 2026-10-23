# Port-local BCM2835 target: ARMv6 ARM state, hard-float VFP.
# Default build remains reference CPU. CO_PROCESSOR=vc4 selects the separately
# audited FP32 FullyConnected overlay; hardware runtime must be linked by mpynn.
# No CMSIS-NN, CMSIS-DSP, or Ethos-U is implied by the VC4 selection.
TARGET_TOOLCHAIN_PREFIX := arm-none-eabi-
TARGET_ARCH := armv6
PLATFORM_FLAGS := \
  -DTF_LITE_MCU_DEBUG_LOG \
  -mcpu=arm1176jzf-s -marm -mfpu=vfp -mfloat-abi=hard \
  -funsigned-char -mlittle-endian -fomit-frame-pointer \
  -fno-exceptions -fno-rtti -fno-use-cxa-atexit -nostdlib
CXXFLAGS += $(PLATFORM_FLAGS) -DTF_LITE_STRIP_ERROR_STRINGS
CCFLAGS += $(PLATFORM_FLAGS)
MICROLITE_LIBS :=
