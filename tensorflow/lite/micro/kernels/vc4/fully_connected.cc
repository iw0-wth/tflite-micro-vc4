/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/micro/kernels/fully_connected.h"

#include <cstring>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
#include "tensorflow/lite/kernels/internal/reference/fully_connected.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/kernels/vc4/vc4_runtime_api.h"

#if defined(__GNUC__)
extern "C" int vc4_is_available(void) __attribute__((weak));
extern "C" int vc4_should_offload_fp32(uint32_t, uint32_t)
    __attribute__((weak));
extern "C" int vc4_alloc(uint32_t, uint32_t, uint32_t, vc4_buffer_t*)
    __attribute__((weak));
extern "C" int vc4_copy_to_buffer(vc4_buffer_t*, const void*, uint32_t)
    __attribute__((weak));
extern "C" int vc4_upload_weights_fp32(vc4_buffer_t*, const float*, uint32_t,
                                        uint32_t) __attribute__((weak));
extern "C" int vc4_free(vc4_buffer_t*) __attribute__((weak));
extern "C" int vc4_matvec_fp32(const float*, uint32_t, const float*,
                                uint32_t, uint32_t, uint32_t,
                                float*) __attribute__((weak));
#endif

namespace tflite {
namespace {

struct Vc4FullyConnectedData {
  OpDataFullyConnected reference;
  vc4_buffer_t weights;
  bool weights_uploaded;
};

#if defined(__arm__) && defined(__VFP_FP__) && !defined(__SOFTFP__)
/* ARM1176 VFPv2 has long scalar dependency latency. Four independent
 * accumulators expose instruction-level parallelism to VFP without changing
 * the operator API. The host still uses the upstream reference implementation. */
void FullyConnectedFloatVfp4(const FullyConnectedParams& params,
                              const RuntimeShape& input_shape,
                              const float* input_data,
                              const RuntimeShape& filter_shape,
                              const float* filter_data,
                              const float* bias_data,
                              const RuntimeShape& output_shape,
                              float* output_data) {
  const int input_rank = input_shape.DimensionsCount();
  const int filter_rank = filter_shape.DimensionsCount();
  const int output_rank = output_shape.DimensionsCount();
  const int input_depth = input_shape.Dims(input_rank - 1);
  const int output_depth = filter_shape.Dims(filter_rank - 2);
  const int batches = FlatSizeSkipDim(output_shape, output_rank - 1);
  for (int batch = 0; batch < batches; ++batch) {
    const float* input = input_data + batch * input_depth;
    float* output = output_data + batch * output_depth;
    for (int row = 0; row < output_depth; ++row) {
      const float* weights = filter_data + row * input_depth;
      float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
      int col = 0;
      for (; col + 3 < input_depth; col += 4) {
        sum0 += input[col] * weights[col];
        sum1 += input[col + 1] * weights[col + 1];
        sum2 += input[col + 2] * weights[col + 2];
        sum3 += input[col + 3] * weights[col + 3];
      }
      for (; col < input_depth; ++col) sum0 += input[col] * weights[col];
      float value = (sum0 + sum1) + (sum2 + sum3);
      if (bias_data != nullptr) value += bias_data[row];
      output[row] = ActivationFunctionWithMinMax(
          value, params.float_activation_min, params.float_activation_max);
    }
  }
}
#endif

void Vc4FullyConnectedFree(TfLiteContext* context, void* buffer) {
  (void)context;
  auto* data = static_cast<Vc4FullyConnectedData*>(buffer);
  if (data != nullptr && data->weights_uploaded && vc4_free != nullptr) {
    // On timeout, the port runtime moves the handle to its deferred-free queue
    // and only releases it after the scheduler confirms QPU completion.
    (void)vc4_free(&data->weights);
    data->weights_uploaded = false;
  }
}

void* FullyConnectedInit(TfLiteContext* context, const char* buffer,
                         size_t length) {
  (void)buffer;
  (void)length;
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  auto* data = static_cast<Vc4FullyConnectedData*>(
      context->AllocatePersistentBuffer(context, sizeof(Vc4FullyConnectedData)));
  if (data != nullptr) {
    std::memset(data, 0, sizeof(*data));
  }
  return data;
}

TfLiteStatus FullyConnectedPrepare(TfLiteContext* context, TfLiteNode* node) {
  MicroContext* micro_context = GetMicroContext(context);

  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  auto* vc4_data = static_cast<Vc4FullyConnectedData*>(node->user_data);
  auto* data = &vc4_data->reference;
  const auto params =
      static_cast<const TfLiteFullyConnectedParams*>(node->builtin_data);

  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kFullyConnectedInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* filter = micro_context->AllocateTempInputTensor(
      node, kFullyConnectedWeightsTensor);
  TF_LITE_ENSURE(context, filter != nullptr);
  TfLiteTensor* bias =
      micro_context->AllocateTempInputTensor(node, kFullyConnectedBiasTensor);
  TfLiteTensor* output = micro_context->AllocateTempOutputTensor(
      node, kFullyConnectedOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);

  if ((input->type == kTfLiteFloat32 && filter->type != kTfLiteFloat32) ||
      (input->type == kTfLiteInt8 &&
       (filter->type != kTfLiteInt8 && filter->type != kTfLiteInt4)) ||
      (input->type == kTfLiteInt16 && filter->type != kTfLiteInt8)) {
    MicroPrintf("Input type: %s with filter type : %s not supported.",
                TfLiteTypeGetName(input->type),
                TfLiteTypeGetName(filter->type));
    return kTfLiteError;
  }

  if (filter->type == kTfLiteInt4) {
    int filter_size =
        RuntimeShape(filter->dims->size,
                     reinterpret_cast<const int32_t*>(filter->dims->data))
            .FlatSize();
    context->RequestScratchBufferInArena(context, filter_size,
                                         &data->filter_buffer_index);
  }

  TF_LITE_ENSURE_OK(context, CalculateOpDataFullyConnected(
                                 context, params->activation, input->type,
                                 input, filter, bias, output, data));

  if (input->type == kTfLiteFloat32 && filter->type == kTfLiteFloat32 &&
      filter->dims != nullptr && filter->dims->size >= 2 &&
      vc4_is_available != nullptr && vc4_should_offload_fp32 != nullptr &&
      vc4_alloc != nullptr && vc4_copy_to_buffer != nullptr &&
      vc4_free != nullptr && vc4_is_available()) {
    const int dimensions = filter->dims->size;
    const uint32_t output_depth =
        static_cast<uint32_t>(filter->dims->data[dimensions - 2]);
    const uint32_t input_depth =
        static_cast<uint32_t>(filter->dims->data[dimensions - 1]);
    if (filter->bytes > 0 && filter->bytes <= UINT32_MAX &&
        vc4_should_offload_fp32(output_depth, input_depth)) {
      vc4_buffer_t candidate = {};
      const uint32_t bytes = static_cast<uint32_t>(filter->bytes);
      if (vc4_alloc(bytes, 4096u, 0xCu, &candidate) == 0) {
        const int upload_status = vc4_upload_weights_fp32 != nullptr
            ? vc4_upload_weights_fp32(&candidate, filter->data.f,
                                       output_depth, input_depth)
            : vc4_copy_to_buffer(&candidate, filter->data.f, bytes);
        if (upload_status == 0) {
          vc4_data->weights = candidate;
          vc4_data->weights_uploaded = true;
        } else {
          (void)vc4_free(&candidate);
        }
      }
    }
  }

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(filter);
  if (bias != nullptr) {
    micro_context->DeallocateTempTfLiteTensor(bias);
  }
  micro_context->DeallocateTempTfLiteTensor(output);
  return kTfLiteOk;
}

TfLiteStatus FullyConnectedEval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);
  const auto* params =
      static_cast<const TfLiteFullyConnectedParams*>(node->builtin_data);

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kFullyConnectedInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kFullyConnectedWeightsTensor);
  const TfLiteEvalTensor* bias =
      tflite::micro::GetEvalInput(context, node, kFullyConnectedBiasTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kFullyConnectedOutputTensor);

  TFLITE_DCHECK(node->user_data != nullptr);

  const auto* vc4_data =
      static_cast<const Vc4FullyConnectedData*>(node->user_data);
  const auto& data = vc4_data->reference;

  // Checks in Prepare ensure input, output and filter types are all the same.
  switch (input->type) {
    case kTfLiteFloat32: {
      const RuntimeShape input_shape = tflite::micro::GetTensorShape(input);
      const RuntimeShape filter_shape = tflite::micro::GetTensorShape(filter);
      const RuntimeShape output_shape = tflite::micro::GetTensorShape(output);
      const int input_dims = input_shape.DimensionsCount();
      const int filter_dims = filter_shape.DimensionsCount();
      const int output_dims = output_shape.DimensionsCount();
      const int input_depth = input_dims > 0 ? input_shape.Dims(input_dims - 1) : 0;
      const int output_depth = filter_dims >= 2 ? filter_shape.Dims(filter_dims - 2) : 0;
      const int batches = output_dims > 0
                              ? FlatSizeSkipDim(output_shape, output_dims - 1)
                              : 0;
      const auto* input_data = tflite::micro::GetTensorData<float>(input);
      const auto* filter_data = tflite::micro::GetTensorData<float>(filter);
      const auto* bias_data = tflite::micro::GetOptionalTensorData<float>(bias);
      auto* output_data = tflite::micro::GetTensorData<float>(output);
      bool accelerated = false;

      // The VC4 runtime is optional. If it is absent, unavailable, or the
      // shape is outside the first FP32 matvec kernel's limits, use the exact
      // upstream reference implementation. Runtime failures after dispatch
      // (notably timeout/I/O error) are hard errors: do not reuse buffers while
      // a QPU might still be writing to them.
      if (vc4_is_available != nullptr &&
          vc4_should_offload_fp32 != nullptr &&
          vc4_matvec_fp32 != nullptr && vc4_is_available() &&
          vc4_data->weights_uploaded && input_depth > 0 &&
          output_depth > 0 && batches > 0 &&
          vc4_should_offload_fp32(static_cast<uint32_t>(output_depth),
                                  static_cast<uint32_t>(input_depth))) {
        accelerated = true;
        for (int batch = 0; batch < batches; ++batch) {
          const int status = vc4_matvec_fp32(
              filter_data,
              vc4_data->weights_uploaded ? vc4_data->weights.bus : 0u,
              input_data + static_cast<size_t>(batch) * input_depth,
              static_cast<uint32_t>(output_depth),
              static_cast<uint32_t>(input_depth), 12,
              output_data + static_cast<size_t>(batch) * output_depth);
          if (status == kVc4StatusUnavailable ||
              status == kVc4StatusUnsupported ||
              status == kVc4StatusInvalidShape ||
              status == kVc4StatusNotInitialized) {
            accelerated = false;
            break;
          }
          if (status != kVc4StatusOk) {
            MicroPrintf("VC4 FullyConnected failed (%d).", status);
            return kTfLiteError;
          }
        }
        if (accelerated) {
          const FullyConnectedParams fc_params =
              FullyConnectedParamsFloat(params->activation);
          for (int batch = 0; batch < batches; ++batch) {
            for (int channel = 0; channel < output_depth; ++channel) {
              const int index = batch * output_depth + channel;
              const float bias_value = bias_data ? bias_data[channel] : 0.0f;
              output_data[index] = ActivationFunctionWithMinMax(
                  output_data[index] + bias_value,
                  fc_params.float_activation_min,
                  fc_params.float_activation_max);
            }
          }
        }
      }

      if (!accelerated) {
#if defined(__arm__) && defined(__VFP_FP__) && !defined(__SOFTFP__)
        FullyConnectedFloatVfp4(FullyConnectedParamsFloat(params->activation),
                                input_shape, input_data, filter_shape,
                                filter_data, bias_data, output_shape,
                                output_data);
#else
        tflite::reference_ops::FullyConnected(
            FullyConnectedParamsFloat(params->activation), input_shape,
            input_data, filter_shape, filter_data,
            tflite::micro::GetTensorShape(bias), bias_data, output_shape,
            output_data);
#endif
      }
      break;
    }

    case kTfLiteInt8: {
      switch (filter->type) {
        case kTfLiteInt4: {
          int8_t* unpacked_filter_data = static_cast<int8_t*>(
              context->GetScratchBuffer(context, data.filter_buffer_index));
          tflite::tensor_utils::UnpackDenseInt4IntoInt8(
              tflite::micro::GetTensorData<int8_t>(filter),
              tflite::micro::GetTensorShape(filter).FlatSize(),
              unpacked_filter_data);
          tflite::reference_integer_ops::FullyConnected(
              FullyConnectedParamsQuantized(data),
              tflite::micro::GetTensorShape(input),
              tflite::micro::GetTensorData<int8_t>(input),
              tflite::micro::GetTensorShape(filter), unpacked_filter_data,
              tflite::micro::GetTensorShape(bias),
              tflite::micro::GetOptionalTensorData<int32_t>(bias),
              tflite::micro::GetTensorShape(output),
              tflite::micro::GetTensorData<int8_t>(output));
          break;
        }
        case kTfLiteInt8: {
          tflite::reference_integer_ops::FullyConnected(
              FullyConnectedParamsQuantized(data),
              tflite::micro::GetTensorShape(input),
              tflite::micro::GetTensorData<int8_t>(input),
              tflite::micro::GetTensorShape(filter),
              tflite::micro::GetTensorData<int8_t>(filter),
              tflite::micro::GetTensorShape(bias),
              tflite::micro::GetOptionalTensorData<int32_t>(bias),
              tflite::micro::GetTensorShape(output),
              tflite::micro::GetTensorData<int8_t>(output));
          break;
        }
        default: {
          MicroPrintf("Filter type %s (%d) not supported.",
                      TfLiteTypeGetName(filter->type), input->type);
          return kTfLiteError;
        }
      }
      break;
    }

    case kTfLiteInt16: {
      switch (filter->type) {
        case kTfLiteInt8: {
          tflite::reference_integer_ops::FullyConnected(
              FullyConnectedParamsQuantized(data),
              tflite::micro::GetTensorShape(input),
              tflite::micro::GetTensorData<int16_t>(input),
              tflite::micro::GetTensorShape(filter),
              tflite::micro::GetTensorData<int8_t>(filter),
              tflite::micro::GetTensorShape(bias),
              tflite::micro::GetOptionalTensorData<int64_t>(bias),
              tflite::micro::GetTensorShape(output),
              tflite::micro::GetTensorData<int16_t>(output));
          break;
        }
        default: {
          MicroPrintf("Filter type %s (%d) not supported.",
                      TfLiteTypeGetName(filter->type), input->type);
          return kTfLiteError;
        }
      }
      break;
    }

    default: {
      MicroPrintf("Input type %s (%d) not supported.",
                  TfLiteTypeGetName(input->type), input->type);
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_FULLY_CONNECTED() {
  return tflite::micro::RegisterOp(FullyConnectedInit, FullyConnectedPrepare,
                                   FullyConnectedEval, Vc4FullyConnectedFree);
}

TFLMInferenceRegistration RegisterInference_FULLY_CONNECTED() {
  return tflite::micro::RegisterOp(FullyConnectedEval);
}

}  // namespace tflite
