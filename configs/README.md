# Quantization Configuration Parameters

This README explains the various parameters available for model quantization under different algorithms in the MI-optimize framework. Each quantization method offers a unique set of parameters tailored to optimize performance while considering the specific needs of different model types.

## Overview

The quantization process adjusts the precision of model weights and activations, which can significantly reduce memory usage and computational overhead, thereby speeding up inference without substantially compromising model accuracy. Below are the configurations for several quantization algorithms including RTN, GPTQ, AWQ, SmoothQuant, and ZeroQuant.

### Common Parameters

These parameters are shared across different quantization algorithms:
- `algo`: quantize method of model, choices include `rtn`, `gptq`, `awq`, `smoothquant`, `zeroquant`
- `model_type`: Indicates the type of model to quantize (e.g., `llama`, `chatglm`, `baichuan`, `other_model`).
- `device`: Specifies the computation device, e.g., `cuda:0` for a specific GPU.
- `offload`: Device used for offloading computations to save memory, typically `cpu`.
- `skip_layers`: Specifies the layers to skip during quantization, should match the module name. default: ['lm_head']
- `w_dtype`: Weight data type, choices include `int2`, `int3`, `int4`, `int8`,`float16`.
- `a_dtype`: Activation data type, choices include `int2`, `int3`, `int4`, `int8`,`float16`.
- `num_calibrate`: Number of samples used for calibration. default: 128
- `calibrate_seq_length`: Sequence length for calibration. default: 2048
- `calibrate_name`: Specifies the dataset used for calibration, e.g., 
  - `wikitext2`
  - `c4`
  - `ptb`
  - `cmmlu`
  - `ceval`
  - `QuestionAnswering_advqa`
  - `QuestionAnswering_squad`
  - `QuestionAnswering_newsqa`
  - `QuestionAnswering_searchqa`
  - `SentimentAnalysis_amazon`
  - `SentimentAnalysis_dynasent`
  - `SentimentAnalysis_semeval`
  - `SentimentAnalysis_sst5`
  - `NaturalLanguageInference_mnli`
  - `NaturalLanguageInference_anli`
  - `NaturalLanguageInference_wanli`
  - `NaturalLanguageInference_contractnli`
  - `ToxicDetection_civilcomments`
  - `ToxicDetection_advcivil`
  - `ToxicDetection_implicithate`
  - `ToxicDetection_toxigen`

### Specific Parameters by Algorithm

#### RTN (Round-To-Nearest)

- `w_groupsize`: Specifies the size of groups for quantization. (Options: 32, 64, 128)
- `w_qtype`: Quantization type for weights, e.g., `per_channel`. `per_group`. `per_tensor`
- `a_qtype`: Quantization type for activations, e.g., `per_tensor`. `per_token`
- `block_sequential`: Enables block-sequential quantization.
- `layer_sequential`: Enables layer-sequential quantization.

#### GPTQ (Gradient Preserving Tensor Quantization)

- `blocksize`: Defines the block size for tensor quantization.
- `percdamp`: Damping percentage to maintain gradient flow.
- `actorder`: Boolean to specify if activation order is preserved.
- `block_sequential`: Enables block-sequential quantization.
- `layer_sequential`: Enables layer-sequential quantization.

#### AWQ (Adaptive Weight Quantization)

- `w_groupsize`: Group size specifically for weight quantization.
- `w_qtype`: Weight quantization type, e.g., `per_group`.
- `block_sequential`: Enables block-sequential quantization.
- `layer_sequential`: Enables layer-sequential quantization.

#### SmoothQuant

- `w_groupsize`: Group size for weight quantization.
- `w_qtype`: Weight quantization type, e.g., `per_channel`.
- `a_qtype`: Activation quantization type, e.g., `per_token`.
- `alpha`: Coefficient for smoothing.
- `quant_out`: Boolean to determine if output quantization is applied.

#### ZeroQuant

- `groupsize`: Specifies the group size for both weights and activations.
- `w_has_zero`: Boolean to indicate if zero is included in weight quantization.
- `a_has_zero`: Boolean to indicate if zero is included in activation quantization.
- `w_unsign`: Boolean to specify if weights are unsigned.
- `a_unsign`: Boolean to specify if activations are unsigned.
- `quantization_type`: Type of quantization, e.g., `static`.
- `block_sequential`: Enables block-sequential quantization.
- `layer_sequential`: Enables layer-sequential quantization.

