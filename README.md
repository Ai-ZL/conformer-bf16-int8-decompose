# Description
In order to make the model conform to a specific architecture for the FPGA implementation.

The original model is [STT En Conformer-CTC Small](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_en_conformer_ctc_small_ls) from NVIDIA NeMo.

# Step
1. Optimize the model
2. Quantize the model to compute the nonlinear function with fp32 data and the rest with int8.
3. Change fp32 to bf16
4. decompose the nonlinear function

# Download model and tokens
```python
import nemo.collections.asr as nemo_asr
asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="stt_en_conformer_ctc_small_ls")
asr_model.export('model.onnx')

with open('tokens.txt', 'w') as f:
  for i, s in enumerate(m.decoder.vocabulary):
    f.write(f"{s} {i}\n")
  f.write(f"<blk> {i+1}\n")
```

# File organization
```python
├── Readme.md                          # help
├── test_inference.py                  # test model inference
├── test_eval.py                       # evaluate the model
├── optimize_model.py                  # optimize the model
├── quantize_model.py                  # quantize the model
├── design
├── model
├── data
```
