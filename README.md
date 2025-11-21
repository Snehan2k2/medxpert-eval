# MedXpert Evaluation

This repository provides a simple pipeline for evaluating models using **vLLM** as the inference server and a Python evaluation script.

## 🚀 Usage

### 1. Serve the Model with vLLM

Start the vLLM server using your model (replace `<model-name-or-path>`):

```bash
vllm serve <model-name-or-path> \
  --port 1111 \
  --dtype auto \
  --tensor-parallel-size 1

