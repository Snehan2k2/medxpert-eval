# MedXpertQA Evaluation

This repository is a refined and extended modification of the **official MedXpertQA evaluation code**.
The original evaluation framework performs **two-step scoring**, but this version simplifies and accelerates the workflow by:

- Performing the evaluation in **a single step**  
- Using an **LLM-as-a-judge** to score responses directly  
- Preserving **all original prompts and task designs** from the official MedXpertQA benchmark  

The goal of this repository is to provide a faster and more reliable evaluation pipeline. The original MedXpertQA code uses a *two-step* process, where the second step forces the model to output the final answer choice. Many smaller models fail at this instruction-following step even when they know the answer, leading to misleading scores. This version replaces that with a single evaluation pass and an LLM-as-a-judge, avoiding instruction-following failures while staying aligned with the benchmark’s intent.

## ⚙️ Usage

### 1. Serve the Model with vLLM

Start the vLLM server using your model:

```bash
vllm serve <model-name-or-path> \
  --port 1111 \
  --dtype auto \
  --tensor-parallel-size 1
```

### 2. Run the evaluation script

Run the evaluation script to compute accuracy

```bash
python main.py --num-threads 64 \
  --output-dir <output-dir> 
```

## 🔗 Resources

- **Official MedXpertQA Evaluation Code:**  
  https://github.com/TsinghuaC3I/MedXpertQA
