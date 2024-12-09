
# **Pruning for Performance: Evaluating the Impact of Model Reduction on Reasoning Abilities**

This repository contains the code and resources for the dissertation project titled **"Pruning for Performance: Evaluating the Impact of Model Reduction on the Reasoning Abilities of Phi-2 and Phi-3 Mini"**. The research focuses on the impact of structured and unstructured pruning on reasoning capabilities in Microsoft Phi-2 and Phi-3 Mini Large Language Models (LLMs).

---

## **Overview**

This project evaluates the effects of model pruning using the WaNDa method across various sparsity levels. Key highlights include:
- Analysis of commonsense, logical, and analogical reasoning tasks.
- Structured (2:4) and unstructured pruning strategies.
- Zero-shot evaluations to measure accuracy, F1 scores, and perplexity.

---

## **Features**

- **Pruning:** Efficiently applies structured and unstructured pruning using the WaNDa method.
- **Reasoning Evaluations:** Benchmarks the pruned and baseline models on CommonsenseQA, CosmosQA, LogiQA, ReClor, and E-KAR datasets.
- **Customizable:** Supports modifications for different sparsity levels, models, and evaluation datasets.

---

## **Technologies Used**

- **Programming Language:** Python
- **Libraries/Frameworks:**
  - PyTorch
  - Hugging Face Transformers
  - WaNDa Pruning Library
  - scikit-learn
  - wandb
  - Accelerate
- **Platform:** Google Colab Pro (L4 GPU recommended for efficient pruning)

---

## **Getting Started**

### **Prerequisites**
- Python 3.8 or higher
- An active Google Colab Pro account (or equivalent GPU setup)

### **Setup Instructions**

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. **Create and Activate a Conda Environment**
   ```bash
   conda create -n pruning_env python=3.8 -y
   conda activate pruning_env
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download WaNDa Library**
   ```bash
   git clone https://github.com/example/wanda.git
   cd wanda
   ```

5. **Modify the WaNDa Code for Calibration**
   - In `wanda/prune.py`, update the `get_loaders` function to use `wikitext2` for calibration.

---

## **Running the Code**

### **Pruning the Models**
Run the following command to prune a model:
```bash
python main.py --model microsoft/phi-2 --sparsity_type unstructured --sparsity_ratio 0.2 --prune_method wanda --save /path/to/results/ --save_model /path/to/pruned_model/
```
- Modify the `--sparsity_type` and `--sparsity_ratio` as needed.

### **Evaluating the Models**
After pruning, use the `test.py` script to perform zero-shot evaluations:
```bash
python test.py --dataset /path/to/dataset --model /path/to/pruned_model/
```
- Ensure the dataset and model paths are correctly specified.

---

## **Results**

### **Performance Metrics**
- **Perplexity:** Evaluates model confidence on WikiText-2.
- **Accuracy and F1 Scores:** Measures reasoning task performance on five datasets.

### **Key Observations**
- Phi-2 exhibited lower perplexity and higher confidence compared to Phi-3 Mini.
- Structured pruning (2:4) caused greater performance degradation than unstructured pruning.

---

## **Dataset Details**

| Dataset       | Reasoning Ty          |
|---------------|-----------------------|
| CommonsenseQA | Commonsense Reasoning |
| CosmosQA      | Commonsense Reasoning |
| LogiQA        | Logical Reasoning     |
| ReClor        | Logical Reasoning     |
| E-KAR         | Analogical Reasoning  |

---

## **License**

This project is licensed under the MIT License. See the `LICENSE` file for more details.
