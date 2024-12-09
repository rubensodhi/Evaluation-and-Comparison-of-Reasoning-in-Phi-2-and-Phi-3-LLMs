
# **Pruning for Performance: Evaluating the Impact of Model Reduction on Reasoning Abilities using wanda**

This repository contains the code and resources for the dissertation project titled **"Pruning for Performance: Evaluating the Impact of Model Reduction on the Reasoning Abilities of Phi-2 and Phi-3 Mini using wanda"**. The research focuses on the impact of structured and unstructured pruning on reasoning capabilities in Microsoft Phi-2 and Phi-3 Mini Large Language Models (LLMs).

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
The models were evaluated using the following metrics:
- **Perplexity:** Measures model confidence, with lower values indicating higher confidence.
- **Accuracy:** Proportion of correct predictions made by the model.
- **F1 Score:** Harmonic mean of precision and recall, providing a balanced measure of performance.

### **Key Results**

#### **1. Pruning Time and Resource Usage**
![image](https://github.com/user-attachments/assets/c073f29a-7bbb-46b2-ac77-9eb1094a34b0)
- The Phi-2 model required **4 minutes** and **9.7 GB** of GPU RAM for each pruning operation.
- The Phi-3 Mini model required significantly more time (**17 minutes**) and GPU memory (**19.7 GB**).



#### **2. Perplexity Analysis**
![image](https://github.com/user-attachments/assets/ea02a16b-8642-4265-9eb9-95b796f19018)

- **Baseline Perplexity:** 
  - Phi-2: **9.71** (more confident)
  - Phi-3 Mini: **92.27**
- **Effect of Sparsity on Perplexity:**
  - At 20% sparsity, Phi-2 perplexity increased slightly to **10.08**, while Phi-3 Mini rose to **99.31**.
  - At 80% sparsity, perplexity skyrocketed for both models, indicating severe degradation (Phi-2: **130250**, Phi-3 Mini: **367495**).
- **Structured Pruning (2:4):**
  - Phi-2: **29.48**
  - Phi-3 Mini: **160.13**

#### **3. Zero-shot Evaluations**

**Commonsense Reasoning (CommonsenseQA, CosmosQA):**

![image](https://github.com/user-attachments/assets/e3acb6bd-aa01-4771-a9b8-f1067d88e577)

- Phi-2 consistently outperformed Phi-3 Mini in CommonsenseQA, maintaining better accuracy and F1 scores across sparsity levels.

![image](https://github.com/user-attachments/assets/7690756c-7086-47d2-8102-29b02866262f)

- CosmosQA showed a slight improvement at lower sparsity levels for both models, but performance degraded significantly at higher sparsity.

**Logical Reasoning (LogiQA, ReClor):**

![image](https://github.com/user-attachments/assets/db4c1361-b51f-426e-a576-3b2fe2fa0458)
![image](https://github.com/user-attachments/assets/c48d3dcc-4144-40a1-a851-a74faa505080)

- Both models experienced slight declines in accuracy and F1 scores as sparsity increased.
- Phi-3 Mini demonstrated slightly better resilience in structured pruning, particularly on ReClor.

**Analogical Reasoning (E-KAR):**

![image](https://github.com/user-attachments/assets/56676180-1697-4fe5-a3a2-c9bd658fe9ba)

- Surprisingly, both models showed improved performance at lower sparsity levels (20%-40%) before degrading significantly at 60%-80% sparsity.
- Phi-3 Mini achieved higher baseline and pruned scores than Phi-2.

#### **4. Structured vs. Unstructured Pruning**

![image](https://github.com/user-attachments/assets/264c3c10-e03c-4720-b0cd-627e94f90811)

- **Structured Pruning (2:4):**
  - Commonsense tasks saw a significant drop in accuracy and F1 scores.
  - Logical tasks showed better resilience in Phi-3 Mini compared to Phi-2.
- **Unstructured Pruning:**
  - Phi-2 maintained lower perplexity and performed better at higher sparsity levels in Commonsense tasks.
  - Phi-3 Mini was more robust in Analogical reasoning tasks.

### **Visual Summary**
Below is an example of the key trends observed during evaluations:
- Accuracy and F1 scores gradually decline as sparsity increases, with Commonsense tasks being the most impacted.
- Analogical reasoning showed temporary improvement at mid-level sparsity ratios.

For more detailed figures, refer to the `Results - Visualisations/` directory.
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
