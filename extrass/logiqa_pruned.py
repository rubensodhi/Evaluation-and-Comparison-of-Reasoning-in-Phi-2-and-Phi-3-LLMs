# -*- coding: utf-8 -*-
"""LogiQA_pruned.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1xQLIprmo39UWhFa0J1ETgPM7qFO_8ZQ7
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Path to the pruned model
pruned_model_path = '/content/wanda/model/smol'  # Adjust to your pruned model path

# Load the pruned model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(pruned_model_path)
pruned_model = AutoModelForCausalLM.from_pretrained(pruned_model_path).to('cuda')

# Ensure the pruned model handles padding correctly
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
pruned_model.config.pad_token_id = tokenizer.pad_token_id

print(f"Pruned model loaded successfully from {pruned_model_path}.")

def load_logiqa_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    dataset = []
    i = 0
    while i < len(lines):
        if lines[i].strip() == '':
            i += 1  # Skip the blank line
            continue

        correct_choice = lines[i].strip()
        context = lines[i + 1].strip()
        question = lines[i + 2].strip()
        choices = [lines[i + 3].strip(), lines[i + 4].strip(), lines[i + 5].strip(), lines[i + 6].strip()]

        dataset.append({
            'context': context,
            'question': question,
            'choices': choices,
            'correct_choice': correct_choice
        })

        i += 7  # Move to the next block

    return dataset

# Load the LogiQA dataset
file_path = '/mnt/data/Eval.txt'  # Update with your actual file path
logiqa_data = load_logiqa_data(file_path)

print(f"Loaded {len(logiqa_data)} examples from LogiQA dataset.")
print(logiqa_data[0])  # Print the first example to verify

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def evaluate_logiqa_pruned(pruned_model, tokenizer, dataset):
    y_true = []
    y_pred = []

    for item in dataset:
        context = item['context']
        question = item['question']
        choices = item['choices']
        correct_answer = item['correct_choice']

        scores = []
        for choice in choices:
            # Prepare the input with the context, question, and choice
            input_text = f"{context} {question} {choice}"
            inputs = tokenizer(input_text, return_tensors="pt").to('cuda')

            # Generate the output logits
            with torch.no_grad():
                outputs = pruned_model(**inputs)

            # Calculate the score (likelihood) for the choice
            logits = outputs.logits[:, :-1, :]
            target_ids = inputs['input_ids'][:, 1:]
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(logits.permute(0, 2, 1), target_ids)
            score = -loss.sum().item()  # Negative log-likelihood as the score
            scores.append(score)

        # Select the choice with the highest score
        predicted_index = torch.argmax(torch.tensor(scores)).item()

        # Map the predicted index back to the choice label
        predicted_choice = ['a', 'b', 'c', 'd'][predicted_index]

        # Append true and predicted labels
        y_true.append(correct_answer)
        y_pred.append(predicted_choice)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', labels=['a', 'b', 'c', 'd'])
    recall = recall_score(y_true, y_pred, average='weighted', labels=['a', 'b', 'c', 'd'])
    f1 = f1_score(y_true, y_pred, average='weighted', labels=['a', 'b', 'c', 'd'])

    return accuracy, precision, recall, f1

# Evaluate the pruned model on the LogiQA dataset
accuracy, precision, recall, f1 = evaluate_logiqa_pruned(pruned_model, tokenizer, logiqa_data)

print(f"LogiQA Evaluation Metrics (Pruned Model):")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1-Score: {f1 * 100:.2f}%")

from google.colab import drive
drive.mount('/content/drive')