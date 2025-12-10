# %% [markdown]
# # Bert baseline for POLAR

# %% [markdown]
# ## Introduction
# 
# In this part of the starter notebook, we will take you through the process of all three Subtasks.

# %% [markdown]
# ## Subtask 1 - Polarization detection
# 
# This is a binary classification to determine whether a post contains polarized content (Polarized or Not Polarized).

# %%
#!unzip dev_phase.zip

# %% [markdown]
# ## Imports

# %%
#!pip uninstall -y transformers
#!pip install transformers==4.57.3

# %%
!pip install pandas numpy scikit-learn transformers datasets torch accelerate
!pip install --upgrade transformers

# %%
#!pip install pandas
#!pip install scikit-learn
#!pip install transformers
#!pip install datasets
#!pip install accelerate
!pip install wandb
!pip install -q hf_transfer


# %%
import pandas as pd
import random
import numpy as np

from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.metrics import precision_recall_fscore_support

import numpy as np
from sklearn.model_selection import KFold


import torch

from sklearn.metrics import f1_score

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from torch.utils.data import Dataset

from sklearn.utils.class_weight import compute_class_weight


# %%
import wandb

# Disable wandb logging for this script
wandb.init(mode="disabled")

# %%
# Suppress specific warnings

import warnings
from transformers.utils import logging

# Hide ONLY the "not initialized" weight warning
warnings.filterwarnings(
    "ignore",
    message="Some weights of.*were not initialized",
    category=UserWarning
)

# Hide ONLY the tokenizer deprecation warning
warnings.filterwarnings(
    "ignore",
    message="`tokenizer` is deprecated and will be removed in version 5.0.0",
    category=FutureWarning
)

# Also suppress the internal HF logger message for uninitialized weights
logging.set_verbosity_error()

# %% [markdown]
# ## Data Import
# 
# The training data consists of a short text and binary labels
# 
# The data is structured as a CSV file with the following fields:
# - id: a unique identifier for the sample
# - text: a sentence or short text
# - polarization:  1 text is polarized, 0 text is not polarized
# 
# The data is in all three subtask folders the same but only containing the labels for the specific task.

# %%
# Load the training and validation data for subtask 1

#train = pd.read_csv('subtask1/train/eng.csv')
#val = pd.read_csv('subtask1/train/eng.csv')
#train.head()

df = pd.read_csv('subtask1/train/eng.csv')   # only use TRAIN
df.head()

# %%
# Class Weights Fix Imbalance
print(df['polarization'].value_counts(normalize=True))

# Compute weights for classes 0 and 1
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=df['polarization']
)

class_weights = torch.tensor(class_weights, dtype=torch.float)
print("Class weights:", class_weights)




# %%
# Cross-Validation

kf = KFold(n_splits=5, shuffle=True, random_state=42)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# %%
# Helper Functions: Oversample/Undersample Fix Imbalance
from sklearn.utils import resample

def oversample_df(df):
    df_majority = df[df['polarization'] == 0]
    df_minority = df[df['polarization'] == 1]

    # oversample minority to match majority count
    df_minority_upsampled = resample(
        df_minority,
        replace=True, 
        n_samples=len(df_majority),
        random_state=42
    )

    df_upsampled = pd.concat([df_majority, df_minority_upsampled]).sample(frac=1, random_state=42)
    return df_upsampled

def undersample_df(df):
    df_majority = df[df['polarization'] == 0]
    df_minority = df[df['polarization'] == 1]

    df_majority_down = resample(
        df_majority,
        replace=False,
        n_samples=len(df_minority),
        random_state=42
    )

    # Combine balanced datasets
    df_downsampled = pd.concat([df_majority_down, df_minority]).sample(frac=1, random_state=42)
    return df_downsampled

def extreme_unbalance_df(df, minority_frac=0.01):
    """
    Create an extremely unbalanced dataset by keeping all majority-class samples
    and only a small fraction of minority-class samples.
    
    minority_frac: float, fraction of minority examples to keep (e.g., 0.01 = 1%)
    """
    df_majority = df[df['polarization'] == 0]
    df_minority = df[df['polarization'] == 1]

    # Keep only a tiny portion of minority class
    df_minority_small = df_minority.sample(
        frac=minority_frac, 
        replace=False, 
        random_state=42
    )

    # Combine and shuffle
    df_extreme = pd.concat([df_majority, df_minority_small]).sample(
        frac=1, random_state=42
    )

    return df_extreme

# %%
# Check Distributions (redefined in folds, just to check here)
train_df = undersample_df(df.iloc[train_idx])
val_df   = df.iloc[val_idx]   # NEVER touch validation

train_df = extreme_unbalance_df(df.iloc[train_idx], minority_frac=0.1)
#val_df   = df.iloc[val_idx]   # NEVER modify validation
val_df = extreme_unbalance_df(df.iloc[val_idx], minority_frac=0.1)

print("Train distribution:", train_df['polarization'].value_counts().to_dict())
print("Val distribution:", val_df['polarization'].value_counts().to_dict())

# %%
df['polarization'].value_counts()

# %%
preds = trainer.predict(val_dataset)
print(np.unique(preds.predictions.argmax(axis=1), return_counts=True))

# %% [markdown]
# # Dataset
# -  Create a pytorch class for handling data
# -  Wrapping the raw texts and labels into a format that Huggingface’s Trainer can use for training and evaluation

# %%
# Fix the dataset class by inheriting from torch.utils.data.Dataset
class PolarizationDataset(torch.utils.data.Dataset):
  def __init__(self,texts,labels,tokenizer,max_length =128):
    self.texts=texts
    self.labels=labels
    self.tokenizer= tokenizer
    self.max_length = max_length # Store max_length

  def __len__(self):
    return len(self.texts)

  def __getitem__(self,idx):
    text=self.texts[idx]
    label=self.labels[idx]
    encoding=self.tokenizer(text,truncation=True,padding="max_length",max_length=self.max_length,return_tensors='pt')

    # Ensure consistent tensor conversion for all items
    item = {key: encoding[key].squeeze() for key in encoding.keys()}
    item['labels'] = torch.tensor(label, dtype=torch.long)
    return item

# %% [markdown]
# Now, we'll tokenize the text data and create the datasets using `bert-base-uncased` as the tokenizer.

# %%
# Define Metrics

def compute_metrics(p):
        preds = p.predictions.argmax(axis=1)
        labels = p.label_ids
    
        # class-wise metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, labels=[0,1], average=None, zero_division=0
        )
    
        return {
            "f1_macro": f1.mean(),
            "f1_class_0": f1[0],
            "f1_class_1": f1[1],
            "recall_class_1": recall[1],  # often the most important
            "precision_class_1": precision[1]
        }

# %%
# 1. Unbalanced 5fold Cross-Validation

from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import numpy as np

fold_results = []
fold_f1_macro = []
fold_f1_c0 = []
fold_f1_c1 = []

for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
    print(f"\n===== FOLD {fold+1} =====")

    train_df = df.iloc[train_idx]
    val_df   = df.iloc[val_idx]

    train_dataset = PolarizationDataset(
        train_df['text'].tolist(),
        train_df['polarization'].tolist(),
        tokenizer
    )

    val_dataset = PolarizationDataset(
        val_df['text'].tolist(),
        val_df['polarization'].tolist(),
        tokenizer
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2
    )

    training_args = TrainingArguments(
        output_dir=f'./results_fold_{fold}',
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_steps=100,
        disable_tqdm=False,
        fp16=True,       # ⚡ Use mixed precision (A100 optimized)
        tf32=True,       # ⚡ Even faster matmul on A100
        load_best_model_at_end=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    results = trainer.evaluate()
    fold_results.append(results['eval_f1_macro'])

    fold_f1_macro.append(results['eval_f1_macro'])
    fold_f1_c0.append(results['eval_f1_class_0'])
    fold_f1_c1.append(results['eval_f1_class_1'])
    
    print(f"Fold {fold+1} F1 macro: {results['eval_f1_macro']:.4f}")
    print(f"Fold {fold+1} F1 class 0: {results['eval_f1_class_0']:.4f}")
    print(f"Fold {fold+1} F1 class 1: {results['eval_f1_class_1']:.4f}")

# %%
# 2. Weighted Loss Balanced 5-Fold Cross Validation
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import torch
from torch.nn import CrossEntropyLoss

fold_f1_macro = []
fold_f1_c0 = []
fold_f1_c1 = []

# Custom Trainer that applies class weights
class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs["logits"]
    
        loss_fct = CrossEntropyLoss(weight=self.class_weights.to(model.device))
        loss = loss_fct(logits, labels)
    
        return (loss, outputs) if return_outputs else loss


for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
    print(f"\n===== FOLD {fold+1} =====")

    train_df = df.iloc[train_idx]
    val_df   = df.iloc[val_idx]

    # Compute class weights ONLY from training data
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0,1]),
        y=train_df["polarization"]
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    print("Class weights:", class_weights)

    # Build datasets
    train_dataset = PolarizationDataset(
        train_df["text"].tolist(),
        train_df["polarization"].tolist(),
        tokenizer
    )
    val_dataset = PolarizationDataset(
        val_df["text"].tolist(),
        val_df["polarization"].tolist(),
        tokenizer
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )

    training_args = TrainingArguments(
        output_dir=f"./weighted_fold_{fold}",
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        weight_decay=0.01,
        fp16=True,
        tf32=True,
        logging_steps=100
    )

    def compute_metrics(p):
        preds = p.predictions.argmax(axis=1)
        labels = p.label_ids

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, labels=[0,1], average=None, zero_division=0
        )

        return {
            "f1_macro": f1.mean(),
            "f1_class_0": f1[0],
            "f1_class_1": f1[1],
            "recall_class_1": recall[1],
            "precision_class_1": precision[1]
        }

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    results = trainer.evaluate()

    fold_f1_macro.append(results["eval_f1_macro"])
    fold_f1_c0.append(results["eval_f1_class_0"])
    fold_f1_c1.append(results["eval_f1_class_1"])

    print(f"Fold {fold+1} F1 macro : {results['eval_f1_macro']:.4f}")
    print(f"Fold {fold+1} F1 class0: {results['eval_f1_class_0']:.4f}")
    print(f"Fold {fold+1} F1 class1: {results['eval_f1_class_1']:.4f}")

# %%
# 3. Oversampling 5fold Cross-Validation
# === Oversampling Baseline Experiment ===

from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils import resample
import numpy as np

fold_results = []
fold_f1_macro = []
fold_f1_c0 = []
fold_f1_c1 = []

def oversample_df(df):
    df_major = df[df['polarization']==0]
    df_minor = df[df['polarization']==1]

    df_minor_up = resample(
        df_minor,
        replace=True,
        n_samples=len(df_major),
        random_state=42
    )

    return pd.concat([df_major, df_minor_up]).sample(frac=1, random_state=42)


def compute_metrics(p):
    preds = p.predictions.argmax(axis=1)
    labels = p.label_ids
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, labels=[0,1], average=None, zero_division=0
    )
    
    return {
        "f1_macro": f1.mean(),
        "f1_class_0": f1[0],
        "f1_class_1": f1[1],
        "recall_class_1": recall[1],
        "precision_class_1": precision[1]
    }


for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
    print(f"\n===== OVERSAMPLING FOLD {fold+1} =====")

    # oversample ONLY training split
    train_df = oversample_df(df.iloc[train_idx])
    val_df   = df.iloc[val_idx]

    train_dataset = PolarizationDataset(
        train_df['text'].tolist(),
        train_df['polarization'].tolist(),
        tokenizer
    )

    val_dataset = PolarizationDataset(
        val_df['text'].tolist(),
        val_df['polarization'].tolist(),
        tokenizer
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2
    )

    training_args = TrainingArguments(
        output_dir=f'./oversample_fold_{fold}',
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        fp16=True,
        tf32=True,
        logging_steps=100
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    results = trainer.evaluate()

    fold_f1_macro.append(results['eval_f1_macro'])
    fold_f1_c0.append(results['eval_f1_class_0'])
    fold_f1_c1.append(results['eval_f1_class_1'])

    print(f"Fold {fold+1} F1 macro: {results['eval_f1_macro']:.4f}")
    print(f"Fold {fold+1} F1 class 0: {results['eval_f1_class_0']:.4f}")
    print(f"Fold {fold+1} F1 class 1: {results['eval_f1_class_1']:.4f}")

print("\n===== FINAL OVERSAMPLING RESULTS =====")
print("Macro F1 per fold:", fold_f1_macro)
print("Mean Macro F1:", np.mean(fold_f1_macro))
print("Std Macro F1:", np.std(fold_f1_macro))

# %%
# 4. Undersampling 5fold Cross-Validation

# === Undersampling Baseline Experiment ===

from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils import resample
import numpy as np

fold_results = []
fold_f1_macro = []
fold_f1_c0 = []
fold_f1_c1 = []


def compute_metrics(p):
    preds = p.predictions.argmax(axis=1)
    labels = p.label_ids
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, labels=[0, 1], average=None, zero_division=0
    )
    
    return {
        "f1_macro": f1.mean(),
        "f1_class_0": f1[0],
        "f1_class_1": f1[1],
        "recall_class_1": recall[1],
        "precision_class_1": precision[1]
    }


for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
    print(f"\n===== UNDERSAMPLING FOLD {fold+1} =====")

    # undersample ONLY the training data
    train_df = undersample_df(df.iloc[train_idx])
    val_df   = df.iloc[val_idx]  # NEVER modify validation set

    train_dataset = PolarizationDataset(
        train_df['text'].tolist(),
        train_df['polarization'].tolist(),
        tokenizer
    )

    val_dataset = PolarizationDataset(
        val_df['text'].tolist(),
        val_df['polarization'].tolist(),
        tokenizer
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2
    )

    training_args = TrainingArguments(
        output_dir=f'./undersample_fold_{fold}',
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        fp16=True,
        tf32=True,
        logging_steps=100
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    results = trainer.evaluate()

    fold_f1_macro.append(results['eval_f1_macro'])
    fold_f1_c0.append(results['eval_f1_class_0'])
    fold_f1_c1.append(results['eval_f1_class_1'])

    print(f"Fold {fold+1} F1 macro: {results['eval_f1_macro']:.4f}")
    print(f"Fold {fold+1} F1 class 0: {results['eval_f1_class_0']:.4f}")
    print(f"Fold {fold+1} F1 class 1: {results['eval_f1_class_1']:.4f}")

print("\n===== FINAL UNDERSAMPLING RESULTS =====")
print("Macro F1 per fold:", fold_f1_macro)
print("Mean Macro F1:", np.mean(fold_f1_macro))
print("Std Macro F1:", np.std(fold_f1_macro))


# %%
# 5. Extreme Unbalance 10% 5-Fold Cross Validation

fold_f1_macro = []
fold_f1_c0 = []
fold_f1_c1 = []

for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
    print(f"\n===== EXTREME 4% FOLD {fold+1} =====")

    # EXTREME IMBALANCE only in TRAINING data
    train_df = extreme_unbalance_df(df.iloc[train_idx], minority_frac=0.4)
    val_df   = df.iloc[val_idx]   

    print("DEBUG FOLD", fold)
    print("Train distribution:", train_df['polarization'].value_counts().to_dict())
    print("Val distribution:", val_df['polarization'].value_counts().to_dict())

    train_dataset = PolarizationDataset(
        train_df['text'].tolist(),
        train_df['polarization'].tolist(),
        tokenizer
    )

    val_dataset = PolarizationDataset(
        val_df['text'].tolist(),
        val_df['polarization'].tolist(),
        tokenizer
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2
    )

    training_args = TrainingArguments(
        output_dir=f'./extreme10pct_fold_{fold}',
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        fp16=True,
        tf32=True,
        logging_steps=100
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    results = trainer.evaluate()

    fold_f1_macro.append(results['eval_f1_macro'])
    fold_f1_c0.append(results['eval_f1_class_0'])
    fold_f1_c1.append(results['eval_f1_class_1'])

print("\n===== FINAL Extreme 40% RESULTS =====")
print("Macro F1 per fold:", fold_f1_macro)
print("Mean Macro F1:", np.mean(fold_f1_macro))
print("Std Macro F1:", np.std(fold_f1_macro))

# %%
# 6. Extreme Unbalance 1% 5-Fold Cross Validation

from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils import resample
import numpy as np

fold_results = []
fold_f1_macro = []
fold_f1_c0 = []
fold_f1_c1 = []


def compute_metrics(p):
    preds = p.predictions.argmax(axis=1)
    labels = p.label_ids
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, labels=[0, 1], average=None, zero_division=0
    )
    
    return {
        "f1_macro": f1.mean(),
        "f1_class_0": f1[0],
        "f1_class_1": f1[1],
        "recall_class_1": recall[1],
        "precision_class_1": precision[1]
    }


for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
    print(f"\n===== EXTREME FOLD {fold+1} =====")

    # undersample ONLY the training data

    train_df = extreme_unbalance_df(df.iloc[train_idx], minority_frac=0.01)
    val_df   = df.iloc[val_idx]


    print("DEBUG FOLD", fold)
    print("Train distribution:", train_df['polarization'].value_counts().to_dict())
    print("Val distribution:", val_df['polarization'].value_counts().to_dict())

    train_dataset = PolarizationDataset(
        train_df['text'].tolist(),
        train_df['polarization'].tolist(),
        tokenizer
    )

    val_dataset = PolarizationDataset(
        val_df['text'].tolist(),
        val_df['polarization'].tolist(),
        tokenizer
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2
    )

    training_args = TrainingArguments(
        output_dir=f'./undersample_fold_{fold}',
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        fp16=True,
        tf32=True,
        logging_steps=100
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    results = trainer.evaluate()

    fold_f1_macro.append(results['eval_f1_macro'])
    fold_f1_c0.append(results['eval_f1_class_0'])
    fold_f1_c1.append(results['eval_f1_class_1'])

    print(f"Fold {fold+1} F1 macro: {results['eval_f1_macro']:.4f}")
    print(f"Fold {fold+1} F1 class 0: {results['eval_f1_class_0']:.4f}")
    print(f"Fold {fold+1} F1 class 1: {results['eval_f1_class_1']:.4f}")

print("\n===== FINAL Extreme 1% Debalancing RESULTS =====")
print("Macro F1 per fold:", fold_f1_macro)
print("Mean Macro F1:", np.mean(fold_f1_macro))
print("Std Macro F1:", np.std(fold_f1_macro))



# %%
print("\n===== FINAL 5-FOLD RESULTS =====")
print("Macro F1 per fold:", fold_f1_macro)
print("Mean Macro F1:", np.mean(fold_f1_macro))
print("Std Macro F1:", np.std(fold_f1_macro))

print("\nClass 0 F1 per fold:", fold_f1_c0)
print("Mean F1 class 0:", np.mean(fold_f1_c0))

print("\nClass 1 F1 per fold:", fold_f1_c1)
print("Mean F1 class 1:", np.mean(fold_f1_c1))

# %%


# %%
## Random Parameter Optimization Search 

# %%
# Set Parameters

def sample_loguniform(low, high):
    return np.exp(np.random.uniform(np.log(low), np.log(high)))

def sample_param_space():
    return {
        "learning_rate": sample_loguniform(1e-6, 4e-5),
        "num_train_epochs": random.choice([1,2,3]),
        "warmup_ratio": random.uniform(0.0, 0.15),
        "weight_decay": random.choice([0.0, 0.01, 0.1]),
        "batch_size": random.choice([8,16]),
        "max_length": random.choice([128,256]),
        "max_grad_norm": random.choice([0.5,1.0,2.0]),
    }


def sample_phase1_params():
    return {
        "learning_rate": float(10 ** np.random.uniform(-5.5, -4)),  
        "num_train_epochs": int(np.random.choice([2, 3, 4])),
        "warmup_ratio": float(np.random.uniform(0.0, 0.2)),
        "weight_decay": float(np.random.uniform(0.0, 0.1)),
    }


# %%
# Run CV with Weighted Loss for Hyperparameter Tuning

from transformers import TrainingArguments, AutoModelForSequenceClassification, Trainer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import torch

import os

import csv

def log_results(params, score, filename="phase1_results.csv"):
    header = ["learning_rate", "num_train_epochs", "warmup_ratio", "macro_f1"]

    # Append mode
    file_exists = os.path.isfile(filename)

    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)

        # Write header only once
        if not file_exists:
            writer.writerow(header)

        writer.writerow([
            params["learning_rate"],
            params["num_train_epochs"],
            params["warmup_ratio"],
            score
        ])

# ==========================
# Weighted Trainer for Balanced Training
# ==========================
class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs["logits"]

        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(model.device))
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


# ======================================================
# Cross-validation FUNCTION for hyperparameter tuning
# ======================================================
def run_cv_with_params(df, tokenizer, kf, params):
    """Run 5-fold CV with weighted loss and return mean macro F1."""

    # ----- Compute class weights ONCE -----
    class_weights_np = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=df['polarization']
    )
    class_weights = torch.tensor(class_weights_np, dtype=torch.float)
    print("Class weights:", class_weights)

    fold_macro_f1 = []

    # ==========================
    # 5-FOLD LOOP
    # ==========================
    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        train_df = df.iloc[train_idx]
        val_df   = df.iloc[val_idx]

        train_dataset = PolarizationDataset(
            train_df['text'].tolist(),
            train_df['polarization'].tolist(),
            tokenizer
        )

        val_dataset = PolarizationDataset(
            val_df['text'].tolist(),
            val_df['polarization'].tolist(),
            tokenizer
        )

        # Fresh model for each fold
        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=2
        )

        # Training arguments using sampled hyperparameters
        training_args = TrainingArguments(
            output_dir=f'./tuning_phase1_fold{fold}',
            eval_strategy="epoch",
            save_strategy="no",
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            fp16=True,
            tf32=True,

            # PARAMETERS WE ARE TUNING
            learning_rate=params["learning_rate"],
            num_train_epochs=params["num_train_epochs"],
            warmup_ratio=params["warmup_ratio"],
        )

        # Metrics
        def compute_metrics(p):
            preds = p.predictions.argmax(axis=1)
            labels = p.label_ids
            _, _, f1, _ = precision_recall_fscore_support(
                labels, preds, labels=[0,1], average=None, zero_division=0
            )
            return {
                "f1_macro": f1.mean(),
                "f1_class_0": f1[0],
                "f1_class_1": f1[1],
            }

        # Weighted trainer
        trainer = WeightedTrainer(
            class_weights=class_weights,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )

        trainer.train()
        results = trainer.evaluate()
        fold_macro_f1.append(results['eval_f1_macro'])

    return np.mean(fold_macro_f1)

# %%
import os
import numpy as np
import random

best_score = -1
best_params = None

for i in range(20):  # Phase 1: 20 cheap trials
    print(f"\n===== TRIAL {i+1} / 20 =====")

    # -----------------------
    # 🔥 Generate a unique seed per trial
    # -----------------------
    seed = int.from_bytes(os.urandom(4), "little")
    np.random.seed(seed)
    random.seed(seed)

    # -----------------------
    # Sample new hyperparameters
    # -----------------------
    params = sample_phase1_params()
    params["seed"] = seed  # attach seed to params
    print("Sampled params:", params)

    # -----------------------
    # Run 5-fold CV with weighted training
    # (pass seed into TrainingArguments IN run_cv_with_params)
    # -----------------------
    score = run_cv_with_params(df, tokenizer, kf, params)
    print(f"Trial {i+1} Macro F1 = {score:.4f}")

    # Log output to CSV
    log_results(params, score)

    # Track best model
    if score > best_score:
        best_score = score
        best_params = params

print("\n===== PHASE 1 COMPLETE =====")
print("Best macro F1:", best_score)
print("Best params:", best_params)

# %%
for i in range(20):
    params = sample_phase2_params()
    set_seed(params["seed"])
    score = run_cv_with_params(df, tokenizer, kf, params)
    log_results(params, score)

# %%


# %%
# ============================================
# PHASE 2 — FULL TUNING SCRIPT (ONE CELL)
# ============================================

import numpy as np
import random
import torch
import os
import csv

from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight

from transformers import (
    TrainingArguments,
    Trainer,
    BertConfig,
    AutoModelForSequenceClassification,
    set_seed
)


# =========================================================
# PARAMETER SAMPLING (Phase 2 narrowed search region)
# =========================================================

def sample_phase2_params():
    # FORCE random entropy
    np.random.seed(None)
    random.seed()

    return {
        "learning_rate": float(np.random.uniform(8e-6, 2.5e-5)),
        "warmup_ratio": float(np.random.uniform(0.0, 0.06)),
        "dropout": float(np.random.uniform(0.1, 0.4)),
        "seed": int(np.random.choice([1,2,3,4,5])),
        "num_train_epochs": 2,
        "weight_decay": 0.01
    }

# =========================================================
# MODEL LOADER WITH CUSTOM DROPOUT
# =========================================================
def load_model_with_dropout(dropout):
    config = BertConfig.from_pretrained(
        "bert-base-uncased",
        num_labels=2,
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout,
    )
    return AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", config=config
    )


# =========================================================
# WEIGHTED TRAINER FOR CLASS IMBALANCE
# =========================================================
class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs["logits"]
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(model.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss
        

# =========================================================
# LOG RESULTS INTO CSV
# =========================================================
def log_results(params, score, filename="phase2_results.csv"):
    header = ["learning_rate", "warmup_ratio", "dropout", "seed", "macro_f1"]

    file_exists = os.path.isfile(filename)
    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow([
            params["learning_rate"],
            params["warmup_ratio"],
            params["dropout"],
            params["seed"],
            score
        ])


# =========================================================
# RUN 5-FOLD CROSS-VALIDATION USING WEIGHTED BERT
# =========================================================
def run_cv_with_params(df, tokenizer, kf, params):

    # ----- compute class weights ONCE -----
    class_weights_np = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0,1]),
        y=df["polarization"]
    )
    class_weights = torch.tensor(class_weights_np, dtype=torch.float)

    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):

        train_df = df.iloc[train_idx]
        val_df   = df.iloc[val_idx]

        train_dataset = PolarizationDataset(
            train_df["text"].tolist(),
            train_df["polarization"].tolist(),
            tokenizer,
        )
        val_dataset = PolarizationDataset(
            val_df["text"].tolist(),
            val_df["polarization"].tolist(),
            tokenizer,
        )

        # Seed for reproducibility
        set_seed(params["seed"])

        # Load model with custom dropout
        model = load_model_with_dropout(params["dropout"])

        # Training args
        training_args = TrainingArguments(
            output_dir=f"./phase2_fold{fold}",
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            fp16=True,
            eval_strategy="epoch",
            save_strategy="no",
            learning_rate=params["learning_rate"],
            warmup_ratio=params["warmup_ratio"],
            weight_decay=params["weight_decay"],
            num_train_epochs=params["num_train_epochs"],
            report_to="none"
        )

        # Metrics
        def compute_metrics(p):
            preds = p.predictions.argmax(axis=1)
            labels = p.label_ids
            _, _, f1, _ = precision_recall_fscore_support(
                labels, preds, labels=[0,1], average=None, zero_division=0
            )
            return {
                "f1_macro": float(f1.mean()),
                "f1_class0": float(f1[0]),
                "f1_class1": float(f1[1])
            }

        # Trainer
        trainer = WeightedTrainer(
            class_weights=class_weights,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )

        trainer.train()
        eval_res = trainer.evaluate()
        fold_scores.append(eval_res["eval_f1_macro"])

    return float(np.mean(fold_scores))


# =========================================================
# MAIN PHASE 2 LOOP — 20 TRIALS
# =========================================================
best_score = -1
best_params = None

for i in range(20):
    print(f"\n=========== TRIAL {i+1} / 20 ===========")
    params = sample_phase2_params()
    print("Params:", params)

    score = run_cv_with_params(df, tokenizer, kf, params)
    print(f"Trial {i+1} Macro-F1 = {score:.4f}")

    log_results(params, score)

    if score > best_score:
        best_score = score
        best_params = params

print("\n===== PHASE 2 COMPLETE =====")
print("Best Macro F1:", best_score)
print("Best params:", best_params)

# %%


# %%


# %%


# %%


# %%


# %%
# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Create datasets
train_dataset = PolarizationDataset(train['text'].tolist(), train['polarization'].tolist(), tokenizer)
val_dataset = PolarizationDataset(val['text'].tolist(), val['polarization'].tolist(), tokenizer)

# %% [markdown]
# Next, we'll load the pre-trained `bert-base-uncased` model for sequence classification. Since this is a binary classification task (Polarized/Not Polarized), we set `num_labels=2`.

# %%
# Load the model
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# %% [markdown]
# Now, we'll define the training arguments and the evaluation metric. We'll use macro F1 score for evaluation.

# %%
# Define metrics function
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {'f1_macro': f1_score(p.label_ids, preds, average='macro')}

# Define training arguments
training_args = TrainingArguments(
        output_dir=f"./",
        num_train_epochs=3,
        learning_rate=2e-5,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=100,
        disable_tqdm=False
    )


# %% [markdown]
# Finally, we'll initialize the `Trainer` and start training.

# %%
# Initialize the Trainer
trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset,            # evaluation dataset
    compute_metrics=compute_metrics,     # the callback that computes metrics of interest
    data_collator=DataCollatorWithPadding(tokenizer) # Data collator for dynamic padding
)

# Train the model
trainer.train()

# Evaluate the model on the validation set
eval_results = trainer.evaluate()
print(f"Macro F1 score on validation set: {eval_results['eval_f1_macro']}")

# %% [markdown]
# # Subtask 2: Polarization Type Classification
# Multi-label classification to identify the target of polarization as one of the following categories: Gender/Sexual, Political, Religious, Racial/Ethnic, or Other.
# For this task we will load the data for subtask 2.

# %%
train = pd.read_csv('subtask2/train/eng.csv')
val = pd.read_csv('subtask2/train/eng.csv')
train.head()

# %%
# Fix the dataset class by inheriting from torch.utils.data.Dataset
class PolarizationDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length # Store max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, truncation=True, padding=False, max_length=self.max_length, return_tensors='pt')

        # Ensure consistent tensor conversion for all items
        item = {key: encoding[key].squeeze() for key in encoding.keys()}
        # CHANGE THIS LINE: Use torch.float instead of torch.long for multi-label classification
        item['labels'] = torch.tensor(label, dtype=torch.float)
        return item


# %%
# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Create train and Test dataset for multilabel
train_dataset = PolarizationDataset(train['text'].tolist(), train[['gender/sexual','political','religious','racial/ethnic','other']].values.tolist(), tokenizer)
val_dataset = PolarizationDataset(val['text'].tolist(), val[['gender/sexual','political','religious','racial/ethnic','other']].values.tolist(), tokenizer)
dev_dataset = PolarizationDataset(val['text'].tolist(), val[['gender/sexual','political','religious','racial/ethnic','other']].values.tolist(), tokenizer)


# %%
# Load the model
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5, problem_type="multi_label_classification") # 5 labels

# %%
# Define metrics function for multi-label classification
def compute_metrics_multilabel(p):
    # Sigmoid the predictions to get probabilities
    probs = torch.sigmoid(torch.from_numpy(p.predictions))
    # Convert probabilities to predicted labels (0 or 1)
    preds = (probs > 0.5).int().numpy()
    # Compute macro F1 score
    return {'f1_macro': f1_score(p.label_ids, preds, average='macro')}

# Define training arguments
training_args = TrainingArguments(
    output_dir=f"./",
    num_train_epochs=3,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=100,
    disable_tqdm=False
)

# %%
# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics_multilabel,  # Use the new metrics function
    data_collator=DataCollatorWithPadding(tokenizer)
)

# Train the model
trainer.train()

# Evaluate the model on the validation set
eval_results = trainer.evaluate()
print(f"Macro F1 score on validation set for Subtask 2: {eval_results['eval_f1_macro']}")

# %% [markdown]
# # Subtask 3: Manifestation Identification
# Multi-label classification to classify how polarization is expressed, with multiple possible labels including Vilification, Extreme Language, Stereotype, Invalidation, Lack of Empathy, and Dehumanization.
# 
# 

# %%
train = pd.read_csv('subtask3/train/eng.csv')
val = pd.read_csv('subtask3/train/eng.csv')

train.head()

# %%
# Fix the dataset class by inheriting from torch.utils.data.Dataset
class PolarizationDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length # Store max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, truncation=True, padding=False, max_length=self.max_length, return_tensors='pt')

        # Ensure consistent tensor conversion for all items
        item = {key: encoding[key].squeeze() for key in encoding.keys()}
        # CHANGE THIS LINE: Use torch.float instead of torch.long for multi-label classification
        item['labels'] = torch.tensor(label, dtype=torch.float)
        return item

# %%
# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Create train and Test dataset for multilabel
train_dataset = PolarizationDataset(train['text'].tolist(), train[['vilification','extreme_language','stereotype','invalidation','lack_of_empathy','dehumanization']].values.tolist(), tokenizer)
val_dataset = PolarizationDataset(val['text'].tolist(), val[['vilification','extreme_language','stereotype','invalidation','lack_of_empathy','dehumanization']].values.tolist(), tokenizer)

# %%
# Load the model
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6, problem_type="multi_label_classification") # use 6 labels

# %%
# Define training arguments
training_args = TrainingArguments(
    output_dir=f"./",
    num_train_epochs=3,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=100,
    disable_tqdm=False
)

# Define metrics function for multi-label classification
def compute_metrics_multilabel(p):
    # Sigmoid the predictions to get probabilities
    probs = torch.sigmoid(torch.from_numpy(p.predictions))
    # Convert probabilities to predicted labels (0 or 1)
    preds = (probs > 0.5).int().numpy()
    # Compute macro F1 score
    return {'f1_macro': f1_score(p.label_ids, preds, average='macro')}

# %%
# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics_multilabel,  # Use the new metrics function
    data_collator=DataCollatorWithPadding(tokenizer)
)

# Train the model
trainer.train()

# Evaluate the model on the validation set
eval_results = trainer.evaluate()
print(f"Macro F1 score on validation set for Subtask 3: {eval_results['eval_f1_macro']}")


