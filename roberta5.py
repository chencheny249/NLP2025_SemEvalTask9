#region libraries
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#endregion

#region intro
#read in data
train_raw=pd.read_csv("dev_phase/subtask1/train/train_eng.csv")

# Dataset class
class PolarizationDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {key: encoding[key].squeeze() for key in encoding.keys()}
        item['labels'] = torch.tensor(int(label), dtype=torch.long)
        return item
    
# Metrics function
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {'f1_macro': f1_score(p.label_ids, preds, average='macro')}

#endregion 

#region Roberta Base 5-fold cv
#hyperparams to test
param_grid = {
    "learning_rate": [2e-5, 3e-5],
    "num_train_epochs": [3, 5],
    "per_device_train_batch_size": [32, 64],
}

grid = list(ParameterGrid(param_grid))

#5-fold CV
k = 5
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)


#get rid of nas
train_raw = train_raw.dropna(subset=['text', 'polarization']).reset_index(drop=True)

# split train into train and mini test set
train, test = train_test_split(train_raw, test_size=0.1, random_state=42, stratify=train_raw['polarization'])


# loadRoBERTa tokenizer
tokenizer = AutoTokenizer.from_pretrained('roberta-base')

#empty to store results
all_results = []

#test all combos of hyperparams
for params in grid:
    f1_scores = []

    for train_idx, val_idx in skf.split(train['text'], train['polarization']):
        #split for cv
        train_fold = train.iloc[train_idx].reset_index(drop=True)
        val_fold = train.iloc[val_idx].reset_index(drop=True)

        #make datasets
        train_dataset = PolarizationDataset(train_fold['text'].tolist(),
                                            train_fold['polarization'].tolist(),
                                            tokenizer)
        val_dataset = PolarizationDataset(val_fold['text'].tolist(),
                                          val_fold['polarization'].tolist(),
                                          tokenizer)
        # Re-initialize model for each fold
        model = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

        # Training arguments
        training_args = TrainingArguments(
            output_dir="./roberta_output",
            num_train_epochs=params['num_train_epochs'],
            learning_rate=params['learning_rate'],
            per_device_train_batch_size=params['per_device_train_batch_size'],
            per_device_eval_batch_size=8,
            eval_strategy="epoch",
            save_strategy="no",
            logging_steps=100,
            disable_tqdm=False,
            report_to=[]
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            data_collator=DataCollatorWithPadding(tokenizer)
        )

        trainer.train()
        eval_results = trainer.evaluate()
        f1_scores.append(eval_results['eval_f1_macro'])

    mean_f1 = np.mean(f1_scores)
    print(f"Params: {params}, Mean Macro F1: {mean_f1:.4f}")
    all_results.append((params, mean_f1))

#resutls to a df
results_df=pd.DataFrame(all_results)
results_df.columns=['hps', 'f2_avg']

#get best hps
results_df[results_df.f2_avg==results_df.f2_avg.max()].hps.values

#make mini test set
test_dataset = PolarizationDataset(test['text'].tolist(), test['polarization'].tolist(), tokenizer)

#best model - use on test set

best_model = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

best_args = TrainingArguments(
    output_dir="./best_model",
    learning_rate=3e-5,
    per_device_train_batch_size=32,
    num_train_epochs=3,
    eval_strategy="no",
    save_strategy='no',
    report_to=[]
)

best_trainer = Trainer(
    model=best_model,
    args=best_args,
    train_dataset=train_dataset,
    eval_dataset=None,  # or None if you’ve already validated
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer)
)

best_trainer.train()
best_trainer.save_model("./roberta_best_model")
predictions = best_trainer.predict(test_dataset)

#get predictions
y_pred = np.argmax(predictions.predictions, axis=1)
y_true = predictions.label_ids

#scores
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')  # or 'macro' for both classes equally
recall = recall_score(y_true, y_pred, average='macro')
#f1 = f1_score(y_true, y_pred, average='macro')
f1 = compute_metrics(predictions)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")



cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Non-polarized", "Polarized"]
)
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix on Test Set")
plt.show()
#endregion


#region roberta base chat gpt results
dev=pd.read_csv('dev_eng_with_predictions.csv')
dev_test = PolarizationDataset(dev['text'].tolist(), dev['polarization_pred'].tolist(), tokenizer)
dev_pred = best_trainer.predict(dev_test)
y_pred_dev = np.argmax(dev_pred.predictions, axis=1)
y_gpt = dev_pred.label_ids
accuracy = accuracy_score(y_gpt, y_pred_dev)
precision = precision_score(y_gpt, y_pred_dev, average='macro')  # or 'macro' for both classes equally
recall = recall_score(y_gpt, y_pred_dev, average='macro')
f1 = f1_score(y_gpt, y_pred_dev, average='macro')
cm = confusion_matrix(y_gpt, y_pred_dev)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Non-polarized", "Polarized"]
)
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix on Test Set")
plt.show()

#endregion

#region roberta large 5fold
param_grid = {
    "learning_rate": [1e-5, 2e-5],
    "num_train_epochs": [3, 5],
    "per_device_train_batch_size": [8, 16]
}

grid = list(ParameterGrid(param_grid))

k = 5
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
# Load RoBERTa tokenizer
tokenizer = AutoTokenizer.from_pretrained('roberta-large')
train_raw = train_raw.dropna(subset=['text', 'polarization']).reset_index(drop=True)

#split
train, test = train_test_split(train_raw, test_size=0.1, random_state=42, stratify=train_raw['polarization'])
all_results_large = []

for params in grid:
    f1_scores = []

    for train_idx, val_idx in skf.split(train['text'], train['polarization']):
        # Split the data
        train_fold = train.iloc[train_idx].reset_index(drop=True)
        val_fold = train.iloc[val_idx].reset_index(drop=True)

        # Create datasets
        train_dataset = PolarizationDataset(train_fold['text'].tolist(),
                                            train_fold['polarization'].tolist(),
                                            tokenizer)
        val_dataset = PolarizationDataset(val_fold['text'].tolist(),
                                          val_fold['polarization'].tolist(),
                                          tokenizer)
        # Re-initialize model for each fold
        model = AutoModelForSequenceClassification.from_pretrained('roberta-large', num_labels=2)

        # Training arguments
        training_args = TrainingArguments(
            output_dir="./roberta_output",
            num_train_epochs=params['num_train_epochs'],
            learning_rate=params['learning_rate'],
            per_device_train_batch_size=8,
            per_device_train_batch_size=params['per_device_train_batch_size'],
            per_device_eval_batch_size=8,
            eval_strategy="epoch",
            save_strategy="no",
            logging_steps=100,
            disable_tqdm=False,
            report_to=[]
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            data_collator=DataCollatorWithPadding(tokenizer)
        )

        trainer.train()
        eval_results = trainer.evaluate()
        f1_scores.append(eval_results['eval_f1_macro'])

    mean_f1 = np.mean(f1_scores)
    print(f"Params: {params}, Mean Macro F1: {mean_f1:.4f}")
    all_results_large.append((params, mean_f1))

results_df_roberta_large=pd.DataFrame(all_results_large)
results_df_roberta_large.columns=['hps', 'f2_avg']
results_df_roberta_large.to_csv('results_df_roberta_large.csv')
all_results_large_df=pd.DataFrame(all_results_large)
all_results_large_df.columns=['hps','f2_avg']
all_results_large_df[all_results_large_df.f2_avg==all_results_large_df.f2_avg.max()].hps.values



best_model_rl = AutoModelForSequenceClassification.from_pretrained('roberta-large', num_labels=2)

best_args_rl = TrainingArguments(
    output_dir="./best_model_rl",
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    num_train_epochs=5,
    eval_strategy="no",
    save_strategy='no',
    report_to=[]
)

best_trainer_rl = Trainer(
    model=best_model_rl,
    args=best_args_rl,
    train_dataset=train_dataset,
    eval_dataset=None,  # or None if you’ve already validated
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer)
)

best_trainer_rl.train()
best_trainer_rl.save_model("./roberta_large_best_model")
predictions_rl = best_trainer_rl.predict(test_dataset)

y_pred_rl = np.argmax(predictions_rl.predictions, axis=1)
y_true_rl = predictions_rl.label_ids
accuracy_rl = accuracy_score(y_true_rl, y_pred_rl)
precision_rl = precision_score(y_true_rl, y_pred_rl, average='macro')  # or 'macro' for both classes equally
recall_rl = recall_score(y_true_rl, y_pred_rl, average='macro')
f1_rl = f1_score(y_true_rl, y_pred_rl, average='macro')
cm_rl = confusion_matrix(y_true_rl, y_pred_rl)
disp_rl = ConfusionMatrixDisplay(
    confusion_matrix=cm_rl,
    display_labels=["Non-polarized", "Polarized"]
)
disp_rl.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix on Test Set")
plt.show()
dev_pred_rl = best_trainer_rl.predict(dev_test)
y_pred_dev_rl = np.argmax(dev_pred_rl.predictions, axis=1)
y_gpt_rl = dev_pred_rl.label_ids

accuracy_gpt_rl = accuracy_score(y_gpt_rl, y_pred_dev_rl)
precision_gpt_rl  = precision_score(y_gpt_rl, y_pred_dev_rl, average='macro')  # or 'macro' for both classes equally
recall_gpt_rl  = recall_score(y_gpt_rl, y_pred_dev_rl, average='macro')
f1_gpt_rl  = f1_score(y_gpt_rl, y_pred_dev_rl, average='macro')

print(f"Accuracy:  {accuracy_gpt_rl :.4f}")
print(f"Precision: {precision_gpt_rl :.4f}")
print(f"Recall:    {recall_gpt_rl :.4f}")
print(f"F1 Score:  {f1_gpt_rl :.4f}")


cm_gpt_rl = confusion_matrix(y_gpt_rl, y_pred_dev_rl)
disp_gpt_rl = ConfusionMatrixDisplay(
    confusion_matrix=cm_gpt_rl,
    display_labels=["Non-polarized", "Polarized"]
)
disp_gpt_rl.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix on Dev Set (Compared to ChatGPT Results)")
plt.show()
#endregion

#region error analysis
#reload results (saved them out for easier access)
pred_base=pd.read_csv('pred_base.csv')
pred_base_gpt=pd.read_csv('pred_base_gpt.csv')
pred_large=pd.read_csv('pred_large.csv')
pred_large_gpt=pd.read_csv('pred_large_gpt.csv')

y_base_pred=np.argmax(pred_base[['logit_0','logit_1']], axis=1)
y_base_true=pred_base.label.values
y_large_pred=np.argmax(pred_large[['logit_0','logit_1']], axis=1)
y_large_true=pred_large.label.values

y_base_pred_gpt=np.argmax(pred_base_gpt[['logit_0','logit_1']], axis=1)
y_base_true_gpt=pred_base_gpt.label.values
y_large_pred_gpt=np.argmax(pred_large_gpt[['logit_0','logit_1']], axis=1)
y_large_true_gpt=pred_large_gpt.label.values

wrong_base_i=y_base_true!=y_base_pred


temp=pred_base

logits = torch.tensor(temp[["logit_0", "logit_1"]].values)
probs = torch.softmax(logits, dim=1)

temp["prob_0_base"] = probs[:, 0].numpy()
temp["prob_1_base"] = probs[:, 1].numpy()

temp_large=pred_large
logits = torch.tensor(temp_large[["logit_0", "logit_1"]].values)
probs = torch.softmax(logits, dim=1)

temp_large["prob_0_large"] = probs[:, 0].numpy()
temp_large["prob_1_large"] = probs[:, 1].numpy()
temp_large.reset_index()
temp_combo=temp.merge(temp_large[['prob_0_large','prob_1_large']], on=temp.index)

test1=test.copy()
test1.reset_index(inplace=True)
test1=test1.merge(temp_combo[['prob_0_base','prob_1_base','prob_0_large','prob_1_large']], on=temp.index)

test1['pred_base']=y_base_pred
test1['pred_large']=y_large_pred
test1=test1.drop(columns=['id','key_0', 'index'],axis=1)
test1=test1[['text', 'polarization', 'pred_base','prob_0_base','prob_1_base', 'pred_large','prob_0_large', 'prob_1_large']]

mismatches = test1[
    (test1['polarization'] != test1['pred_base']) | 
    (test1['polarization'] != test1['pred_large']) | 
    (test1['pred_base'] != test1['pred_large'])
]
pd.set_option('display.max_colwidth', None)
mismatches.head(20)


test_gpt=pd.read_csv('dev_eng_with_predictions.csv')
test_gpt.drop(columns=['polarization'],axis=1,inplace=True)
test_gpt.rename(columns={'polarization_pred':'polarization'},inplace=True)
import torch

temp=pred_base_gpt

logits = torch.tensor(temp[["logit_0", "logit_1"]].values)
probs = torch.softmax(logits, dim=1)

temp["prob_0_base"] = probs[:, 0].numpy()
temp["prob_1_base"] = probs[:, 1].numpy()

temp_large=pred_large_gpt
logits = torch.tensor(temp_large[["logit_0", "logit_1"]].values)
probs = torch.softmax(logits, dim=1)

temp_large["prob_0_large"] = probs[:, 0].numpy()
temp_large["prob_1_large"] = probs[:, 1].numpy()

temp_combo=temp.merge(temp_large[['prob_0_large','prob_1_large']], on=temp.index)

gpt=test_gpt.copy()
gpt=gpt.merge(temp_combo[['prob_0_base','prob_1_base','prob_0_large','prob_1_large']], on=temp.index)

gpt['pred_base_gpt']=y_base_pred_gpt
gpt['pred_large_gpt']=y_large_pred_gpt
#gpt=gpt.reset_index(inplace=True)
gpt=gpt.drop(columns=['id','key_0'],axis=1)
gpt=gpt[['text', 'polarization', 'pred_base_gpt','prob_0_base','prob_1_base', 'pred_large_gpt','prob_0_large', 'prob_1_large']]
mismatches_gpt = gpt[
    (gpt['polarization'] != gpt['pred_base_gpt']) | 
    (gpt['polarization'] != gpt['pred_large_gpt']) | 
    (gpt['pred_base_gpt'] != gpt['pred_large_gpt'])
]
mismatches_gpt.head()