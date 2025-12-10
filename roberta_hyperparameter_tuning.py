import pandas as pd
import torch
import numpy as np
import os
import gc
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import ParameterGrid, StratifiedKFold, train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)



# Paths 
TRAIN_DATA_PATH = "/Users/arwynlewis/Desktop/NLP/NLP2025_SemEvalTask9/dev_phase/subtask1/train/eng.csv"
TEST_DATA_PATH = "/Users/arwynlewis/Desktop/NLP/NLP2025_SemEvalTask9/dev_phase/subtask1/dev/eng.csv"
OUTPUT_DIR = "./roberta_output"
FINAL_MODEL_DIR = "./final_roberta_polarization"
SUBMISSION_FILE = "submission_roberta.csv"

# Training configuration
USE_CPU = True  # Set to True to force CPU training 
N_FOLDS = 3
RANDOM_STATE = 42

# Hyperparameter grid
PARAM_GRID = {
    "learning_rate": [2e-5, 3e-5],
    "num_train_epochs": [3],
    "per_device_train_batch_size": [1, 2],
    "gradient_accumulation_steps": [32]
}




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



def setup_device(use_cpu=True):
    if use_cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
        device = torch.device('cpu')
        print("Using CPU")
    elif torch.backends.mps.is_available():
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
        device = torch.device('mps')
        print("Using MPS (Apple Silicon GPU)")
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA GPU")
    else:
        device = torch.device('cpu')
        print("Using CPU (no GPU available)")
    
    return device


def cleanup_memory(use_cpu=True):
    gc.collect()
    
    if not use_cpu:
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {'f1_macro': f1_score(p.label_ids, preds, average='macro')}



def run_hyperparameter_tuning(train_data, tokenizer, param_grid, n_folds=3, use_cpu=True):
    grid = list(ParameterGrid(param_grid))
    print(f"\n{'='*80}")
    print(f"HYPERPARAMETER TUNING")
    print(f"{'='*80}")
    print(f"Total parameter combinations: {len(grid)}")
    print(f"Cross-validation folds: {n_folds}")
    print(f"Total models to train: {len(grid) * n_folds}")
    print(f"{'='*80}\n")
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    all_results = []
    device = setup_device(use_cpu)
    
    for param_idx, params in enumerate(grid):
        print(f"\n{'='*80}")
        print(f"Testing parameter combination {param_idx + 1}/{len(grid)}")
        print(f"Params: {params}")
        print(f"Effective batch size: {params['per_device_train_batch_size'] * params['gradient_accumulation_steps']}")
        print(f"{'='*80}\n")
        
        f1_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_data['text'], 
                                                                   train_data['polarization'])):
            print(f"Fold {fold_idx + 1}/{n_folds}")
            
            # Clean memory before each fold
            cleanup_memory(use_cpu)
            
            # Split data
            train_fold = train_data.iloc[train_idx].reset_index(drop=True)
            val_fold = train_data.iloc[val_idx].reset_index(drop=True)

            # Create datasets
            train_dataset = PolarizationDataset(
                train_fold['text'].tolist(),
                train_fold['polarization'].tolist(),
                tokenizer
            )
            val_dataset = PolarizationDataset(
                val_fold['text'].tolist(),
                val_fold['polarization'].tolist(),
                tokenizer
            )
            
            # Initialize model
            model = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
            model.gradient_checkpointing_enable()
            
            if not use_cpu:
                model = model.to(device)

            # Training arguments
            training_args = TrainingArguments(
                output_dir=OUTPUT_DIR,
                num_train_epochs=params['num_train_epochs'],
                learning_rate=params['learning_rate'],
                per_device_train_batch_size=params['per_device_train_batch_size'],
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=params['gradient_accumulation_steps'],
                eval_strategy="epoch",
                save_strategy="no",
                logging_steps=100,
                disable_tqdm=False,
                report_to=[],
                fp16=False,
                dataloader_num_workers=0,
                load_best_model_at_end=False,
                dataloader_pin_memory=False,
                max_grad_norm=1.0,
                gradient_checkpointing=True,
                optim="adamw_torch",
                save_total_limit=0,
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics,
                data_collator=DataCollatorWithPadding(tokenizer)
            )

            try:
                trainer.train()
                eval_results = trainer.evaluate()
                f1_scores.append(eval_results['eval_f1_macro'])
                print(f"Fold {fold_idx + 1} F1: {eval_results['eval_f1_macro']:.4f}\n")
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n{'!'*80}")
                    print("MEMORY ERROR!")
                    print(f"{'!'*80}")
                    print("Try: USE_CPU = True at top of script")
                    print(f"{'!'*80}\n")
                    raise
                else:
                    raise
            
            # Cleanup again
            del model, trainer, train_dataset, val_dataset
            cleanup_memory(use_cpu)

        # Aggregate results
        if f1_scores:
            mean_f1 = np.mean(f1_scores)
            std_f1 = np.std(f1_scores)
            print(f"\n{'='*80}")
            print(f"Params: {params}")
            print(f"Mean Macro F1: {mean_f1:.4f} (+/- {std_f1:.4f})")
            print(f"{'='*80}\n")
            
            all_results.append({
                'params': params,
                'mean_f1': mean_f1,
                'std_f1': std_f1,
                'fold_scores': f1_scores
            })
    
    return all_results


def display_tuning_results(all_results):
    if not all_results:
        print("No results available.")
        return None
    
    # Create results DataFrame
    results_df = pd.DataFrame([
        {
            'learning_rate': r['params']['learning_rate'],
            'epochs': r['params']['num_train_epochs'],
            'batch_size': r['params']['per_device_train_batch_size'],
            'grad_accum': r['params']['gradient_accumulation_steps'],
            'effective_batch': r['params']['per_device_train_batch_size'] * r['params']['gradient_accumulation_steps'],
            'mean_f1': r['mean_f1'],
            'std_f1': r['std_f1']
        }
        for r in all_results
    ])
    
    results_df = results_df.sort_values('mean_f1', ascending=False)
    
    print("\n" + "="*80)
    print("ALL RESULTS (sorted by Mean F1):")
    print("="*80)
    print(results_df.to_string(index=False))
    
    # Best parameters
    best_result = max(all_results, key=lambda x: x['mean_f1'])
    print("\n" + "="*80)
    print("BEST PARAMETERS:")
    print("="*80)
    print(f"Parameters: {best_result['params']}")
    print(f"Mean Macro F1: {best_result['mean_f1']:.4f} (+/- {best_result['std_f1']:.4f})")
    print(f"Fold scores: {[f'{score:.4f}' for score in best_result['fold_scores']]}")
    print("="*80)
    
    return best_result



def train_final_model(train_data, tokenizer, best_params, use_cpu=True):
    print("\n" + "="*80)
    print("TRAINING FINAL MODEL")
    print("="*80)
    print(f"Training samples: {len(train_data)}")
    print(f"Using parameters: {best_params}")
    print("="*80 + "\n")
    
    device = setup_device(use_cpu)
    
    # Create full dataset
    full_dataset = PolarizationDataset(
        train_data['text'].tolist(),
        train_data['polarization'].tolist(),
        tokenizer
    )
    
    # Initialize model
    if use_cpu:
        final_model = AutoModelForSequenceClassification.from_pretrained(
            'roberta-base', 
            num_labels=2,
            device_map='cpu',
            torch_dtype=torch.float32
        )
    else:
        final_model = AutoModelForSequenceClassification.from_pretrained(
            'roberta-base', 
            num_labels=2
        )
        final_model = final_model.to(device)
    
    final_model.gradient_checkpointing_enable()
    
    # Training arguments with best hyperparameters
    training_args = TrainingArguments(
        output_dir=FINAL_MODEL_DIR,
        num_train_epochs=best_params['num_train_epochs'],
        learning_rate=best_params['learning_rate'],
        per_device_train_batch_size=best_params['per_device_train_batch_size'],
        gradient_accumulation_steps=best_params['gradient_accumulation_steps'],
        save_strategy="epoch",
        logging_steps=50,
        report_to=[],
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        optim="adamw_torch",
        save_total_limit=1,
        no_cuda=use_cpu,
        use_cpu=use_cpu,
    )
    
    # Train
    trainer = Trainer(
        model=final_model,
        args=training_args,
        train_dataset=full_dataset,
        data_collator=DataCollatorWithPadding(tokenizer)
    )
    
    trainer.train()
    
    # Save
    print(f"\nSaving model to {FINAL_MODEL_DIR}")
    trainer.save_model(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)
    
    return trainer



def generate_predictions(trainer, tokenizer, test_data):
    print("\n" + "="*80)
    print("GENERATING PREDICTIONS")
    print("="*80)
    print(f"Test samples: {len(test_data)}")
    
    # Handle labels
    if 'polarization' in test_data.columns and test_data['polarization'].notna().any():
        test_data_clean = test_data.dropna(subset=['polarization']).reset_index(drop=True)
        test_labels = test_data_clean['polarization'].tolist()
        has_labels = True
        print(f"Found {len(test_data_clean)} samples with labels")
        test_data = test_data_clean
    else:
        test_labels = [0] * len(test_data)
        has_labels = False
        print("No labels found - generating predictions only")
    
    # Create dataset
    test_dataset = PolarizationDataset(
        test_data['text'].tolist(),
        test_labels,
        tokenizer
    )
    
    # Predict
    predictions = trainer.predict(test_dataset)
    predicted_labels = np.argmax(predictions.predictions, axis=1)
    prediction_probs = torch.softmax(torch.tensor(predictions.predictions), dim=1).numpy()
    
    return predicted_labels, prediction_probs, has_labels, test_data


def evaluate_predictions(test_data, predicted_labels, prediction_probs, has_labels, cv_f1=None):
    
    # Evaluation metrics
    if has_labels:
        f1 = f1_score(test_data['polarization'], predicted_labels, average='macro')
        print("\n" + "="*80)
        print("TEST SET PERFORMANCE")
        print("="*80)
        print(f"Test Set Macro F1: {f1:.4f}")
        if cv_f1:
            print(f"CV Mean F1: {cv_f1:.4f}")
            print(f"Difference: {f1 - cv_f1:+.4f}")
        print("="*80 + "\n")
        
        print(classification_report(
            test_data['polarization'], 
            predicted_labels,
            target_names=['Non-Polarized (0)', 'Polarized (1)'],
            digits=4
        ))
    
    # Statistics
    print("\n" + "="*80)
    print("PREDICTION STATISTICS")
    print("="*80)
    print(f"Non-Polarized (0): {(predicted_labels == 0).sum():>5} ({(predicted_labels == 0).sum()/len(predicted_labels)*100:>5.1f}%)")
    print(f"Polarized (1):     {(predicted_labels == 1).sum():>5} ({(predicted_labels == 1).sum()/len(predicted_labels)*100:>5.1f}%)")
    print(f"\nAverage Confidence: {prediction_probs.max(axis=1).mean():.3f}")
    print("="*80)
    
    # Sample predictions
    print("\n" + "="*80)
    print("SAMPLE PREDICTIONS (first 10)")
    print("="*80)
    print(f"{'ID':<40} {'Pred':<15} {'Conf':<10} {'Text'}")
    print("="*80)
    
    for i in range(min(10, len(test_data))):
        pred_label = "Polarized" if predicted_labels[i] == 1 else "Non-Polarized"
        conf = prediction_probs[i].max()
        text = test_data['text'].iloc[i]
        text_short = text[:40] + "..." if len(text) > 40 else text
        row_id = test_data['id'].iloc[i] if 'id' in test_data.columns else i
        print(f"{str(row_id):<40} {pred_label:<15} {conf:<10.3f} {text_short}")
    
    print("="*80)


def save_submission(test_data, predicted_labels, filename=SUBMISSION_FILE):
    """Save predictions to submission file."""
    submission = pd.DataFrame({
        'id': test_data['id'] if 'id' in test_data.columns else test_data.index,
        'polarization': predicted_labels
    })
    
    submission.to_csv(filename, index=False)
    print(f"\nPredictions saved to '{filename}'")




def main():
    
    # Load data
    train_raw = pd.read_csv(TRAIN_DATA_PATH)
    train_raw = train_raw.dropna(subset=['text', 'polarization']).reset_index(drop=True)
    print(f"Loaded {len(train_raw)} training samples")
    print(f"   Class distribution: {train_raw['polarization'].value_counts().to_dict()}")
    
    # Split for final evaluation
    train_data, _ = train_test_split(
        train_raw, 
        test_size=0.1, 
        random_state=RANDOM_STATE, 
        stratify=train_raw['polarization']
    )
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    print("Tokenizer loaded")
    
    # Hyperparameter tuning
    print("\n" + "="*80)
    print("STEP 1: HYPERPARAMETER TUNING")
    print("="*80)
    
    tuning_results = run_hyperparameter_tuning(
        train_data=train_data,
        tokenizer=tokenizer,
        param_grid=PARAM_GRID,
        n_folds=N_FOLDS,
        use_cpu=USE_CPU
    )
    
    best_result = display_tuning_results(tuning_results)
    
    if best_result is None:
        print("No tuning results. Exiting.")
        return
    
    # Train final model
    print("\n" + "="*80)
    print("STEP 2: TRAIN FINAL MODEL")
    print("="*80)
    
    trainer = train_final_model(
        train_data=train_raw,  # Use full dataset
        tokenizer=tokenizer,
        best_params=best_result['params'],
        use_cpu=USE_CPU
    )
    
    # Generate predictions
    print("\n" + "="*80)
    print("STEP 3: GENERATE PREDICTIONS")
    print("="*80)
    
    test_data = pd.read_csv(TEST_DATA_PATH)
    predicted_labels, prediction_probs, has_labels, test_data = generate_predictions(
        trainer=trainer,
        tokenizer=tokenizer,
        test_data=test_data
    )
    
    # Evaluate
    evaluate_predictions(
        test_data=test_data,
        predicted_labels=predicted_labels,
        prediction_probs=prediction_probs,
        has_labels=has_labels,
        cv_f1=best_result['mean_f1']
    )
    
    # Save submission
    save_submission(test_data, predicted_labels)
    print(f"\nBest CV F1: {best_result['mean_f1']:.4f}")
    print(f"Model saved to: {FINAL_MODEL_DIR}")
    print(f"Submission saved to: {SUBMISSION_FILE}")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
