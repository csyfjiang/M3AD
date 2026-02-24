import argparse
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.preprocessing import label_binarize
import pandas as pd
from datetime import datetime
import glob
import warnings

# Import project modules
from config import get_config
from models import build_model
from data.data_ad_loader import AlzheimerNPZDataset, AlzheimerTransform
from torch.utils.data import DataLoader

# Ignore warnings
warnings.filterwarnings("ignore")

# Diagnosis and change label mappings
DIAGNOSIS_LABELS = {1: 'CN', 2: 'MCI', 3: 'Dementia'}
CHANGE_LABELS = {1: 'Stable', 2: 'Conversion', 3: 'Reversion'}


class AlzheimerEvaluator:
    """Alzheimer's Disease Dual-Task Classification Evaluator"""

    def __init__(self, config, checkpoint_path, device='cuda'):
        """
        Args:
            config: Configuration object
            checkpoint_path: Model checkpoint path
            device: Device
        """
        self.config = config
        self.device = device

        # Build model
        print(f"Building model: {config.MODEL.TYPE}/{config.MODEL.NAME}")
        self.model = build_model(config)

        # Load checkpoint
        print(f"Loading checkpoint: {checkpoint_path}")
        self.load_checkpoint(checkpoint_path)

        # Move to device
        self.model = self.model.to(device)
        self.model.eval()

        # Calculate model parameters
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Number of parameters: {n_parameters:,}")

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            epoch = checkpoint.get('epoch', 'unknown')
            print(f"Loading model from epoch {epoch}")
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # Handle DataParallel wrapped model
        if any(key.startswith('module.') for key in state_dict.keys()):
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace('module.', '')
                new_state_dict[new_key] = value
            state_dict = new_state_dict

        # Load state dict
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)

        if missing_keys:
            print(f"Warning: Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys: {unexpected_keys}")

        print("✓ Model loaded successfully")

    def predict_single(self, npz_path, output_path=None):
        """
        Single sample inference

        Args:
            npz_path: NPZ file path
            output_path: Result output path

        Returns:
            dict: Prediction results
        """
        print(f"Single sample inference: {npz_path}")

        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"NPZ file not found: {npz_path}")

        # Create transform (eval mode, no augmentation)
        transform = AlzheimerTransform(
            output_size=(self.config.DATA.IMG_SIZE, self.config.DATA.IMG_SIZE),
            is_train=False,
            use_crop=False,
            use_rotation=False
        )

        # Load data
        try:
            data = np.load(npz_path)
            sample = {
                'slice': data['slice'].astype(np.float32),
                'label': int(data['label'][()]) if data['label'].shape == () else int(data['label']),
                'change_label': int(data['change_label'][()]) if data['change_label'].shape == () else int(
                    data['change_label']),
                'prior': data['prior'].astype(np.float32),
                'case_name': os.path.splitext(os.path.basename(npz_path))[0]
            }
        except Exception as e:
            raise ValueError(f"Failed to load NPZ file: {e}")

        # Apply transform
        sample = transform(sample)

        # Prepare input
        image = sample['image'].unsqueeze(0).to(self.device)  # [1, 3, H, W]
        clinical_prior = sample['prior'].unsqueeze(0).to(self.device)  # [1, 3]

        # Inference
        with torch.no_grad():
            if torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    outputs = self.model(image, clinical_prior=clinical_prior)
            else:
                outputs = self.model(image, clinical_prior=clinical_prior)

            diag_logits, change_logits = outputs

            # Calculate probabilities
            diag_probs = F.softmax(diag_logits, dim=1)
            change_probs = F.softmax(change_logits, dim=1)

            # Get predictions
            diag_pred = torch.argmax(diag_probs, dim=1).cpu().numpy()[0] + 1  # Convert back to 1,2,3
            change_pred = torch.argmax(change_probs, dim=1).cpu().numpy()[0] + 1

            # Get confidence
            diag_confidence = diag_probs.max().cpu().item()
            change_confidence = change_probs.max().cpu().item()

        # Build result
        result = {
            'case_name': sample['case_name'],
            'file_path': npz_path,
            'ground_truth': {
                'diagnosis': int(sample['label'].item()),
                'diagnosis_name': DIAGNOSIS_LABELS[sample['label'].item()],
                'change': int(sample['change_label'].item()),
                'change_name': CHANGE_LABELS[sample['change_label'].item()]
            },
            'predictions': {
                'diagnosis': int(diag_pred),
                'diagnosis_name': DIAGNOSIS_LABELS[diag_pred],
                'diagnosis_confidence': float(diag_confidence),
                'diagnosis_probabilities': [float(x) for x in diag_probs.cpu().numpy()[0]],
                'change': int(change_pred),
                'change_name': CHANGE_LABELS[change_pred],
                'change_confidence': float(change_confidence),
                'change_probabilities': [float(x) for x in change_probs.cpu().numpy()[0]]
            },
            'clinical_prior': [float(x) for x in sample['prior'].cpu().numpy()],
            'correct': {
                'diagnosis': bool(diag_pred == sample['label'].item()),
                'change': bool(change_pred == sample['change_label'].item())
            },
            'timestamp': datetime.now().isoformat()
        }

        # Save result
        if output_path:
            # Process output path, ensure directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir:  # Only create directory if path is not empty
                os.makedirs(output_dir, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"Result saved to: {output_path}")

        # Print result
        print("\n" + "=" * 60)
        print("Prediction Result Details")
        print("=" * 60)
        print(f"Sample name: {result['case_name']}")
        print(f"File path: {result['file_path']}")

        print(f"\n[Diagnosis Task]")
        print(f"True label: {result['ground_truth']['diagnosis_name']} (Label value: {result['ground_truth']['diagnosis']})")
        print(f"Prediction: {result['predictions']['diagnosis_name']} (Label value: {result['predictions']['diagnosis']})")
        print(f"Correct: {'✓' if result['correct']['diagnosis'] else '✗'}")
        print(f"Class confidences:")
        diag_probs = result['predictions']['diagnosis_probabilities']
        for i, (label_id, label_name) in enumerate([(1, 'CN'), (2, 'MCI'), (3, 'Dementia')], 0):
            marker = " ★" if label_id == result['predictions']['diagnosis'] else ""
            print(f"  {label_name}: {diag_probs[i]:.4f}{marker}")

        print(f"\n[Change Task]")
        print(f"True label: {result['ground_truth']['change_name']} (Label value: {result['ground_truth']['change']})")
        print(f"Prediction: {result['predictions']['change_name']} (Label value: {result['predictions']['change']})")
        print(f"Correct: {'✓' if result['correct']['change'] else '✗'}")
        print(f"Class confidences:")
        change_probs = result['predictions']['change_probabilities']
        for i, (label_id, label_name) in enumerate([(1, 'Stable'), (2, 'Conversion'), (3, 'Reversion')], 0):
            marker = " ★" if label_id == result['predictions']['change'] else ""
            print(f"  {label_name}: {change_probs[i]:.4f}{marker}")

        print(f"\n[Clinical Prior]")
        prior_values = result['clinical_prior']
        prior_names = ['CN Prior', 'MCI Prior', 'AD Prior']
        for i, (name, value) in enumerate(zip(prior_names, prior_values)):
            print(f"  {name}: {value:.4f}")

        print(f"\n[Overall Assessment]")
        print(f"Diagnosis task confidence: {result['predictions']['diagnosis_confidence']:.4f}")
        print(f"Change task confidence: {result['predictions']['change_confidence']:.4f}")
        print(f"Prediction time: {result['timestamp']}")
        print("=" * 60)

        return result

    def predict_batch(self, input_dir, output_dir, recursive=False):
        """
        Batch inference

        Args:
            input_dir: Input directory
            output_dir: Output directory
            recursive: Whether to search recursively

        Returns:
            dict: Batch prediction results
        """
        print(f"Batch inference: {input_dir}")

        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        # Find NPZ files
        if recursive:
            npz_files = glob.glob(os.path.join(input_dir, '**', '*.npz'), recursive=True)
        else:
            npz_files = glob.glob(os.path.join(input_dir, '*.npz'))

        if len(npz_files) == 0:
            raise ValueError(f"No NPZ files found in directory {input_dir}")

        print(f"Found {len(npz_files)} NPZ files")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Create dataset and dataloader
        transform = AlzheimerTransform(
            output_size=(self.config.DATA.IMG_SIZE, self.config.DATA.IMG_SIZE),
            is_train=False,
            use_crop=False,
            use_rotation=False
        )

        dataset = AlzheimerNPZDataset(
            data_dir=input_dir,
            split='eval',
            transform=transform,
            file_list=[os.path.basename(f) for f in npz_files]
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.EVAL.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.DATA.NUM_WORKERS,
            pin_memory=self.config.DATA.PIN_MEMORY
        )

        # Batch inference
        all_results = []
        all_diag_true = []
        all_diag_pred = []
        all_diag_probs = []
        all_change_true = []
        all_change_pred = []
        all_change_probs = []

        print("Starting batch inference...")
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Inference progress"):
                images = batch['image'].to(self.device)
                clinical_priors = batch['prior'].to(self.device)

                # Inference
                if torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images, clinical_prior=clinical_priors)
                else:
                    outputs = self.model(images, clinical_prior=clinical_priors)

                diag_logits, change_logits = outputs

                # Calculate probabilities
                diag_probs = F.softmax(diag_logits, dim=1)
                change_probs = F.softmax(change_logits, dim=1)

                # Get predictions
                diag_preds = torch.argmax(diag_probs, dim=1).cpu().numpy() + 1
                change_preds = torch.argmax(change_probs, dim=1).cpu().numpy() + 1

                # Collect results
                for i in range(len(batch['case_name'])):
                    result = {
                        'case_name': batch['case_name'][i],
                        'ground_truth': {
                            'diagnosis': int(batch['label'][i].item()),
                            'diagnosis_name': DIAGNOSIS_LABELS[batch['label'][i].item()],
                            'change': int(batch['change_label'][i].item()),
                            'change_name': CHANGE_LABELS[batch['change_label'][i].item()]
                        },
                        'predictions': {
                            'diagnosis': int(diag_preds[i]),
                            'diagnosis_name': DIAGNOSIS_LABELS[diag_preds[i]],
                            'diagnosis_confidence': float(diag_probs[i].max()),
                            'diagnosis_probabilities': [float(x) for x in diag_probs[i].cpu().numpy()],
                            'change': int(change_preds[i]),
                            'change_name': CHANGE_LABELS[change_preds[i]],
                            'change_confidence': float(change_probs[i].max()),
                            'change_probabilities': [float(x) for x in change_probs[i].cpu().numpy()]
                        },
                        'clinical_prior': [float(x) for x in batch['prior'][i].cpu().numpy()],
                        'correct': {
                            'diagnosis': bool(diag_preds[i] == batch['label'][i].item()),
                            'change': bool(change_preds[i] == batch['change_label'][i].item())
                        }
                    }
                    all_results.append(result)

                    # Collect data for metrics calculation
                    all_diag_true.append(int(batch['label'][i].item()))
                    all_diag_pred.append(int(diag_preds[i]))
                    all_diag_probs.append(diag_probs[i].cpu().numpy())

                    all_change_true.append(int(batch['change_label'][i].item()))
                    all_change_pred.append(int(change_preds[i]))
                    all_change_probs.append(change_probs[i].cpu().numpy())

        # Calculate overall metrics
        metrics = self.calculate_metrics(
            all_diag_true, all_diag_pred, all_diag_probs,
            all_change_true, all_change_pred, all_change_probs
        )

        # Build final result
        batch_result = {
            'summary': {
                'total_samples': len(all_results),
                'input_directory': input_dir,
                'output_directory': output_dir,
                'timestamp': datetime.now().isoformat()
            },
            'metrics': metrics,
            'predictions': all_results
        }

        # Save results
        results_path = os.path.join(output_dir, 'batch_predictions.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(batch_result, f, indent=2, ensure_ascii=False)

        # Save metrics
        metrics_path = os.path.join(output_dir, 'evaluation_metrics.json')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        # Generate visualizations
        if self.config.EVAL.VISUALIZATION:
            self.generate_visualizations(
                all_diag_true, all_diag_pred, all_diag_probs,
                all_change_true, all_change_pred, all_change_probs,
                output_dir
            )

        # Print summary
        print(f"\n=== Batch inference completed ===")
        print(f"Total samples: {len(all_results)}")
        print(f"Diagnosis accuracy: {metrics['diagnosis']['accuracy']:.3f}")
        print(f"Change accuracy: {metrics['change']['accuracy']:.3f}")
        print(f"Results saved to: {output_dir}")

        return batch_result

    def calculate_metrics(self, diag_true, diag_pred, diag_probs,
                          change_true, change_pred, change_probs):
        """Calculate evaluation metrics"""

        def compute_task_metrics(y_true, y_pred, y_probs, task_name):
            """Calculate metrics for single task"""
            # Convert to 0-based index for sklearn
            y_true_0 = [y - 1 for y in y_true]
            y_pred_0 = [y - 1 for y in y_pred]

            # Basic metrics
            accuracy = accuracy_score(y_true_0, y_pred_0)
            f1_macro = f1_score(y_true_0, y_pred_0, average='macro')
            f1_weighted = f1_score(y_true_0, y_pred_0, average='weighted')
            precision_macro = precision_score(y_true_0, y_pred_0, average='macro')
            recall_macro = recall_score(y_true_0, y_pred_0, average='macro')

            # Confusion matrix
            cm = confusion_matrix(y_true_0, y_pred_0, labels=[0, 1, 2])

            # Classification report
            report = classification_report(y_true_0, y_pred_0, labels=[0, 1, 2],
                                           target_names=['Class_1', 'Class_2', 'Class_3'],
                                           output_dict=True)

            # ROC AUC (multi-class)
            try:
                y_true_bin = label_binarize(y_true_0, classes=[0, 1, 2])
                y_probs_array = np.array(y_probs)

                if y_true_bin.shape[1] > 2:  # Multi-class
                    auc = roc_auc_score(y_true_bin, y_probs_array, multi_class='ovr', average='macro')
                else:  # Binary
                    auc = roc_auc_score(y_true_0, y_probs_array[:, 1])
            except:
                auc = None

            return {
                'accuracy': float(accuracy),
                'f1_macro': float(f1_macro),
                'f1_weighted': float(f1_weighted),
                'precision_macro': float(precision_macro),
                'recall_macro': float(recall_macro),
                'auc': float(auc) if auc is not None else None,
                'confusion_matrix': cm.tolist(),
                'classification_report': report
            }

        # Calculate metrics for both tasks
        diag_metrics = compute_task_metrics(diag_true, diag_pred, diag_probs, 'diagnosis')
        change_metrics = compute_task_metrics(change_true, change_pred, change_probs, 'change')

        # Average metrics
        avg_metrics = {
            'accuracy': (diag_metrics['accuracy'] + change_metrics['accuracy']) / 2,
            'f1_macro': (diag_metrics['f1_macro'] + change_metrics['f1_macro']) / 2,
            'f1_weighted': (diag_metrics['f1_weighted'] + change_metrics['f1_weighted']) / 2
        }

        return {
            'diagnosis': diag_metrics,
            'change': change_metrics,
            'average': avg_metrics
        }

    def generate_visualizations(self, diag_true, diag_pred, diag_probs,
                                change_true, change_pred, change_probs, output_dir):
        """Generate visualization charts"""
        print("Generating visualization charts...")

        # Set matplotlib Chinese font
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        # 1. Confusion matrices
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Diagnosis task confusion matrix
        diag_true_0 = [y - 1 for y in diag_true]
        diag_pred_0 = [y - 1 for y in diag_pred]
        cm_diag = confusion_matrix(diag_true_0, diag_pred_0, labels=[0, 1, 2])

        sns.heatmap(cm_diag, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['CN', 'MCI', 'Dementia'],
                    yticklabels=['CN', 'MCI', 'Dementia'],
                    ax=axes[0])
        axes[0].set_title('Diagnosis Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('True')

        # Change task confusion matrix
        change_true_0 = [y - 1 for y in change_true]
        change_pred_0 = [y - 1 for y in change_pred]
        cm_change = confusion_matrix(change_true_0, change_pred_0, labels=[0, 1, 2])

        sns.heatmap(cm_change, annot=True, fmt='d', cmap='Greens',
                    xticklabels=['Stable', 'Conversion', 'Reversion'],
                    yticklabels=['Stable', 'Conversion', 'Reversion'],
                    ax=axes[1])
        axes[1].set_title('Change Confusion Matrix')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('True')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 2. ROC curves
        if len(set(diag_true)) > 2:  # Multi-class ROC
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))

            # Diagnosis task ROC
            diag_true_bin = label_binarize(diag_true_0, classes=[0, 1, 2])
            diag_probs_array = np.array(diag_probs)

            for i, class_name in enumerate(['CN', 'MCI', 'Dementia']):
                fpr, tpr, _ = roc_curve(diag_true_bin[:, i], diag_probs_array[:, i])
                auc = roc_auc_score(diag_true_bin[:, i], diag_probs_array[:, i])
                axes[0].plot(fpr, tpr, label=f'{class_name} (AUC = {auc:.3f})')

            axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
            axes[0].set_xlabel('False Positive Rate')
            axes[0].set_ylabel('True Positive Rate')
            axes[0].set_title('Diagnosis ROC Curves')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Change task ROC
            change_true_bin = label_binarize(change_true_0, classes=[0, 1, 2])
            change_probs_array = np.array(change_probs)

            for i, class_name in enumerate(['Stable', 'Conversion', 'Reversion']):
                fpr, tpr, _ = roc_curve(change_true_bin[:, i], change_probs_array[:, i])
                auc = roc_auc_score(change_true_bin[:, i], change_probs_array[:, i])
                axes[1].plot(fpr, tpr, label=f'{class_name} (AUC = {auc:.3f})')

            axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
            axes[1].set_xlabel('False Positive Rate')
            axes[1].set_ylabel('True Positive Rate')
            axes[1].set_title('Change ROC Curves')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
            plt.close()

        # 3. Accuracy comparison
        diag_acc = accuracy_score(diag_true_0, diag_pred_0)
        change_acc = accuracy_score(change_true_0, change_pred_0)
        avg_acc = (diag_acc + change_acc) / 2

        fig, ax = plt.subplots(figsize=(8, 6))
        tasks = ['Diagnosis', 'Change', 'Average']
        accs = [diag_acc, change_acc, avg_acc]
        colors = ['skyblue', 'lightgreen', 'lightcoral']

        bars = ax.bar(tasks, accs, color=colors, alpha=0.7)
        ax.set_ylabel('Accuracy')
        ax.set_title('Task Accuracy Comparison')
        ax.set_ylim(0, 1)

        # Add value labels
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')

        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Visualization charts saved to: {output_dir}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser('Alzheimer Dual-Task Classification Evaluation')

    # Basic settings
    parser.add_argument('--cfg', type=str, default='configs/eval.yaml',
                        help='Configuration file path')
    parser.add_argument('--opts', nargs='+', default=None,
                        help='Modify configuration options')

    # Evaluation mode
    parser.add_argument('--mode', type=str, choices=['single', 'batch'], required=True,
                        help='Evaluation mode: single(single sample) or batch(batch inference)')

    # Input output
    parser.add_argument('--input', type=str, required=True,
                        help='Input path: NPZ file path for single mode, directory path for batch mode')
    parser.add_argument('--output', type=str,
                        help='Output path: JSON file path for single mode, directory path for batch mode')

    # Model settings
    parser.add_argument('--checkpoint', type=str,
                        help='Model checkpoint path (if not specified in config file)')

    # Distributed training parameters (compatible with config.py)
    parser.add_argument('--local_rank', type=int, default=0,
                        help='local rank for distributed training')

    # Other settings
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device selection: cuda or cpu')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for batch inference')
    parser.add_argument('--recursive', action='store_true',
                        help='Whether to search subdirectories recursively in batch mode')

    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    # Load configuration
    config = get_config(args)

    # Set checkpoint path
    if args.checkpoint:
        config.defrost()
        config.EVAL.CHECKPOINT_PATH = args.checkpoint
        config.freeze()

    # Check checkpoint path
    if not config.EVAL.CHECKPOINT_PATH:
        raise ValueError("Checkpoint path must be specified (via config file or --checkpoint parameter)")

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")

    # Create evaluator
    evaluator = AlzheimerEvaluator(config, config.EVAL.CHECKPOINT_PATH, device)

    if args.mode == 'single':
        # Single sample inference
        output_path = args.output or config.EVAL.SINGLE.OUTPUT_PATH or './single_prediction.json'
        result = evaluator.predict_single(args.input, output_path)

    elif args.mode == 'batch':
        # Batch inference
        output_dir = args.output or config.EVAL.BATCH.OUTPUT_DIR or './batch_predictions'
        result = evaluator.predict_batch(args.input, output_dir, args.recursive)

    print("\nEvaluation completed!")


if __name__ == '__main__':
    main()