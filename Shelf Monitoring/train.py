#!/usr/bin/env python3
# YOLOv8 Shelf Product Detection and Counting
# train.py - Model training implementation

import os
import yaml
import numpy as np
import json
import time
from pathlib import Path
import torch
import cv2
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
from ultralytics import YOLO

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Base path to project folder - all paths are relative to this
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# Configuration with enhanced parameters
CONFIG = {
    'data_yaml': os.path.join(BASE_PATH, 'data', 'data.yaml'),
    'epochs': 50,                     # Increased from 10 to 50 for better convergence
    'model_size': 'm',                # Upgraded from 'n' to 'm' for better performance
    'img_size': 640,                  # Standard size for detection
    'batch_size': 16,                 # Adjust based on GPU memory
    'output_dir': os.path.join(BASE_PATH, 'results'),
    'models_dir': os.path.join(BASE_PATH, 'models'),
    'conf_threshold': 0.25,           # Confidence threshold for detection
    'save_txt': True,                 # Save text results
    'save_conf': True,                # Save confidence in text file
    'augment': True,                  # Enable built-in augmentations
    'early_stopping': True,           # Enable early stopping to prevent overfitting
    'patience': 10,                   # Early stopping patience
    'save_period': 5,                 # Save checkpoint every 5 epochs
}

# Create necessary directories
os.makedirs(CONFIG['output_dir'], exist_ok=True)
os.makedirs(CONFIG['models_dir'], exist_ok=True)
os.makedirs(os.path.join(BASE_PATH, 'data'), exist_ok=True)

# Load data configuration
def load_data_config(yaml_path):
    """Load and process the data configuration from YAML."""
    with open(yaml_path, 'r') as file:
        data_config = yaml.safe_load(file)

    # Update paths to be relative to the project folder
    for key in ['train', 'val', 'test']:
        if key in data_config:
            # If path is absolute, leave it; if relative, make it relative to BASE_PATH
            if not os.path.isabs(data_config[key]):
                data_config[key] = os.path.join(BASE_PATH, data_config[key])

    return data_config

# Training the model with enhanced parameters
def train_model(data_yaml, model_size='m', epochs=50, img_size=640, batch_size=16, augment=True):
    """Train a YOLOv8 model with enhanced parameters."""
    # Load a pre-trained YOLOv8 model
    model = YOLO(f'yolov8{model_size}.pt')

    # Train the model with enhanced parameters
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=0 if torch.cuda.is_available() else 'cpu',
        project=os.path.join(CONFIG['output_dir'], 'training'),
        name='train_run',
        exist_ok=True,
        plots=True,
        augment=augment,              # Enable data augmentation
        mixup=0.2,                    # Apply mixup augmentation
        mosaic=1.0,                   # Apply mosaic augmentation
        degrees=10.0,                 # Rotation augmentation
        translate=0.1,                # Translation augmentation
        scale=0.5,                    # Scale augmentation
        fliplr=0.5,                   # Horizontal flip probability
        flipud=0.1,                   # Vertical flip probability
        hsv_h=0.015,                  # HSV hue augmentation
        hsv_s=0.7,                    # HSV saturation augmentation
        hsv_v=0.4,                    # HSV value augmentation
        lr0=0.01,                     # Initial learning rate
        lrf=0.01,                     # Final learning rate ratio
        momentum=0.937,               # SGD momentum
        weight_decay=0.0005,          # Weight decay
        warmup_epochs=3.0,            # Warmup epochs
        warmup_momentum=0.8,          # Warmup momentum
        warmup_bias_lr=0.1,           # Warmup bias learning rate
        box=7.5,                      # Box loss gain
        cls=0.5,                      # Class loss gain
        dfl=1.5,                      # DFL loss gain
        patience=CONFIG['patience'],  # Early stopping patience
        save_period=CONFIG['save_period'],  # Save every N epochs
    )

    # Return the trained model and results
    return model, results

# Evaluating the model and logging metrics
def evaluate_model(model, data_yaml, output_dir):
    """Validate the model and log metrics."""
    # Validate the model
    metrics = model.val(
        data=data_yaml,
        project=os.path.join(output_dir, 'validation'),
        name='val_run'
    )

    # Log and visualize metrics
    metrics_dict = log_metrics(metrics, output_dir)

    return metrics, metrics_dict

# Log and visualize metrics
def log_metrics(metrics, output_dir):
    """Log evaluation metrics to a file and print to console."""
    metrics_dir = os.path.join(output_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_file_path = os.path.join(metrics_dir, 'evaluation_metrics.txt')

    # Extract key metrics from YOLOv8 metrics object
    try:
        # Try to access common YOLOv8 metrics
        map50 = metrics.box.map50  # mAP at IoU 0.5
        map = metrics.box.map      # mAP at IoU 0.5:0.95
        precision = metrics.box.p  # Precision
        recall = metrics.box.r     # Recall
        f1 = metrics.box.f1        # F1-score

        # Get per-class metrics if available
        class_map50 = metrics.box.ap_class_dict.get(50, {})
        class_precision = metrics.box.p_class
        class_recall = metrics.box.r_class
        class_f1 = metrics.box.f1_class

        # Create metrics dictionary
        metrics_dict = {
            "mAP@0.5": float(map50),
            "mAP@0.5:0.95": float(map),
            "Precision": float(precision),
            "Recall": float(recall),
            "F1-Score": float(f1),
        }

        # Log main metrics to file
        with open(metrics_file_path, 'w') as f:
            f.write("YOLOV8 EVALUATION METRICS\n")
            f.write("=" * 50 + "\n\n")

            # Overall metrics
            f.write("OVERALL METRICS:\n")
            for metric_name, value in metrics_dict.items():
                f.write(f"{metric_name}: {value:.4f}\n")

            # Per-class metrics
            f.write("\nPER-CLASS METRICS:\n")
            for i, class_name in enumerate(metrics.names):
                if i < len(class_precision):
                    f.write(f"\nClass '{class_name}':\n")
                    f.write(f"  Precision: {float(class_precision[i]):.4f}\n")
                    f.write(f"  Recall: {float(class_recall[i]):.4f}\n")
                    f.write(f"  F1-Score: {float(class_f1[i]):.4f}\n")
                    if i in class_map50:
                        f.write(f"  mAP@0.5: {float(class_map50[i]):.4f}\n")

        # Print main metrics to console
        print("\nEVALUATION METRICS:")
        print("=" * 50)
        for metric_name, value in metrics_dict.items():
            print(f"{metric_name}: {value:.4f}")

        print(f"\nMetrics saved to: {metrics_file_path}")

        # Create visualization for overall metrics
        plt.figure(figsize=(10, 6))
        plt.bar(metrics_dict.keys(), metrics_dict.values(), color='teal')
        plt.ylim(0, 1)
        plt.title('YOLOv8 Model Overall Performance Metrics')
        plt.tight_layout()

        # Save the metrics plot
        metrics_plot_path = os.path.join(metrics_dir, 'metrics_overall.png')
        plt.savefig(metrics_plot_path)
        plt.close()

        # Save per-class metrics visualization if available
        if len(class_precision) > 0:
            # Get top N classes for visualization
            top_n = min(15, len(class_precision))

            # Sort classes by F1 score
            class_indices = list(range(len(class_f1)))
            class_indices.sort(key=lambda i: float(class_f1[i]), reverse=True)
            class_indices = class_indices[:top_n]

            # Prepare data for visualization
            top_classes = [metrics.names[i] for i in class_indices]
            top_precision = [float(class_precision[i]) for i in class_indices]
            top_recall = [float(class_recall[i]) for i in class_indices]
            top_f1 = [float(class_f1[i]) for i in class_indices]

            # Create per-class metrics visualization
            fig, ax = plt.subplots(figsize=(12, 8))
            x = np.arange(len(top_classes))
            width = 0.25

            ax.bar(x - width, top_precision, width, label='Precision', color='royalblue')
            ax.bar(x, top_recall, width, label='Recall', color='teal')
            ax.bar(x + width, top_f1, width, label='F1-Score', color='coral')

            ax.set_ylim(0, 1)
            ax.set_ylabel('Score')
            ax.set_title(f'Top {top_n} Classes by F1-Score')
            ax.set_xticks(x)
            ax.set_xticklabels(top_classes, rotation=45, ha='right')
            ax.legend()
            plt.tight_layout()

            # Save the per-class metrics plot
            per_class_plot_path = os.path.join(metrics_dir, 'metrics_per_class.png')
            plt.savefig(per_class_plot_path)
            plt.close()

        # Also save metrics as JSON for later use
        json_path = os.path.join(metrics_dir, 'metrics.json')
        with open(json_path, 'w') as f:
            json.dump(metrics_dict, f, indent=4)

        return metrics_dict

    except Exception as e:
        print(f"Error logging metrics: {e}")
        print("Saving raw metrics object to file")

        # Save whatever we can get from the metrics object
        with open(metrics_file_path, 'w') as f:
            f.write(f"Raw metrics: {str(metrics)}\n")

        return {"error": str(e)}

# Run inference on test set
def run_inference(model, test_dir, output_dir, conf_threshold=0.25):
    """Run inference on test images and collect statistics."""
    # Create output directory if it doesn't exist
    detection_dir = os.path.join(output_dir, 'detections')
    os.makedirs(detection_dir, exist_ok=True)

    # Get test images
    test_images = []
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        test_images.extend(list(Path(test_dir).glob(ext)))
    test_images = sorted(test_images)

    if not test_images:
        print(f"No images found in {test_dir}")
        return Counter()

    print(f"Found {len(test_images)} test images")

    # Initialize counter for all detections
    all_detections = Counter()

    # Store confidence scores for analysis
    confidence_scores = {}

    # Run inference on each image
    for img_path in tqdm(test_images, desc="Running inference"):
        # Run inference
        results = model(img_path, conf=conf_threshold)[0]

        # Get detections
        boxes = results.boxes
        class_ids = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()

        # Update detection counters
        for i, cls_id in enumerate(class_ids):
            class_name = model.names[cls_id]
            all_detections[class_name] += 1

            # Store confidence scores
            if class_name not in confidence_scores:
                confidence_scores[class_name] = []
            confidence_scores[class_name].append(float(confidences[i]))

        # Save detection image
        annotated_img = results.plot()
        output_path = os.path.join(detection_dir, os.path.basename(img_path))
        cv2.imwrite(output_path, annotated_img)

    # Save confidence score distribution
    confidence_dir = os.path.join(output_dir, 'confidence_analysis')
    os.makedirs(confidence_dir, exist_ok=True)

    # Plot confidence distributions for top 10 classes
    if confidence_scores:
        top_classes = [class_name for class_name, _ in all_detections.most_common(10)]

        plt.figure(figsize=(12, 8))

        for class_name in top_classes:
            if class_name in confidence_scores and confidence_scores[class_name]:
                plt.hist(confidence_scores[class_name], bins=20, alpha=0.5, label=class_name)

        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Confidence Score Distribution for Top 10 Classes')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(confidence_dir, 'confidence_distribution.png'))
        plt.close()

        # Save confidence stats to file
        conf_stats_path = os.path.join(confidence_dir, 'confidence_stats.txt')
        with open(conf_stats_path, 'w') as f:
            f.write("CONFIDENCE SCORE STATISTICS\n")
            f.write("=" * 50 + "\n\n")

            for class_name, scores in confidence_scores.items():
                if scores:
                    f.write(f"{class_name}:\n")
                    f.write(f"  Count: {len(scores)}\n")
                    f.write(f"  Mean: {np.mean(scores):.4f}\n")
                    f.write(f"  Min: {np.min(scores):.4f}\n")
                    f.write(f"  Max: {np.max(scores):.4f}\n")
                    f.write(f"  Std: {np.std(scores):.4f}\n\n")

    return all_detections

# Save detection counts to file and print to console
def save_and_print_counts(detections, output_dir, class_names):
    """Save and print detection counts by class."""
    # Sort detections by count (descending)
    sorted_detections = sorted(detections.items(), key=lambda x: x[1], reverse=True)

    # Save to file
    count_file_path = os.path.join(output_dir, 'class_counts.txt')
    with open(count_file_path, 'w') as f:
        f.write("PRODUCT CLASS COUNTS (SORTED BY FREQUENCY)\n")
        f.write("="*50 + "\n\n")

        for i, (class_name, count) in enumerate(sorted_detections, 1):
            line = f"{i}. {class_name}: {count} objects\n"
            f.write(line)

        total_objects = sum(detections.values())
        f.write(f"\nTOTAL OBJECTS DETECTED: {total_objects}\n")
        f.write(f"TOTAL UNIQUE CLASSES DETECTED: {len(detections)}\n")

    # Print to console
    print("\nPRODUCT CLASS COUNTS (SORTED BY FREQUENCY)")
    print("="*50)

    for i, (class_name, count) in enumerate(sorted_detections, 1):
        print(f"{i}. {class_name}: {count} objects")

    print(f"\nTOTAL OBJECTS DETECTED: {sum(detections.values())}")
    print(f"TOTAL UNIQUE CLASSES DETECTED: {len(detections)}")
    print(f"\nDetailed count saved to: {count_file_path}")

# Visualize detection counts
def plot_detection_counts(detections, output_dir, top_n=15):
    """Create a bar chart of the top N detected classes."""
    if not detections:
        print("No detections to visualize")
        return

    # Sort detections by count (descending)
    sorted_detections = sorted(detections.items(), key=lambda x: x[1], reverse=True)

    # Take top N for better visualization
    top_detections = sorted_detections[:min(top_n, len(sorted_detections))]
    classes, counts = zip(*top_detections)

    # Create figure and axis
    plt.figure(figsize=(12, 8))
    bars = plt.bar(classes, counts, color='royalblue')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Product Class')
    plt.ylabel('Count')
    plt.title(f'Top {len(top_detections)} Product Detections')

    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{height:.0f}', ha='center', va='bottom')

    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(output_dir, 'top_detections.png')
    plt.savefig(plot_path)
    print(f"Plot saved to: {plot_path}")

# Function to track training progress
def track_training_progress(results_file):
    """Track and visualize training progress over time."""

    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        return

    try:
        # Load training results
        data = np.load(results_file)

        # Extract metrics
        epochs = data['epoch']
        train_box_loss = data['train/box_loss']
        train_cls_loss = data['train/cls_loss']
        train_dfl_loss = data['train/dfl_loss']
        val_box_loss = data['val/box_loss'] if 'val/box_loss' in data else None
        val_cls_loss = data['val/cls_loss'] if 'val/cls_loss' in data else None
        val_dfl_loss = data['val/dfl_loss'] if 'val/dfl_loss' in data else None

        # Precision, recall, mAP metrics
        precision = data['metrics/precision'] if 'metrics/precision' in data else None
        recall = data['metrics/recall'] if 'metrics/recall' in data else None
        map50 = data['metrics/mAP_0.5'] if 'metrics/mAP_0.5' in data else None
        map = data['metrics/mAP_0.5:0.95'] if 'metrics/mAP_0.5:0.95' in data else None

        # Create directory for training progress plots
        progress_dir = os.path.join(CONFIG['output_dir'], 'training_progress')
        os.makedirs(progress_dir, exist_ok=True)

        # Plot training losses
        plt.figure(figsize=(12, 8))
        plt.plot(epochs, train_box_loss, 'b-', label='Box Loss')
        plt.plot(epochs, train_cls_loss, 'r-', label='Class Loss')
        plt.plot(epochs, train_dfl_loss, 'g-', label='DFL Loss')

        if val_box_loss is not None:
            plt.plot(epochs, val_box_loss, 'b--', label='Val Box Loss')
        if val_cls_loss is not None:
            plt.plot(epochs, val_cls_loss, 'r--', label='Val Class Loss')
        if val_dfl_loss is not None:
            plt.plot(epochs, val_dfl_loss, 'g--', label='Val DFL Loss')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(progress_dir, 'training_losses.png'))
        plt.close()

        # Plot metrics
        if precision is not None and recall is not None and map50 is not None and map is not None:
            plt.figure(figsize=(12, 8))
            plt.plot(epochs, precision, 'b-', label='Precision')
            plt.plot(epochs, recall, 'r-', label='Recall')
            plt.plot(epochs, map50, 'g-', label='mAP@0.5')
            plt.plot(epochs, map, 'm-', label='mAP@0.5:0.95')
            plt.xlabel('Epoch')
            plt.ylabel('Metric')
            plt.title('Training Metrics')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(progress_dir, 'training_metrics.png'))
            plt.close()

        print(f"Training progress plots saved to: {progress_dir}")

    except Exception as e:
        print(f"Error tracking training progress: {e}")

# Check if best model exists
def check_best_model_exists():
    """Check if the best model already exists."""
    model_path = os.path.join(CONFIG['models_dir'], 'best_model.pt')
    return os.path.exists(model_path)

# Main execution
def main():
    """Main execution function for training and evaluation."""
    # Record start time
    start_time = time.time()

    # Check if best model already exists
    best_model_exists = check_best_model_exists()
    
    if best_model_exists:
        print(f"Best model already exists at {os.path.join(CONFIG['models_dir'], 'best_model.pt')}")
        print("Skipping training. To retrain, delete or rename the existing model file.")
        
        # Load the existing model for evaluation
        model = YOLO(os.path.join(CONFIG['models_dir'], 'best_model.pt'))
    else:
        # Load data configuration
        data_config = load_data_config(CONFIG['data_yaml'])
        print(f"Dataset has {data_config['nc']} classes")
        print(f"Train path: {data_config['train']}")
        print(f"Validation path: {data_config['val']}")
        print(f"Test path: {data_config['test']}")
        
        # Train model with enhanced parameters
        print("Starting model training with enhanced parameters...")
        model, results = train_model(
            data_yaml=CONFIG['data_yaml'],
            model_size=CONFIG['model_size'],
            epochs=CONFIG['epochs'],
            img_size=CONFIG['img_size'],
            batch_size=CONFIG['batch_size'],
            augment=CONFIG['augment']
        )

        # Track training progress
        results_file = os.path.join(CONFIG['output_dir'], 'training', 'train_run', 'results.csv.npz')
        track_training_progress(results_file)
        
        # Save the model
        best_model_path = Path(os.path.join(CONFIG['output_dir'], 'training', 'train_run', 'weights', 'best.pt'))
        if best_model_path.exists():
            import shutil
            saved_model_path = os.path.join(CONFIG['models_dir'], 'best_model.pt')
            shutil.copy(best_model_path, saved_model_path)
            print(f"Best model saved to {saved_model_path}")
        else:
            print(f"Warning: Best model not found at {best_model_path}")
        
        # Export in ONNX format for broader compatibility
        try:
            onnx_save_path = os.path.join(CONFIG['models_dir'], 'best_model.onnx')
            model.export(format='onnx', imgsz=CONFIG['img_size'], opset=12)
            print(f"Model exported to ONNX format")
        except Exception as e:
            print(f"Error exporting to ONNX: {e}")

    # Load data configuration for evaluation and inference
    data_config = load_data_config(CONFIG['data_yaml'])

    # Evaluate model and log metrics
    print("\nEvaluating model and logging metrics...")
    metrics, metrics_dict = evaluate_model(model, CONFIG['data_yaml'], CONFIG['output_dir'])

    # Run inference on test set
    print("\nRunning inference on test set...")
    test_dir = data_config['test']
    detections = run_inference(
        model,
        test_dir,
        CONFIG['output_dir'],
        conf_threshold=CONFIG['conf_threshold']
    )

    # Save and print detection counts
    print("\nGenerating detection statistics...")
    save_and_print_counts(detections, CONFIG['output_dir'], data_config['names'])

    # Plot detection counts
    plot_detection_counts(detections, CONFIG['output_dir'])

    # Calculate and print total execution time
    end_time = time.time()
    execution_time = end_time - start_time
    hours, remainder = divmod(execution_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTotal execution time: {int(hours)}h {int(minutes)}m {int(seconds)}s")

    # Final summary
    print(f"\nTraining Summary:")
    print(f"Model: YOLOv8{CONFIG['model_size']}")
    print(f"Epochs: {CONFIG['epochs']}")
    print(f"Image Size: {CONFIG['img_size']}")
    if metrics_dict and 'error' not in metrics_dict:
        print(f"mAP@0.5: {metrics_dict.get('mAP@0.5', 'N/A'):.4f}")
        print(f"mAP@0.5:0.95: {metrics_dict.get('mAP@0.5:0.95', 'N/A'):.4f}")
    print(f"\nAll results saved to {CONFIG['output_dir']}")
    print(f"Model saved to {CONFIG['models_dir']}/best_model.pt")

# Run the main function when script is executed directly
if __name__ == "__main__":
    main()