# Shelf Product Detection and Comparison

This project uses YOLOv8 to detect and count products on retail shelves and compare before/after images to identify sold products, added inventory, and other changes.

## Project Structure

```
yolo_shelf_monitor_code/
├── train.py             # Training implementation
├── main.py              # Inference and comparison script
├── data/                # Dataset directory
│   ├── data.yaml        # Dataset configuration
│   ├── train/           # Training data
│   ├── val/             # Validation data
│   └── test/            # Test data (contains images for comparison)
├── models/              # Trained models
│   └── best_model.pt    # Best trained model from training
└── comparison_results/  # Results from product comparison (created by main.py)
```

## Prerequisites

1. Python 3.8 or higher
2. PyTorch
3. Ultralytics (YOLOv8)
4. OpenCV
5. Matplotlib
6. NumPy

Install dependencies with:

```bash
pip install ultralytics matplotlib numpy opencv-python tqdm pyyaml
```

## Usage

### Training (Optional)

If you need to train the model first:

```bash
python train.py
```

This will:
- Train a YOLOv8 model on your dataset
- Save the best model to `models/best_model.pt`
- Generate performance metrics and visualizations

### Product Comparison

To compare two shelf images and analyze product changes:

```bash
python main.py
```

By default, this will:
- Use the model at `models/best_model.pt`
- Select the first two images from `data/test/images/` as before/after images
- Save results to `comparison_results/`

### Command-line Options

You can customize the behavior with these options:

```bash
python main.py --model PATH_TO_MODEL --before BEFORE_IMG --after AFTER_IMG --conf 0.25 --output OUTPUT_DIR
```

Arguments:
- `--model`: Path to the trained YOLOv8 model (default: `models/best_model.pt`)
- `--before`: Path to the "before" image
- `--after`: Path to the "after" image
- `--test-dir`: Directory containing test images (default: `data/test/images/`)
- `--output`: Directory to save results (default: `comparison_results/`)
- `--conf`: Confidence threshold for detection (default: 0.25)

### Examples

Compare specific images:

```bash
python main.py --before data/test/images/image1.jpg --after data/test/images/image2.jpg
```

Use a different model:

```bash
python main.py --model models/custom_model.pt
```

Change confidence threshold:

```bash
python main.py --conf 0.4
```

## Output

The script generates:

1. Console output with:
   - Detection counts for both images
   - Top sold products listed with quantities
   - Analysis summary

2. Files in the `comparison_results/` directory:
   - Annotated images with detection boxes
   - Side-by-side visual comparison
   - Count visualizations for both images
   - Chart of top sold products
   - Detailed text reports of counts and changes

