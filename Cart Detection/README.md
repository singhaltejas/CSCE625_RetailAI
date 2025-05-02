# Cart Detection

This project leverages **YOLONAS** (a YOLO-based neural architecture search model) for real-time object detection and tracking in a retail store environment. It uses cart-mounted video footage to detect and count items added to the cart, assisting in inventory management and automation in smart retail.

## Features

- Uses YOLONAS (YOLO NAS) for accurate object detection  
- Processes pre-recorded retail store videos (e.g., cart footage)  
- Real-time item classification and tracking  
- Live item counting and annotation overlay  
- Easily customizable for different product classes  

## Tech Stack

- Python  
- [Supervision](https://github.com/roboflow/supervision) (for annotation/visualization)  
- [YOLO-NAS](https://github.com/Deci-AI/super-gradients) via SuperGradients  
- OpenCV  
- NumPy           


## Usage

1.  **Clone the repo**

    ```
    git clone https://github.com/singhaltejas/YOLONAS_Retail_Store.git
    cd YOLONAS_Retail_Store
    pip install -r requirements.txt
    ```

2.  **Open the notebook:**

    ```
    jupyter notebook YOLONAS_Retail_Store.ipynb
    ```

3.  **Run all cells** to perform object detection and see visual tracking.


## Model Details

*   **Model:** YOLO-NAS (S variant used)
*   **Backend:** SuperGradients
*   **Input:** Retail cart camera video
*   **Output:** Frame-by-frame object detection with class-wise counts

## 👤 Author

**Tejas Singhal**
Master’s Student, Computer Science @ Texas A&M University
