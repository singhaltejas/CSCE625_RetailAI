# YOLONAS Retail Store Tracker 🛒

This project leverages **YOLONAS** (a YOLO-based neural architecture search model) for real-time object detection and tracking in a retail store environment. It uses cart-mounted video footage to detect and count items added to the cart, assisting in inventory management and automation in smart retail.

## 📌 Features

- 🎯 Uses YOLONAS (YOLO NAS) for accurate object detection  
- 📹 Processes pre-recorded retail store videos (e.g., cart footage)  
- 🧠 Real-time item classification and tracking  
- 📊 Live item counting and annotation overlay  
- 🔧 Easily customizable for different product classes  

## 🛠️ Tech Stack

- Python  
- [Supervision](https://github.com/roboflow/supervision) (for annotation/visualization)  
- [YOLO-NAS](https://github.com/Deci-AI/super-gradients) via SuperGradients  
- OpenCV  
- NumPy  

## 📁 Folder Structure

.
├── YOLONAS_Retail_Store.ipynb # Main notebook for detection & counting
├── assets/ # (optional) Contains sample input/output media
└── README.md # Project overview and setup


## 🧪 Usage

1.  **Clone the repo and install the requirements:**

    ```
    git clone https://github.com/<your-username>/YOLONAS_Retail_Store.git
    cd YOLONAS_Retail_Store
    pip install -r requirements.txt
    ```

2.  **Open the notebook:**

    ```
    jupyter notebook YOLONAS_Retail_Store.ipynb
    ```

3.  **Configure the following in the notebook:**
    *   Path to input video
    *   Detection class filters (e.g., bottles, snacks)

4.  **Run all cells** to perform object detection and see visual tracking.

## 🖼️ Sample Output

Real-time detection with bounding boxes and item count.

<!-- You might want to add an actual image or GIF here -->
<!-- Example: ![Sample Output](assets/output_example.gif) -->

## 🔍 Model Details

*   **Model:** YOLO-NAS (S variant used)
*   **Backend:** SuperGradients
*   **Input:** Retail cart camera video
*   **Output:** Frame-by-frame object detection with class-wise counts

## 👤 Author

**Tejas Singhal**
Master’s Student, Computer Science @ Texas A&M University
