# CSCE625_RetailAI

# 🛍️ Smart Retail AI Suite

An AI-powered solution to revolutionize the retail experience through computer vision and real-time analytics. This project is composed of three integrated modules that work together to track customer behavior, manage inventory, and automate checkout processes.

---

## 📦 Project Modules

### 1. 🧍 Person Detection
- Detects customers entering and exiting the store.
- Tracks in-store movement to generate heatmaps of high-traffic areas.
- Enables real-time occupancy analytics and customer behavior insights.

### 🛒 Cart Detection System

This project leverages a **YOLO-NAS** model to enable intelligent item detection and tracking in retail shopping environments.

- Leverages the YOLO-NAS model for high-performance, real-time detection of retail items from cart-mounted video footage.
- Tracks and counts detected items frame-by-frame using Supervision, providing clear overlays and summary stats.
- Easily configurable for specific product classes and video inputs, ideal for smart retail and automated inventory systems.



### 3. 🧾 Shelf Monitoring
- Uses a YOLO-based model to monitor stock levels on shelves.
- Detects low-stock or misplaced items.
- Helps automate restocking and inventory management.

---

## 🔁 End-to-End Use Case Scenario

### 👣 Step-by-Step Flow:

1. **Customer Enters the Store**
   - Person Detection logs their entry and starts tracking.
   
2. **In-Store Navigation**
   - Heatmaps generated from movement data.
   - Staff can respond dynamically based on crowd density.

3. **Shopping with Cart**
   - Cart Detection module logs each item added to cart.
   - Digital cart updated in real-time.

4. **Interaction with Shelves**
   - Shelf Monitoring activated as customers approach shelves.
   - Stock levels are adjusted accordingly.

5. **Customer Exits**
   - Person Detection confirms exit.
   - Final cart snapshot can be used for billing or analytics.

---

## 🚀 Getting Started

1. **Clone the repo and cd into it**
```bash
# Clone the repo
git clone https://github.com/singhaltejas/CSCE625_RetailAI.git
cd CSCE625_RetailAI
```

1. **For Cart Detection**
```bash
cd "Cart Detection"
```
Then run the YOLONAS_Retail_Store.ipynb cell by cell.

2. **For Shelf Monitoring**
```bash
cd "Shelf Monitoring"
```
Then install dependencies using :
```bash
pip install -r requirements.txt

```

If you need to train the model first: (Optional)

```bash
python train.py
```

To compare two shelf images and analyze product changes:

```bash
python main.py
```

3. **For Store Foot Analyzer**
```bash
cd "Store Foot Analyzer"
```

Update the path for the example: "Store (1)" as attached in the same folder in the main code. (If necessary) and then run the Hotzone_Mapper.ipynb cell by cell.

Each module has its own README for further details:


- [📦 Cart Detection](./Cart%20Detection/README.md)
- [🧾 Shelf Monitoring](./Shelf%20Monitoring/README.md)
- [🧍 Person Detection](./person-detection/README.md)


## 🤝 Contributors
- Tejas Singhal [🔗 LinkedIn](https://www.linkedin.com/in/tejas-singhal/)
- Kunal Somendra Singh [🔗 LinkedIn](https://www.linkedin.com/in/kunal-s-singh/)
- Hitarth Chopra [🔗 LinkedIn](https://www.linkedin.com/in/hitarth-chopra-27b07a240/)

