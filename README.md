# CSCE625_RetailAI

# 🛍️ Smart Retail AI Suite

An AI-powered solution to revolutionize the retail experience through computer vision and real-time analytics. This project is composed of three integrated modules that work together to track customer behavior, manage inventory, and automate checkout processes.

---

## 📦 Project Modules

### 1. 🧍 Person Detection
- Detects customers entering and exiting the store.
- Tracks in-store movement to generate heatmaps of high-traffic areas.
- Enables real-time occupancy analytics and customer behavior insights.

### 2. 🛒 Cart Detection
- Uses a **YOLONAS** model to detect items being added to shopping carts.
- Maintains a real-time digital cart for each customer.
- Supports seamless checkout and loss prevention.

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

> ⚠️ Each module is currently developed as a standalone project. Follow individual module instructions to run them.

```bash
# Clone the repo
git clone https://github.com/singhaltejas/CSCE625_RetailAI.git
cd CSCE625_RetailAI
```

Each module has its own README:


- [📦 Cart Detection](./Cart%20Detection/README.md)
- [🧾 Shelf Monitoring](./Shelf%20Monitoring/README.md)
- [🧍 Person Detection](./person-detection/README.md)


## 🤝 Contributors
- Tejas Singhal [🔗 LinkedIn](https://www.linkedin.com/in/tejas-singhal/)
- Kunal Somendra Singh [🔗 LinkedIn](https://www.linkedin.com/in/kunal-s-singh/)
- Hitarth Chopra [🔗 LinkedIn](https://www.linkedin.com/in/hitarth-chopra-27b07a240/)

