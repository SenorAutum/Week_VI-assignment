# Week_VI-assignment
This my repository for week 6 assignment in AI for Software Engineering
# Edge AI & IoT: Recyclable Classification & Smart Agriculture

**Assignment:** Emerging AI Trends & Implementation  
**Author:** Brandon Mwanzia 
**Status:** Prototype Complete

## 📂 Project Structure
1. **Edge AI Prototype:** A TensorFlow Lite model designed to run on low-power devices to classify recyclable materials.
2. **Smart Agriculture Concept:** A theoretical architecture for AI-driven crop management.

## 🚀 Part 1: Edge AI Prototype
### How to Run
1. Install requirements: `pip install tensorflow numpy matplotlib`
2. Run the script: `python edge_ai_recycling.py`
3. The script will generate a `recycling_model_quantized.tflite` file.

### Technical Details
* **Framework:** TensorFlow / Keras
* **Optimization:** Post-training quantization (Default) to reduce model size for Raspberry Pi deployment.
* **Accuracy:** Reached ~[Insert Accuracy from your run]% on the validation set.

## 🌾 Part 2: Smart Agriculture IoT
### Concept
We propose an LSTM-based system using NPK and Soil Moisture sensors. The system uses a feedback loop where the AI predicts yield drops and automatically triggers irrigation valves via LoRaWAN.

### Data Flow
Sensors -> Edge Gateway -> Cloud (AI Inference) -> User Dashboard -> Actuators (Sprinklers).

<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/b7767c6d-e573-45b5-afcb-66942532065e" />

