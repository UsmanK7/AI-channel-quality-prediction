# AI Channel Quality Prediction for 5G and Beyond Wireless Networks

## 📸 Project Gallery

| CQI Classification Training | CQI Predictions | SNR Predictions | SNR Training History |
|---|---|---|---|
| ![CQI Training History](https://github.com/user-attachments/assets/64a6dac4-8fd8-4089-a96f-6c04f5da79f5) | ![CQI Predictions](https://github.com/user-attachments/assets/6e3e5404-c490-48d0-a122-5bf0fee76c86) | ![SNR Predictions](https://github.com/user-attachments/assets/21d4d25f-8533-4c22-8730-21b276d65d19) | ![SNR Training History](https://github.com/user-attachments/assets/a5363c51-922d-4931-94f5-c2fb4e841f1f) |
| Model convergence and validation accuracy over epochs | Actual vs predicted Channel Quality Indicator values | Signal-to-Noise Ratio regression predictions | Loss function optimization and model learning curves |

## 📋 Project Overview

This project implements advanced deep learning techniques for predicting wireless channel quality in 5G and Beyond (B5G) networks. The system focuses on two critical prediction tasks:

- ✅ **CQI Classification:** Predicting Channel Quality Indicator (1-15 scale)
- ✅ **SNR Regression:** Estimating Signal-to-Noise Ratio in dB

## ✅ Key Features

### 📡 5G/B5G Channel Modeling
Realistic wireless channel simulation using 3GPP standards and advanced propagation models

### 🧠 Multi-Architecture AI
MLP, CNN, Transformer, and Ensemble models for comprehensive prediction capabilities

### 📊 Advanced Analytics
Comprehensive visualization tools and performance analysis with production-ready code

### 👥 Dynamic Scheduling
Real-time user scheduling demonstration and network optimization capabilities

## 🛠 Technical Specifications

| Specification | Details |
|---|---|
| **Programming Language** | Python 3.8+ |
| **Deep Learning Framework** | TensorFlow/Keras 2.x |
| **Dataset Size** | 15,000 synthetic samples |
| **Features** | 8 wireless network parameters |
| **Models** | 4 different architectures |
| **Evaluation Metrics** | Accuracy, MAE, MSE, R² |

## 🏗 System Architecture

<img width="2840" height="580" alt="Screenshot 2025-07-23 151135" src="https://github.com/user-attachments/assets/cb50db19-9a07-4176-987a-40d859ce1743" />


### Project Structure

```
project/
├── main.py                 # Main execution script
├── model.py               # Neural network architectures
├── utils.py               # Data generation & visualization
├── dataset.csv            # Generated synthetic dataset
└── results/               # Output directory
    ├── models/            # Trained model files
    ├── plots/             # Visualization outputs
    └── metrics/           # Performance results
```

## 📊 Dataset Generation

### Wireless Channel Modeling

Implementation of state-of-the-art wireless propagation models based on 3GPP TR 38.901 standards:

#### Path Loss Models:

**Urban Micro (UMi) LOS:**
```
PL = 32.4 + 21×log₁₀(d) + 20×log₁₀(fc) [d ≤ 1000m]
PL = 32.4 + 40×log₁₀(d) + 20×log₁₀(fc) - 9.5 [d > 1000m]
```

**Urban Micro (UMi) NLOS:**
```
PL = 35.3×log₁₀(d) + 22.4 + 21.3×log₁₀(fc) - 0.3×h_UT
```

### Feature Engineering

| Feature | Range | Distribution |
|---|---|---|
| User Speed | 0-300 km/h | Gamma/Normal |
| Distance to BS | 10-5000 m | Exponential |
| Frequency | 0.7-60 GHz | Discrete |
| LOS/NLOS | Categorical | Bernoulli |
| Power Level | 20-50 dBm | Normal |
| Interference | -105 to -70 dB | Function |

### Dataset Statistics

| Metric | Value |
|---|---|
| **Total Samples** | 15,000 |
| **Input Features** | 8 |
| **Train/Test Split** | 80/20 |

## 🤖 Model Architectures

### 1. Multi-Layer Perceptron (MLP)
**Purpose:** CQI Classification
<img width="785" height="1117" alt="1" src="https://github.com/user-attachments/assets/6c3d03a3-c030-4fff-a681-5f42a592ce9f" />


**Features:**
- Residual Connections
- Batch Normalization
- AdamW Optimizer

### 2. 1D Convolutional Neural Network
**Purpose:** SNR Regression
<img width="602" height="1110" alt="2" src="https://github.com/user-attachments/assets/e585828c-736f-409c-82bf-6fd4ce6c99c0" />


**Features:**
- Dilated Convolutions
- Global Pooling
- Huber Loss

### 3. Transformer Model
**Purpose:** Advanced Architecture
<img width="487" height="1077" alt="3" src="https://github.com/user-attachments/assets/fd0eae2d-1647-4bae-87f8-d1900c341356" />


**Features:**
- Self-Attention
- Layer Normalization
- Positional Encoding

### 4. Ensemble Model
**Purpose:** Combined Approach
<img width="632" height="985" alt="4" src="https://github.com/user-attachments/assets/186e116e-7916-4314-8cf6-c23105fa2008" />


**Features:**
- Dual-Branch
- Feature Fusion
- Weighted Outputs

## 🚀 Getting Started

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the main script: `python main.py`
4. Check results in the `results/` directory

## 📈 Performance Metrics

The models are evaluated using comprehensive metrics including accuracy for classification tasks and MAE, MSE, R² for regression tasks.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
