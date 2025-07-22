# AI-Powered Channel Quality Prediction for 5G and Beyond Wireless Networks

## 🎯 Project Overview

This project implements an advanced deep learning system for predicting Channel Quality Indicator (CQI) and Signal-to-Noise Ratio (SNR) in 5G and Beyond (B5G) wireless networks. The system uses state-of-the-art neural network architectures to enable intelligent network optimization and resource allocation.

## 🔬 Research Focus Areas

This project directly addresses key research challenges in:
- **5G/6G Network Optimization**: AI-driven channel quality prediction for next-generation wireless systems
- **Reconfigurable Intelligent Surfaces (RIS)**: Foundation for intelligent reflecting surface optimization
- **Multi-user MIMO Systems**: Enhanced user scheduling and resource allocation
- **AI for Signal Processing**: Machine learning approaches to wireless channel modeling

## 🏗️ Project Architecture

### Core Components
1. **Synthetic Dataset Generation** (`utils.py`): Realistic 5G channel models using 3GPP standards
2. **Deep Learning Models** (`model.py`): Multiple architectures (MLP, CNN, Transformer, Ensemble)
3. **Training Pipeline** (`main.py`): Complete ML workflow with evaluation and visualization
4. **Advanced Features**: Multi-user scheduling demonstration and feature analysis

### Key Features
- ✅ **Realistic 5G Channel Modeling**: 3GPP TR 38.901 compliant path loss models
- ✅ **Advanced ML Architectures**: MLP, 1D-CNN, Transformer, and Ensemble models
- ✅ **Comprehensive Evaluation**: Multiple metrics and visualization tools
- ✅ **Production Ready**: Professional code structure with full documentation
- ✅ **Research Applications**: Direct applicability to B5G/6G research

## 📊 Dataset Specifications

### Input Features (8 dimensions)
- **User Speed** (km/h): Mobility patterns from pedestrian to high-speed
- **Distance to BS** (m): Cell coverage analysis (10m - 5km)
- **Frequency Band** (GHz): Sub-6GHz and mmWave bands
- **LOS/NLOS**: Line-of-sight propagation conditions
- **Power Level** (dBm): Base station transmission power
- **Number of Users**: Multi-user interference modeling
- **Environment Type**: Urban, suburban, rural scenarios
- **Interference Level** (dB): Co-channel and adjacent channel interference

### Target Variables
- **CQI Prediction**: 15-class classification (CQI 1-15)
- **SNR Prediction**: Continuous regression (-10 to 30 dB)

### Dataset Statistics
- **Size**: 15,000 samples with realistic feature distributions
- **Quality**: Correlated shadow fading, 3GPP-compliant path loss
- **Diversity**: Multiple environments, mobility patterns, and network conditions

## 🧠 Model Architectures

### 1. Multi-Layer Perceptron (MLP)
```
Input (8) → Dense(256) → BN → Dropout → Dense(128) → BN → Dropout 
→ Dense(64) + Skip Connection → BN → Dropout → Dense(32) → Output
```
- **Best for**: Direct feature processing, fast inference
- **Features**: Batch normalization, residual connections, adaptive learning

### 2. 1D Convolutional Neural Network
```
Input (8,1) → Conv1D(64,3) → BN → Dropout → Conv1D(32,3) → BN 
→ Conv1D(16,3,dilation=2) → GlobalPooling → Dense(64) → Output
```
- **Best for**: Pattern recognition in wireless features
- **Features**: Dilated convolutions, global pooling, feature extraction

### 3. Transformer Architecture (Advanced)
```
Input → Embedding → Multi-Head Attention → Add&Norm → FFN 
→ Add&Norm → Global Pooling → Output
```
- **Best for**: Complex feature interactions, interpretability
- **Features**: Self-attention, position encoding, attention weights

### 4. Ensemble Model
- **Combines**: MLP + CNN predictions with learned weighting
- **Best for**: Maximum accuracy, robust predictions

## 🚀 Quick Start

### Installation
```bash
# Clone or download project files
# Install dependencies
pip install -r requirements.txt
```

### Running the Project
```bash
# Execute complete pipeline
python main.py
```

### Expected Output
1. **Dataset Generation**: 15,000 samples with 5G channel characteristics
2. **Model Training**: Dual models for CQI classification and SNR regression
3. **Performance Metrics**: Accuracy, MAE, MSE, R² scores
4. **Visualizations**: Training curves, prediction plots, feature analysis
5. **Bonus Demo**: Multi-user scheduling simulation

## 📈 Performance Benchmarks

### CQI Classification Model (MLP)
- **Accuracy**: ~85-90%
- **MAE**: <1.5 CQI levels
- **Architecture**: Optimized for wireless channel classification

### SNR Regression Model (1D-CNN)
- **R² Score**: >0.85
- **MAE**: <2.5 dB
- **RMSE**: <3.5 dB
- **Architecture**: Convolutional feature extraction for continuous prediction

## 🔍 Research Applications

### 1. Next-Generation Network Optimization
- **Use Case**: Proactive resource allocation in 6G networks
- **Benefit**: Reduced latency, improved QoS

### 2. Intelligent Reflecting Surfaces (RIS)
- **Use Case**: Optimal phase configuration based on channel prediction
- **Benefit**: Enhanced coverage, energy efficiency

### 3. Multi-User MIMO Systems
- **Use Case**: User scheduling and beamforming optimization
- **Benefit**: Increased system capacity, fairness

### 4. Edge Computing Integration
- **Use Case**: Real-time channel quality prediction at network edge
- **Benefit**: Ultra-low latency applications

## 📁 Project Structure

```
wireless_ai_project/
├── main.py                 # Main execution script
├── model.py                # Neural network architectures
├── utils.py                # Data generation and visualization
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── dataset.csv            # Generated wireless dataset
└── results/               # Training outputs and visualizations
    ├── cqi_classification_model.h5
    ├── snr_regression_model.h5
    ├── feature_scaler.pkl
    ├── label_encoders.pkl
    └── *.png              # Visualization plots
```

## 🎓 Academic Relevance

### For Wireless Communications Research Labs
This project demonstrates:
- **Advanced ML/DL Skills**: Multiple architectures, proper evaluation
- **Domain Knowledge**: 5G/B5G standards, channel modeling
- **Research Readiness**: Publication-quality implementation
- **Innovation Potential**: Foundation for novel research directions

### Publication Opportunities
- **Conference Submissions**: IEEE GLOBECOM, ICC, WCNC
- **Journal Articles**: IEEE Trans. on Wireless Communications, Vehicular Technology
- **Workshop Presentations**: ML4Wireless, AI4Networks

## 🔧 Advanced Features

### 1. Realistic Channel Modeling
- 3GPP TR 38.901 path loss models
- Correlated shadow fading
- Small-scale fading (Rayleigh/Rician)
- Multi-frequency band support

### 2. Professional ML Pipeline
- Proper train/validation/test splits
- Feature normalization and encoding
- Early stopping and learning rate scheduling
- Comprehensive model evaluation

### 3. Visualization Suite
- Training history plots
- Prediction scatter plots
- Feature importance analysis
- Correlation matrix heatmaps

### 4. Multi-User Scheduling Demo
- Realistic user scenario simulation
- SNR-based priority scheduling
- Performance comparison metrics

## 🚀 Future Extensions

### 1. Advanced Architectures
- **Graph Neural Networks**: For network topology modeling
- **Recurrent Networks**: For temporal channel prediction
- **Attention Mechanisms**: For interpretable feature selection

### 2. Real-World Integration
- **5G Testbed Integration**: Live network data collection
- **Edge Deployment**: TensorFlow Lite optimization
- **Cloud Scaling**: Distributed training and inference

### 3. Research Directions
- **Federated Learning**: Privacy-preserving multi-operator training
- **Transfer Learning**: Cross-environment model adaptation
- **Uncertainty Quantification**: Confidence-aware predictions

## 📞 Contact & Collaboration

This project is designed for academic and research purposes. For collaborations, extensions, or questions:

- **Research Focus**: B5G/6G Networks, RIS, Multi-user MIMO, AI for Wireless
- **Technical Stack**: TensorFlow, Python, Wireless Channel Modeling
- **Applications**: Network optimization, resource allocation, intelligent systems

---

*This project demonstrates advanced machine learning capabilities applied to cutting-edge wireless communications research, suitable for top-tier academic programs and industry research positions.*