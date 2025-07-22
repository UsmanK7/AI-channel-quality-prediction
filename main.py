"""
AI-Powered Channel Quality Prediction for 5G and Beyond Wireless Networks
Main execution script for the complete wireless communications deep learning project

Author: AI Research Assistant
Date: July 2025
Purpose: Scholarship application for wireless communications research lab
Focus: B5G/6G, RIS, multiuser MIMO, and AI for signal processing
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from utils import WirelessDataGenerator, create_results_directory, plot_training_history, plot_predictions
from model import create_mlp_model, create_cnn_model

def main():
    """
    Main function to execute the complete AI-powered channel quality prediction pipeline
    """
    print("=" * 80)
    print("AI-POWERED CHANNEL QUALITY PREDICTION FOR 5G AND BEYOND WIRELESS NETWORKS")
    print("=" * 80)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Create results directory
    create_results_directory()
    
    # Step 1: Generate synthetic wireless dataset
    print("\n🔧 STEP 1: Generating Synthetic Wireless Dataset")
    print("-" * 50)
    
    data_generator = WirelessDataGenerator()
    
    # Check if dataset already exists
    if os.path.exists('dataset.csv'):
        print("📁 Loading existing dataset...")
        df = pd.read_csv('dataset.csv')
    else:
        print("🏗️ Generating new synthetic dataset...")
        df = data_generator.generate_dataset(n_samples=15000)
        df.to_csv('dataset.csv', index=False)
        print(f"✅ Dataset generated and saved: {df.shape[0]} samples, {df.shape[1]} features")
    
    # Display dataset info
    print(f"\n📊 Dataset Overview:")
    print(f"   • Total samples: {len(df)}")
    print(f"   • Features: {list(df.columns[:-2])}")  # Exclude CQI and SNR targets
    print(f"   • Targets: CQI (classification), SNR (regression)")
    print(f"\n📈 Dataset Statistics:")
    print(df.describe())
    
    # Step 2: Data preprocessing
    print("\n🔧 STEP 2: Data Preprocessing")
    print("-" * 50)
    
    # Separate features and targets
    feature_columns = ['user_speed_kmh', 'distance_to_bs_m', 'frequency_ghz', 
                      'los_nlos', 'power_level_dbm', 'num_users', 
                      'environment_type', 'interference_level_db']
    
    X = df[feature_columns].copy()
    y_cqi = df['cqi'].values  # Classification target
    y_snr = df['snr_db'].values  # Regression target
    
    # Encode categorical variables
    le_los = LabelEncoder()
    le_env = LabelEncoder()
    
    X['los_nlos_encoded'] = le_los.fit_transform(X['los_nlos'])
    X['environment_encoded'] = le_env.fit_transform(X['environment_type'])
    
    # Select final feature set
    feature_set = ['user_speed_kmh', 'distance_to_bs_m', 'frequency_ghz', 
                   'los_nlos_encoded', 'power_level_dbm', 'num_users', 
                   'environment_encoded', 'interference_level_db']
    
    X_final = X[feature_set].values
    
    # Split data for both tasks
    X_train, X_test, y_cqi_train, y_cqi_test = train_test_split(
        X_final, y_cqi, test_size=0.2, random_state=42, stratify=y_cqi
    )
    
    _, _, y_snr_train, y_snr_test = train_test_split(
        X_final, y_snr, test_size=0.2, random_state=42, stratify=y_cqi
    )
    
    # Feature normalization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✅ Data preprocessing completed:")
    print(f"   • Training samples: {X_train_scaled.shape[0]}")
    print(f"   • Testing samples: {X_test_scaled.shape[0]}")
    print(f"   • Features: {X_train_scaled.shape[1]}")
    print(f"   • Feature scaling: Applied StandardScaler")
    
    # Step 3: Model Training - CQI Classification
    print("\n🤖 STEP 3: Training CQI Classification Model (MLP)")
    print("-" * 50)
    
    # Create and compile CQI model
    cqi_model = create_mlp_model(
        input_dim=X_train_scaled.shape[1],
        output_dim=15,  # CQI values 1-15
        task_type='classification'
    )
    
    # Train CQI model
    cqi_history = cqi_model.fit(
        X_train_scaled, y_cqi_train - 1,  # Convert to 0-14 for softmax
        validation_split=0.2,
        epochs=100,
        batch_size=64,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
        ],
        verbose=1
    )
    
    # Evaluate CQI model
    cqi_pred_probs = cqi_model.predict(X_test_scaled, verbose=0)
    cqi_pred = np.argmax(cqi_pred_probs, axis=1) + 1  # Convert back to 1-15
    
    cqi_accuracy = np.mean(cqi_pred == y_cqi_test)
    cqi_mae = mean_absolute_error(y_cqi_test, cqi_pred)
    
    print(f"✅ CQI Classification Results:")
    print(f"   • Accuracy: {cqi_accuracy:.4f}")
    print(f"   • MAE: {cqi_mae:.4f}")
    
    # Step 4: Model Training - SNR Regression
    print("\n🤖 STEP 4: Training SNR Regression Model (1D CNN)")
    print("-" * 50)
    
    # Reshape data for CNN (add channel dimension)
    X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
    
    # Create and compile SNR model
    snr_model = create_cnn_model(
        input_shape=(X_train_scaled.shape[1], 1),
        task_type='regression'
    )
    
    # Train SNR model
    snr_history = snr_model.fit(
        X_train_cnn, y_snr_train,
        validation_split=0.2,
        epochs=100,
        batch_size=64,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
        ],
        verbose=1
    )
    
    # Evaluate SNR model
    snr_pred = snr_model.predict(X_test_cnn, verbose=0).flatten()
    
    snr_mae = mean_absolute_error(y_snr_test, snr_pred)
    snr_mse = mean_squared_error(y_snr_test, snr_pred)
    snr_r2 = r2_score(y_snr_test, snr_pred)
    
    print(f"✅ SNR Regression Results:")
    print(f"   • MAE: {snr_mae:.4f} dB")
    print(f"   • MSE: {snr_mse:.4f} dB²")
    print(f"   • R² Score: {snr_r2:.4f}")
    
    # Step 5: Visualization and Results
    print("\n📊 STEP 5: Generating Visualizations")
    print("-" * 50)
    
    # Plot training history
    plot_training_history(cqi_history, 'CQI Classification', 'accuracy')
    plot_training_history(snr_history, 'SNR Regression', 'mae')
    
    # Plot predictions
    plot_predictions(y_cqi_test, cqi_pred, 'CQI', task_type='classification')
    plot_predictions(y_snr_test, snr_pred, 'SNR (dB)', task_type='regression')
    
    # Feature importance analysis (using model weights as proxy)
    plt.figure(figsize=(12, 6))
    feature_names = ['Speed', 'Distance', 'Frequency', 'LOS/NLOS', 
                    'Power', 'Users', 'Environment', 'Interference']
    
    # Get first layer weights from SNR model
    first_layer_weights = snr_model.layers[0].get_weights()[0]
    feature_importance = np.abs(first_layer_weights).mean(axis=1)
    
    plt.subplot(1, 2, 1)
    plt.barh(feature_names, feature_importance)
    plt.xlabel('Average Absolute Weight')
    plt.title('Feature Importance (SNR Model)')
    plt.tight_layout()
    
    # Correlation matrix
    plt.subplot(1, 2, 2)
    corr_matrix = df[feature_columns[:-2] + ['snr_db', 'cqi']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                xticklabels=['Speed', 'Distance', 'Freq', 'LOS', 'Power', 'Users', 'Env', 'Interf', 'SNR', 'CQI'],
                yticklabels=['Speed', 'Distance', 'Freq', 'LOS', 'Power', 'Users', 'Env', 'Interf', 'SNR', 'CQI'])
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    
    plt.savefig('results/feature_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Step 6: Save Models
    print("\n💾 STEP 6: Saving Trained Models")
    print("-" * 50)
    
    cqi_model.save('results/cqi_classification_model.h5')
    snr_model.save('results/snr_regression_model.h5')
    
    # Save preprocessing objects
    import pickle
    with open('results/feature_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('results/label_encoders.pkl', 'wb') as f:
        pickle.dump({'los_nlos': le_los, 'environment': le_env}, f)
    
    print("✅ Models and preprocessors saved to results/ directory")
    
    # Step 7: Bonus - Dynamic User Scheduling Demo
    print("\n🎯 STEP 7: Bonus - Dynamic User Scheduling Demonstration")
    print("-" * 50)
    
    # Simulate a simple multi-user scenario
    demo_users = data_generator.generate_dataset(n_samples=10)
    demo_features = demo_users[feature_set].copy()
    demo_features['los_nlos_encoded'] = le_los.transform(demo_features['los_nlos'])
    demo_features['environment_encoded'] = le_env.transform(demo_features['environment_type'])
    demo_X = scaler.transform(demo_features[feature_set].values)
    demo_X_cnn = demo_X.reshape(demo_X.shape[0], demo_X.shape[1], 1)
    
    # Predict SNR for all users
    predicted_snr = snr_model.predict(demo_X_cnn, verbose=0).flatten()
    
    # Simple scheduling: prioritize users with highest predicted SNR
    user_priorities = np.argsort(predicted_snr)[::-1]  # Descending order
    
    print("📱 Multi-User Scheduling Results:")
    print("   User ID | Predicted SNR (dB) | Priority Rank")
    print("   --------|--------------------|--------------")
    for i, user_idx in enumerate(user_priorities):
        print(f"   User {user_idx:2d}  | {predicted_snr[user_idx]:15.2f} | Rank {i+1:2d}")
    
    print(f"\n🏆 Recommended scheduling order: Users {user_priorities[:5]}")
    
    # Final Summary
    print("\n" + "=" * 80)
    print("🎉 PROJECT COMPLETION SUMMARY")
    print("=" * 80)
    print(f"✅ Dataset: {len(df)} samples with 8 wireless features")
    print(f"✅ CQI Classification: {cqi_accuracy:.1%} accuracy, {cqi_mae:.2f} MAE")
    print(f"✅ SNR Regression: {snr_r2:.3f} R², {snr_mae:.2f} dB MAE")
    print(f"✅ Models saved: cqi_classification_model.h5, snr_regression_model.h5")
    print(f"✅ Visualizations: Training curves, prediction plots, feature analysis")
    print(f"✅ Bonus: Dynamic user scheduling demonstration completed")
    print("\n🎓 This project demonstrates:")
    print("   • Advanced ML/DL techniques for wireless communications")
    print("   • Realistic 5G/B5G channel modeling and prediction")
    print("   • Professional code structure and documentation")
    print("   • Research-ready implementation for academic applications")
    print("\n📂 All results saved to: ./results/ directory")
    print("📄 Dataset available at: ./dataset.csv")
    
if __name__ == "__main__":
    main()