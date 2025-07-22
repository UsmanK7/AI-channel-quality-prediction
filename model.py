"""
Deep Learning Model Architectures for Wireless Channel Quality Prediction

This module contains the neural network architectures used for:
1. CQI (Channel Quality Indicator) classification
2. SNR (Signal-to-Noise Ratio) regression

Both MLP and CNN architectures are implemented with modern deep learning practices.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def create_mlp_model(input_dim, output_dim=1, task_type='regression'):
    """
    Create a Multi-Layer Perceptron (MLP) model for wireless channel prediction
    
    This architecture is designed for tabular wireless network data with:
    - Multiple hidden layers with decreasing neurons
    - Batch normalization for training stability
    - Dropout for regularization
    - Residual connections for better gradient flow
    
    Args:
        input_dim (int): Number of input features
        output_dim (int): Number of output units
        task_type (str): 'regression' or 'classification'
    
    Returns:
        keras.Model: Compiled TensorFlow model
    """
    
    # Input layer
    inputs = keras.Input(shape=(input_dim,), name='wireless_features')
    
    # First hidden block with residual connection
    x = layers.Dense(256, activation='relu', name='dense_1')(inputs)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.Dropout(0.3, name='dropout_1')(x)
    
    # Second hidden block
    x = layers.Dense(128, activation='relu', name='dense_2')(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.Dropout(0.3, name='dropout_2')(x)
    
    # Third hidden block with skip connection
    residual = layers.Dense(64, activation='linear', name='residual_projection')(x)
    x = layers.Dense(64, activation='relu', name='dense_3')(x)
    x = layers.Add(name='skip_connection')([x, residual])
    x = layers.BatchNormalization(name='bn_3')(x)
    x = layers.Dropout(0.2, name='dropout_3')(x)
    
    # Fourth hidden block
    x = layers.Dense(32, activation='relu', name='dense_4')(x)
    x = layers.BatchNormalization(name='bn_4')(x)
    x = layers.Dropout(0.2, name='dropout_4')(x)
    
    # Output layer based on task type
    if task_type == 'classification':
        outputs = layers.Dense(output_dim, activation='softmax', name='cqi_output')(x)
        loss = 'sparse_categorical_crossentropy'
        metrics = ['accuracy']
    else:  # regression
        outputs = layers.Dense(output_dim, activation='linear', name='snr_output')(x)
        loss = 'huber'  # More robust than MSE for outliers
        metrics = ['mae']
    
    # Create and compile model
    model = keras.Model(inputs=inputs, outputs=outputs, name=f'MLP_{task_type}')
    
    # Use adaptive learning rate optimizer
    optimizer = keras.optimizers.AdamW(
        learning_rate=0.001,
        weight_decay=1e-4,
        beta_1=0.9,
        beta_2=0.999
    )
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    return model

def create_cnn_model(input_shape, output_dim=1, task_type='regression'):
    """
    Create a 1D Convolutional Neural Network for wireless feature extraction
    
    This architecture treats the feature vector as a 1D signal and applies:
    - Multiple 1D convolutional layers for feature extraction
    - Global pooling to handle variable-length sequences
    - Dense layers for final prediction
    
    Particularly effective for capturing local patterns in wireless measurements.
    
    Args:
        input_shape (tuple): Shape of input (features, channels)
        output_dim (int): Number of output units
        task_type (str): 'regression' or 'classification'
    
    Returns:
        keras.Model: Compiled TensorFlow model
    """
    
    # Input layer
    inputs = keras.Input(shape=input_shape, name='wireless_signal')
    
    # First convolutional block
    x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same', name='conv1d_1')(inputs)
    x = layers.BatchNormalization(name='bn_conv_1')(x)
    x = layers.Dropout(0.25, name='dropout_conv_1')(x)
    
    # Second convolutional block
    x = layers.Conv1D(32, kernel_size=3, activation='relu', padding='same', name='conv1d_2')(x)
    x = layers.BatchNormalization(name='bn_conv_2')(x)
    x = layers.Dropout(0.25, name='dropout_conv_2')(x)
    
    # Third convolutional block with dilation for larger receptive field
    x = layers.Conv1D(16, kernel_size=3, dilation_rate=2, activation='relu', 
                     padding='same', name='conv1d_3')(x)
    x = layers.BatchNormalization(name='bn_conv_3')(x)
    
    # Global pooling to reduce dimensionality
    x_max = layers.GlobalMaxPooling1D(name='global_max_pool')(x)
    x_avg = layers.GlobalAveragePooling1D(name='global_avg_pool')(x)
    x = layers.Concatenate(name='pooling_concat')([x_max, x_avg])
    
    # Dense layers for final processing
    x = layers.Dense(64, activation='relu', name='dense_cnn_1')(x)
    x = layers.BatchNormalization(name='bn_dense_1')(x)
    x = layers.Dropout(0.3, name='dropout_dense_1')(x)
    
    x = layers.Dense(32, activation='relu', name='dense_cnn_2')(x)
    x = layers.BatchNormalization(name='bn_dense_2')(x)
    x = layers.Dropout(0.2, name='dropout_dense_2')(x)
    
    # Output layer based on task type
    if task_type == 'classification':
        outputs = layers.Dense(output_dim, activation='softmax', name='cqi_cnn_output')(x)
        loss = 'sparse_categorical_crossentropy'
        metrics = ['accuracy']
    else:  # regression
        outputs = layers.Dense(output_dim, activation='linear', name='snr_cnn_output')(x)
        loss = 'huber'
        metrics = ['mae']
    
    # Create and compile model
    model = keras.Model(inputs=inputs, outputs=outputs, name=f'CNN_1D_{task_type}')
    
    # Use adaptive learning rate optimizer with gradient clipping
    optimizer = keras.optimizers.AdamW(
        learning_rate=0.001,
        weight_decay=1e-4,
        beta_1=0.9,
        beta_2=0.999,
        clipnorm=1.0  # Gradient clipping for stability
    )
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    return model

def create_transformer_model(input_dim, output_dim=1, task_type='regression'):
    """
    Create a Transformer-based model for wireless channel prediction
    
    This experimental architecture uses self-attention mechanisms to:
    - Capture long-range dependencies in wireless features
    - Learn complex feature interactions
    - Provide interpretable attention weights
    
    Note: This is an advanced architecture for research purposes.
    
    Args:
        input_dim (int): Number of input features
        output_dim (int): Number of output units
        task_type (str): 'regression' or 'classification'
    
    Returns:
        keras.Model: Compiled TensorFlow model
    """
    
    # Input and embedding
    inputs = keras.Input(shape=(input_dim,), name='wireless_features')
    
    # Reshape for transformer (add sequence dimension)
    x = layers.Reshape((input_dim, 1))(inputs)
    x = layers.Dense(64, name='feature_embedding')(x)  # Feature embedding
    
    # Multi-head self-attention
    attention_output = layers.MultiHeadAttention(
        num_heads=4, 
        key_dim=16,
        name='multi_head_attention'
    )(x, x)
    
    # Add & norm
    x = layers.Add(name='attention_residual')([x, attention_output])
    x = layers.LayerNormalization(name='attention_norm')(x)
    
    # Feed-forward network
    ffn = layers.Dense(128, activation='relu', name='ffn_1')(x)
    ffn = layers.Dense(64, name='ffn_2')(ffn)
    
    # Add & norm
    x = layers.Add(name='ffn_residual')([x, ffn])
    x = layers.LayerNormalization(name='ffn_norm')(x)
    
    # Global pooling and output
    x = layers.GlobalAveragePooling1D(name='global_pool')(x)
    x = layers.Dropout(0.3, name='output_dropout')(x)
    
    # Output layer
    if task_type == 'classification':
        outputs = layers.Dense(output_dim, activation='softmax', name='transformer_cqi_output')(x)
        loss = 'sparse_categorical_crossentropy'
        metrics = ['accuracy']
    else:  # regression
        outputs = layers.Dense(output_dim, activation='linear', name='transformer_snr_output')(x)
        loss = 'huber'
        metrics = ['mae']
    
    model = keras.Model(inputs=inputs, outputs=outputs, name=f'Transformer_{task_type}')
    
    optimizer = keras.optimizers.AdamW(learning_rate=0.0005, weight_decay=1e-4)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    return model

def create_ensemble_model(input_dim, output_dim=1, task_type='regression'):
    """
    Create an ensemble model combining MLP and CNN predictions
    
    This meta-architecture combines the strengths of both:
    - MLP for direct feature processing
    - CNN for pattern recognition
    - Weighted combination of predictions
    
    Args:
        input_dim (int): Number of input features
        output_dim (int): Number of output units
        task_type (str): 'regression' or 'classification'
    
    Returns:
        keras.Model: Compiled ensemble model
    """
    
    # Shared input
    inputs = keras.Input(shape=(input_dim,), name='wireless_features')
    
    # MLP branch
    mlp_x = layers.Dense(128, activation='relu', name='mlp_dense_1')(inputs)
    mlp_x = layers.BatchNormalization(name='mlp_bn_1')(mlp_x)
    mlp_x = layers.Dropout(0.3, name='mlp_dropout_1')(mlp_x)
    mlp_x = layers.Dense(64, activation='relu', name='mlp_dense_2')(mlp_x)
    mlp_output = layers.Dense(32, activation='relu', name='mlp_output')(mlp_x)
    
    # CNN branch
    cnn_inputs = layers.Reshape((input_dim, 1))(inputs)
    cnn_x = layers.Conv1D(32, 3, activation='relu', padding='same', name='cnn_conv_1')(cnn_inputs)
    cnn_x = layers.GlobalAveragePooling1D(name='cnn_pool')(cnn_x)
    cnn_x = layers.Dense(64, activation='relu', name='cnn_dense_1')(cnn_x)
    cnn_output = layers.Dense(32, activation='relu', name='cnn_output')(cnn_x)
    
    # Combine branches
    combined = layers.Concatenate(name='ensemble_concat')([mlp_output, cnn_output])
    combined = layers.Dense(32, activation='relu', name='ensemble_dense')(combined)
    combined = layers.Dropout(0.2, name='ensemble_dropout')(combined)
    
    # Output layer
    if task_type == 'classification':
        outputs = layers.Dense(output_dim, activation='softmax', name='ensemble_cqi_output')(combined)
        loss = 'sparse_categorical_crossentropy'
        metrics = ['accuracy']
    else:  # regression
        outputs = layers.Dense(output_dim, activation='linear', name='ensemble_snr_output')(combined)
        loss = 'huber'
        metrics = ['mae']
    
    model = keras.Model(inputs=inputs, outputs=outputs, name=f'Ensemble_{task_type}')
    
    optimizer = keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    return model