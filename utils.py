"""
Utility Functions for Wireless Channel Quality Prediction

This module contains:
1. Synthetic wireless dataset generation using realistic channel models
2. Visualization functions for training and results
3. Helper functions for data processing and evaluation

The wireless channel models implement:
- Log-distance path loss
- Rayleigh/Rician fading
- Shadow fading effects
- Interference modeling
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class WirelessDataGenerator:
    """
    Advanced wireless dataset generator using realistic 5G/B5G channel models
    
    This class implements state-of-the-art wireless propagation models including:
    - 3GPP TR 38.901 path loss models
    - Correlated shadow fading
    - Realistic user mobility patterns
    - Multi-user interference scenarios
    """
    
    def __init__(self):
        """Initialize the wireless data generator with 5G parameters"""
        # 5G frequency bands (GHz)
        self.frequency_bands = {
            'sub6': [0.7, 2.6, 3.5],      # Sub-6 GHz bands
            'mmwave': [28, 39, 60]         # mmWave bands
        }
        
        # Environment-specific parameters
        self.environment_params = {
            'urban': {'path_loss_exp': 3.5, 'shadow_std': 8.0, 'interference_high': True},
            'suburban': {'path_loss_exp': 3.2, 'shadow_std': 6.0, 'interference_high': False},
            'rural': {'path_loss_exp': 2.8, 'shadow_std': 4.0, 'interference_high': False}
        }
        
        # CQI to SNR mapping (3GPP standards)
        self.cqi_to_snr = {
            1: -6.5, 2: -4.5, 3: -2.5, 4: -0.5, 5: 1.5, 6: 3.5, 7: 5.5, 8: 7.5,
            9: 9.5, 10: 11.5, 11: 13.5, 12: 15.5, 13: 17.5, 14: 19.5, 15: 21.5
        }
    
    def calculate_path_loss(self, distance_m, frequency_ghz, environment='urban', los=True):
        """
        Calculate path loss using 3GPP TR 38.901 models
        
        Args:
            distance_m (float): Distance to base station in meters
            frequency_ghz (float): Carrier frequency in GHz
            environment (str): 'urban', 'suburban', or 'rural'
            los (bool): Line-of-sight condition
        
        Returns:
            float: Path loss in dB
        """
        # Ensure minimum distance
        distance_m = max(distance_m, 10.0)
        
        if environment == 'urban':
            if los:
                # 3GPP UMi LOS model
                if distance_m <= 1000:
                    pl = 32.4 + 21 * np.log10(distance_m) + 20 * np.log10(frequency_ghz)
                else:
                    pl = 32.4 + 40 * np.log10(distance_m) + 20 * np.log10(frequency_ghz) - 9.5
            else:
                # 3GPP UMi NLOS model
                pl = 35.3 * np.log10(distance_m) + 22.4 + 21.3 * np.log10(frequency_ghz) - 0.3 * 20
        
        elif environment == 'suburban':
            if los:
                pl = 28.0 + 22 * np.log10(distance_m) + 20 * np.log10(frequency_ghz)
            else:
                pl = 32.4 + 30 * np.log10(distance_m) + 20 * np.log10(frequency_ghz)
        
        else:  # rural
            if los:
                pl = 20 * np.log10(40 * np.pi * distance_m * frequency_ghz / 3) + min(0.03 * pow(20, 1.72), 10) * np.log10(distance_m) - min(0.044 * pow(20, 1.72), 14.77) + 0.002 * np.log10(20) * distance_m
            else:
                pl = 35.3 * np.log10(distance_m) + 22.4 + 21.3 * np.log10(frequency_ghz)
        
        return max(pl, 30.0)  # Minimum path loss
    
    def calculate_shadow_fading(self, n_samples, correlation=0.8, std_db=6.0):
        """
        Generate correlated shadow fading using AR(1) model
        
        Args:
            n_samples (int): Number of samples
            correlation (float): Correlation coefficient
            std_db (float): Standard deviation in dB
        
        Returns:
            np.ndarray: Shadow fading values in dB
        """
        # Generate correlated log-normal shadow fading
        shadow = np.zeros(n_samples)
        shadow[0] = np.random.normal(0, std_db)
        
        for i in range(1, n_samples):
            shadow[i] = correlation * shadow[i-1] + np.sqrt(1 - correlation**2) * np.random.normal(0, std_db)
        
        return shadow
    
    def calculate_small_scale_fading(self, n_samples, k_factor=None):
        """
        Generate small-scale fading (Rayleigh/Rician)
        
        Args:
            n_samples (int): Number of samples
            k_factor (float): Rician K-factor in dB (None for Rayleigh)
        
        Returns:
            np.ndarray: Fading values in dB
        """
        if k_factor is None:
            # Rayleigh fading (NLOS)
            h = np.random.rayleigh(1/np.sqrt(2), n_samples)
            fading_db = 20 * np.log10(h)
        else:
            # Rician fading (LOS)
            k_linear = 10**(k_factor/10)
            h_los = np.sqrt(k_linear / (k_linear + 1))
            h_nlos = np.sqrt(1 / (2 * (k_linear + 1)))
            
            h_real = h_los + h_nlos * np.random.normal(0, 1, n_samples)
            h_imag = h_nlos * np.random.normal(0, 1, n_samples)
            h = np.sqrt(h_real**2 + h_imag**2)
            fading_db = 20 * np.log10(h)
        
        return fading_db
    
    def calculate_interference(self, num_users, frequency_ghz, environment='urban'):
        """
        Calculate interference levels based on network density
        
        Args:
            num_users (int): Number of users in cell
            frequency_ghz (float): Carrier frequency
            environment (str): Environment type
        
        Returns:
            float: Interference level in dB
        """
        # Base interference levels
        base_interference = {
            'urban': -85,
            'suburban': -95,
            'rural': -105
        }
        
        # Interference increases with user density and frequency
        density_factor = 10 * np.log10(max(num_users / 10, 1))
        frequency_factor = 2 * np.log10(frequency_ghz / 2.0)
        
        interference = base_interference[environment] + density_factor + frequency_factor
        
        # Add random variation
        interference += np.random.normal(0, 3)
        
        return interference
    
    def snr_to_cqi(self, snr_db):
        """
        Map SNR to CQI using realistic 3GPP mapping
        
        Args:
            snr_db (float): SNR in dB
        
        Returns:
            int: CQI value (1-15)
        """
        # Find closest CQI value
        cqi_values = list(self.cqi_to_snr.keys())
        snr_thresholds = list(self.cqi_to_snr.values())
        
        # Add some noise to make mapping more realistic
        noisy_snr = snr_db + np.random.normal(0, 0.5)
        
        for i, threshold in enumerate(snr_thresholds):
            if noisy_snr < threshold:
                return max(1, cqi_values[i])
        
        return 15  # Maximum CQI
    
    def generate_dataset(self, n_samples=15000):
        """
        Generate comprehensive synthetic wireless dataset
        
        Args:
            n_samples (int): Number of samples to generate
        
        Returns:
            pd.DataFrame: Generated dataset with realistic wireless features
        """
        print(f"🏗️ Generating {n_samples} samples with realistic 5G/B5G channel models...")
        
        # Initialize arrays
        data = {
            'user_speed_kmh': [],
            'distance_to_bs_m': [],
            'frequency_ghz': [],
            'los_nlos': [],
            'power_level_dbm': [],
            'num_users': [],
            'environment_type': [],
            'interference_level_db': [],
            'snr_db': [],
            'cqi': []
        }
        
        # Generate correlated shadow fading for entire dataset
        shadow_fading = self.calculate_shadow_fading(n_samples, correlation=0.7)
        
        for i in range(n_samples):
            # User mobility (realistic speed distribution)
            if np.random.random() < 0.6:  # Pedestrian
                speed = np.random.gamma(2, 2)  # 0-10 km/h mainly
            elif np.random.random() < 0.8:  # Vehicle
                speed = np.random.normal(50, 15)  # Urban vehicle speeds
            else:  # High-speed
                speed = np.random.normal(120, 30)  # Highway speeds
            speed = max(0, min(speed, 300))  # Clip to realistic range
            
            # Distance to base station (cell-dependent distribution)
            if np.random.random() < 0.4:  # Urban dense
                distance = np.random.exponential(200) + 50  # Close to BS
            else:  # Suburban/rural
                distance = np.random.exponential(800) + 100  # Further from BS
            distance = max(10, min(distance, 5000))  # 10m to 5km range
            
            # Frequency selection (weighted toward sub-6 GHz)
            if np.random.random() < 0.7:
                frequency = np.random.choice(self.frequency_bands['sub6'])
            else:
                frequency = np.random.choice(self.frequency_bands['mmwave'])
            
            # Environment type (correlated with distance)
            if distance < 500:
                env_probs = [0.7, 0.2, 0.1]  # Likely urban
            elif distance < 1500:
                env_probs = [0.3, 0.6, 0.1]  # Likely suburban
            else:
                env_probs = [0.1, 0.3, 0.6]  # Likely rural
            
            environment = np.random.choice(['urban', 'suburban', 'rural'], p=env_probs)
            
            # LOS/NLOS probability based on environment and distance
            if environment == 'urban':
                los_prob = max(0.1, np.exp(-distance/200))
            elif environment == 'suburban':
                los_prob = max(0.2, np.exp(-distance/500))
            else:  # rural
                los_prob = max(0.4, np.exp(-distance/1000))
            
            is_los = np.random.random() < los_prob
            los_nlos = 'LOS' if is_los else 'NLOS'
            
            # Base station transmission power (environment dependent)
            if environment == 'urban':
                tx_power = np.random.normal(40, 3)  # Urban macro
            elif environment == 'suburban':
                tx_power = np.random.normal(43, 2)  # Suburban macro
            else:
                tx_power = np.random.normal(46, 2)  # Rural macro
            
            tx_power = max(20, min(tx_power, 50))  # Realistic range
            
            # Number of users (time-varying, environment dependent)
            if environment == 'urban':
                num_users = max(1, int(np.random.poisson(25)))
            elif environment == 'suburban':
                num_users = max(1, int(np.random.poisson(15)))
            else:
                num_users = max(1, int(np.random.poisson(8)))
            
            # Calculate path loss
            path_loss = self.calculate_path_loss(distance, frequency, environment, is_los)
            
            # Add shadow fading (already generated)
            shadow_db = shadow_fading[i] * self.environment_params[environment]['shadow_std'] / 6.0
            
            # Small-scale fading
            if is_los and frequency < 10:  # Rician for LOS sub-6GHz
                k_factor = np.random.normal(10, 3)  # dB
                fading_db = self.calculate_small_scale_fading(1, k_factor)[0]
            else:  # Rayleigh for NLOS or mmWave
                fading_db = self.calculate_small_scale_fading(1, None)[0]
            
            # Calculate interference
            interference = self.calculate_interference(num_users, frequency, environment)
            
            # Calculate received power and SNR
            rx_power = tx_power - path_loss + shadow_db + fading_db
            thermal_noise = -174 + 10 * np.log10(100e6)  # 100 MHz bandwidth
            snr = rx_power - thermal_noise - 10 * np.log10(10**(interference/10) + 1)
            
            # Add measurement noise
            snr += np.random.normal(0, 1.0)
            
            # Map SNR to CQI
            cqi = self.snr_to_cqi(snr)
            
            # Store data
            data['user_speed_kmh'].append(speed)
            data['distance_to_bs_m'].append(distance)
            data['frequency_ghz'].append(frequency)
            data['los_nlos'].append(los_nlos)
            data['power_level_dbm'].append(tx_power)
            data['num_users'].append(num_users)
            data['environment_type'].append(environment)
            data['interference_level_db'].append(interference)
            data['snr_db'].append(snr)
            data['cqi'].append(cqi)
            
            if (i + 1) % 1000 == 0:
                print(f"   Generated {i + 1}/{n_samples} samples...")
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Add some derived features for more realism
        df['distance_normalized'] = df['distance_to_bs_m'] / df['distance_to_bs_m'].max()
        df['frequency_normalized'] = df['frequency_ghz'] / df['frequency_ghz'].max()
        
        print(f"✅ Dataset generation completed!")
        print(f"   • SNR range: {df['snr_db'].min():.1f} to {df['snr_db'].max():.1f} dB")
        print(f"   • CQI distribution: {dict(df['cqi'].value_counts().sort_index())}")
        print(f"   • Environment distribution: {dict(df['environment_type'].value_counts())}")
        print(f"   • LOS/NLOS ratio: {dict(df['los_nlos'].value_counts())}")
        
        return df.drop(['distance_normalized', 'frequency_normalized'], axis=1)

def create_results_directory():
    """Create results directory if it doesn't exist"""
    if not os.path.exists('results'):
        os.makedirs('results')
        print("📁 Created results/ directory")

def plot_training_history(history, model_name, metric='accuracy'):
    """
    Plot training history with loss and metrics
    
    Args:
        history: Keras training history object
        model_name (str): Name of the model for plot title
        metric (str): Metric to plot alongside loss
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training & validation loss
    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_title(f'{model_name} - Training Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot training & validation metric
    if metric in history.history:
        axes[1].plot(history.history[metric], label=f'Training {metric.upper()}', linewidth=2)
        axes[1].plot(history.history[f'val_{metric}'], label=f'Validation {metric.upper()}', linewidth=2)
        axes[1].set_title(f'{model_name} - {metric.upper()}', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel(metric.upper())
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'results/{model_name.lower().replace(" ", "_")}_training_history.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def plot_predictions(y_true, y_pred, target_name, task_type='regression'):
    """
    Plot prediction results with various visualizations
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        target_name (str): Name of target variable
        task_type (str): 'regression' or 'classification'
    """
    if task_type == 'regression':
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Scatter plot: Predicted vs Actual
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6, s=20)
        axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel(f'Actual {target_name}')
        axes[0, 0].set_ylabel(f'Predicted {target_name}')
        axes[0, 0].set_title(f'Predicted vs Actual {target_name}')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residual plot
        residuals = y_pred - y_true
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6, s=20)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel(f'Predicted {target_name}')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Histogram of residuals
        axes[1, 0].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Residuals')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Q-Q plot for normality check
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot of Residuals')
        axes[1, 1].grid(True, alpha=0.3)
        
    else:  # classification
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        im = axes[0, 0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted CQI')
        axes[0, 0].set_ylabel('Actual CQI')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[0, 0])
        
        # Prediction distribution
        axes[0, 1].hist(y_true, bins=15, alpha=0.5, label='Actual', range=(1, 15))
        axes[0, 1].hist(y_pred, bins=15, alpha=0.5, label='Predicted', range=(1, 15))
        axes[0, 1].set_xlabel('CQI Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('CQI Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Error by CQI value
        cqi_values = np.unique(y_true)
        errors = []
        for cqi in cqi_values:
            mask = y_true == cqi
            if np.sum(mask) > 0:
                error = np.mean(np.abs(y_pred[mask] - y_true[mask]))
                errors.append(error)
            else:
                errors.append(0)
        
        axes[1, 0].bar(cqi_values, errors)
        axes[1, 0].set_xlabel('True CQI Value')
        axes[1, 0].set_ylabel('Mean Absolute Error')
        axes[1, 0].set_title('Prediction Error by CQI Value')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Accuracy by CQI
        accuracies = []
        for cqi in cqi_values:
            mask = y_true == cqi
            if np.sum(mask) > 0:
                acc = np.mean(y_pred[mask] == y_true[mask])
                accuracies.append(acc)
            else:
                accuracies.append(0)
        
        axes[1, 1].bar(cqi_values, accuracies)
        axes[1, 1].set_xlabel('CQI Value')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_title('Prediction Accuracy by CQI')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'results/{target_name.lower().replace(" ", "_").replace("(", "").replace(")", "")}_predictions.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model_performance(y_true, y_pred, model_name, task_type='regression'):
    """
    Comprehensive model performance evaluation
    
    Args:
        y_true: True values
        y_pred: Predicted values  
        model_name (str): Name of the model
        task_type (str): Type of task
    
    Returns:
        dict: Performance metrics
    """
    if task_type == 'regression':
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Additional metrics
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        metrics = {
            'MAE': mae,
            'MSE': mse, 
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape
        }
        
    else:  # classification
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        metrics = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        }
    
    print(f"\n📊 {model_name} Performance Metrics:")
    print("-" * 40)
    for metric, value in metrics.items():
        print(f"   {metric}: {value:.4f}")
    
    return metrics 