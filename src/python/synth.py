import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_clinical_dataset():
    np.random.seed(42)
    base_time = datetime(2025, 2, 26, 16, 0)
    timestamps = [base_time + timedelta(minutes=i) for i in range(180)]  # 3-hour dataset
    
    # Base physiological signals
    spo2 = np.clip(np.random.normal(97, 1.2, 180), 88, 100)
    pulse = np.random.normal(82, 4, 180)
    
    # Clinical event patterns
    # Hypoxemia episodes
    spo2[30:35] = np.linspace(96, 82, 5)  # Event A
    pulse[30:35] = np.linspace(85, 112, 5)
    
    spo2[65:70] = np.linspace(95, 84, 5)  # Event B
    pulse[65:70] = np.linspace(88, 108, 5)
    
    # Sustained cognitive load
    spo2[90:120] = 95 + np.sin(np.linspace(0, 4*np.pi, 30)) * 3
    pulse[90:120] = 80 + np.abs(np.random.randn(30)) * 8
    
    # Acute stress spikes
    pulse[45:47] = [118, 121]  # Spike 1
    pulse[130:132] = [115, 119]  # Spike 2
    
    # Recovery phases
    spo2[140:150] = np.linspace(94, 98, 10)
    pulse[140:150] = np.linspace(105, 88, 10)
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'SpO2': np.round(spo2, 1),
        'pulse_rate': np.clip(np.round(pulse + np.random.randn(180)*2), 60, 125).astype(int)
    })

# Generate and format data
clinical_data = generate_clinical_dataset()
formatted_data = clinical_data.to_csv(index=False)
