import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from feature_engine.creation import MathFeatures, CombineWithReferenceFeature
from feature_engine.selection import DropFeatures
import statsmodels.api as sm  # Import statsmodels API
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')
sns.set(style="whitegrid")  # Global style for seaborn


def cognitive_workload_pipeline(filepath, output_dir='./output'):
    """
    Comprehensive cognitive workload analysis pipeline for physiological data.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file containing physiological data. Must include at least:
          - 'timestamp' (datetime or parseable date string)
          - 'SpO2' (float, oxygen saturation %)
          - 'pulse_rate' (float, pulse rate in bpm)
    output_dir : str, optional
        Directory to save output files. Defaults to './output'.
        
    Returns
    -------
    df : pandas.DataFrame
        Processed DataFrame with engineered features and cluster labels.
    report : dict
        Dictionary containing analysis results and metrics, including:
          - 'summary_stats': overall descriptive statistics
          - 'workload_events': subset of data with elevated or critical workload
          - 'critical_periods': dictionary of DataFrame slices for key intervals
          - 'cluster_summary': average values of features by cluster
          - 'outliers': subset of data flagged as outliers
          
    Notes
    -----
    [1] Outliers are handled via IQR-based filtering (1.5 * IQR rule).
    [2] Feature engineering includes rolling means, rolling std, HRV, SpO2 changes, etc.
    [3] A z-score approach is used to define levels of cognitive load (Normal, Elevated, High, Critical).
    [4] KMeans is used for cluster analysis on a subset of key features.
    [5] Multiple plots are generated and saved to the output directory.
    [6] Minimal validation steps are included to handle missing data or columns.
    [7] Additional references: 'feature_engine' library for combining & creating features.
    [8] Critical periods are either manually defined or auto-detected (hypoxemia).
    [9] Subset analysis performed for intervals with high or critical workload.
    [10] This pipeline can be extended or modified for domain-specific thresholds.
    """

    # -----------------------------
    # 1. VALIDATION & DATA INGESTION
    # -----------------------------
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File '{filepath}' does not exist.")

    print(f"\n[INFO] Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath, parse_dates=['timestamp'])
    except ValueError as e:
        raise ValueError(
            "Failed to parse CSV. Ensure 'timestamp' can be parsed as dates."
        ) from e
    
    required_cols = {'timestamp', 'SpO2', 'pulse_rate'}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in the input data: {missing_cols}")

    # Drop rows missing critical info
    df.dropna(subset=['SpO2', 'pulse_rate'], inplace=True)
    if df.empty:
        raise ValueError("No valid rows remain after dropping rows with missing SpO2 or pulse_rate.")

    # Convert any possibly non-numeric SpO2 or pulse_rate to numeric
    df['SpO2'] = pd.to_numeric(df['SpO2'], errors='coerce')
    df['pulse_rate'] = pd.to_numeric(df['pulse_rate'], errors='coerce')
    df.dropna(subset=['SpO2', 'pulse_rate'], inplace=True)
    if df.empty:
        raise ValueError("No valid rows remain after ensuring numeric SpO2/pulse_rate.")

    # -----------------------------
    # 2. BASIC TIME-BASED FEATURES
    # -----------------------------
    print("[INFO] Generating time-based features...")
    df.sort_values('timestamp', inplace=True)
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['second'] = df['timestamp'].dt.second
    df['time_of_day'] = df['hour'] * 3600 + df['minute'] * 60 + df['second']
    
    # -----------------------------
    # 3. OUTLIER DETECTION (IQR)
    # -----------------------------
    print("[INFO] Detecting and filtering outliers (IQR-based)...")
    Q1 = df[['SpO2', 'pulse_rate']].quantile(0.25)
    Q3 = df[['SpO2', 'pulse_rate']].quantile(0.75)
    IQR = Q3 - Q1
    
    df['SpO2_outlier'] = (
        (df['SpO2'] < (Q1['SpO2'] - 1.5 * IQR['SpO2'])) |
        (df['SpO2'] > (Q3['SpO2'] + 1.5 * IQR['SpO2']))
    )
    df['pulse_outlier'] = (
        (df['pulse_rate'] < (Q1['pulse_rate'] - 1.5 * IQR['pulse_rate'])) |
        (df['pulse_rate'] > (Q3['pulse_rate'] + 1.5 * IQR['pulse_rate']))
    )

    outliers = df[df['SpO2_outlier'] | df['pulse_outlier']].copy()
    
    # Filter them out from main DF
    df = df[~((df[['SpO2','pulse_rate']] < (Q1 - 1.5 * IQR)) | 
              (df[['SpO2','pulse_rate']] > (Q3 + 1.5 * IQR))).any(axis=1)]
    
    if df.empty:
        raise ValueError("All data was filtered out as outliers! Check your outlier thresholds.")

    # -----------------------------
    # 4. FEATURE ENGINEERING
    # -----------------------------
    print("[INFO] Creating engineered features...")
    
    # Heart Rate Variability (HRV) approximation with a rolling std dev
    # "window=10" here is arbitrary; adjust as needed for sampling rate
    df['HRV'] = df['pulse_rate'].rolling(window=10).std().fillna(0)
    df['HRV_normalized'] = df['HRV'] / df['pulse_rate']
    
    # RMSSD: typical short-term HRV metric
    # This is a single global value, not rolling, because we call np.diff on the entire column
    if len(df) > 1:
        df['RMSSD'] = np.sqrt(np.mean(np.square(np.diff(df['pulse_rate']))))
    else:
        df['RMSSD'] = np.nan  # Not enough data to compute
    
    # SpO2 dynamics (change, acceleration, smoothed curve)
    df['SpO2_change'] = df['SpO2'].diff().fillna(0)
    df['SpO2_accel'] = df['SpO2_change'].diff().fillna(0)
    
    # LOWESS smoothing for SpO2
    # frac=0.05 means 5% of data used in local regression
    if len(df) >= 5:
        df['SpO2_smooth'] = sm.nonparametric.lowess(df['SpO2'], np.arange(len(df)), frac=0.05)[:, 1]
    else:
        # If not enough data to smooth, just copy original
        df['SpO2_smooth'] = df['SpO2']
    
    # Rolling stats at multiple scales (5, 10, 30, 60 "points" windows)
    # Adjust these windows based on your sampling frequency
    windows = [5, 10, 30, 60]
    for w in windows:
        df[f'pulse_rate_mean_{w}'] = (
            df['pulse_rate'].rolling(window=w).mean().fillna(method='bfill')
        )
        df[f'pulse_rate_std_{w}'] = (
            df['pulse_rate'].rolling(window=w).std().fillna(0)
        )
        df[f'SpO2_mean_{w}'] = (
            df['SpO2'].rolling(window=w).mean().fillna(method='bfill')
        )
        df[f'SpO2_min_{w}'] = (
            df['SpO2'].rolling(window=w).min().fillna(method='bfill')
        )
    
    # -----------------------------
    # 5. ADVANCED FEATURE ENGINEERING via Feature-engine
    # -----------------------------
    # Math features: sum, product, max, min for pulse_rate & HRV
    transformer = MathFeatures(
        variables=['pulse_rate', 'HRV'], 
        func=['sum', 'prod', 'max', 'min']
    )
    df = transformer.fit_transform(df)
    
    # Combine each of these with a "reference" of pulse_rate_mean_60
    reference = CombineWithReferenceFeature(
        variables=['pulse_rate', 'HRV', 'SpO2_change'],
        reference=df['pulse_rate_mean_60'],  # reference Series
        func=['diff', 'div']
    )
    df = reference.fit_transform(df)

    # A custom "cognitive load index" that merges multiple signals
    # Adjust weighting or included metrics as needed
    df['cognitive_load_index'] = (
        0.5 * (df['pulse_rate'] / df['pulse_rate_mean_60']) +
        0.3 * df['HRV_normalized'] +
        0.2 * df['SpO2_change'].abs()
    )
    
    # Drop some ephemeral columns we do not need
    dropper = DropFeatures(features=[
        'hour', 
        'minute', 
        'second',
        'SpO2_outlier', 
        'pulse_outlier'
    ])
    df = dropper.fit_transform(df)

    # -----------------------------
    # 6. PSYCHOMETRIC & STATISTICAL ANALYSIS
    # -----------------------------
    print("[INFO] Calculating descriptive statistics and z-scores...")
    stats_summary = df.describe(
        percentiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    )

    # Z-score transformations for anomaly/workload detection
    df['pulse_zscore'] = np.abs(stats.zscore(df['pulse_rate'], nan_policy='omit'))
    df['SpO2_zscore'] = np.abs(stats.zscore(df['SpO2'], nan_policy='omit'))
    df['cognitive_load_zscore'] = np.abs(stats.zscore(df['cognitive_load_index'], nan_policy='omit'))
    
    # Label workload levels
    df['workload_level'] = 'Normal'
    df.loc[df['cognitive_load_zscore'] > 1.5, 'workload_level'] = 'Elevated'
    df.loc[df['cognitive_load_zscore'] > 2.0, 'workload_level'] = 'High'
    df.loc[df['cognitive_load_zscore'] > 2.5, 'workload_level'] = 'Critical'
    
    # Subset of events that exceed Elevated threshold
    workload_events = df[df['cognitive_load_zscore'] > 1.5].copy()
    
    # -----------------------------
    # 7. CLUSTER ANALYSIS
    # -----------------------------
    print("[INFO] Performing KMeans clustering on key physiological features...")
    feature_columns = ['pulse_rate', 'SpO2', 'HRV', 'cognitive_load_index']
    # Only cluster if we have enough data
    if len(df) < 2:
        print("[WARN] Not enough data for clustering. Skipping KMeans.")
        df['cluster'] = np.nan
        cluster_summary = pd.DataFrame()
    else:
        X = df[feature_columns].copy()
        # Scale for KMeans
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Reasonable default for small datasets: up to 3 clusters
        n_clusters = min(3, max(1, len(df) // 100)) 
        if n_clusters == 1:
            # Not enough data for multiple clusters
            df['cluster'] = 0
            cluster_summary = pd.DataFrame({'message': ['Only one cluster possible']})
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            df['cluster'] = kmeans.fit_predict(X_scaled)
            cluster_summary = df.groupby('cluster')[feature_columns].mean()

    # -----------------------------
    # 8. CRITICAL PERIOD ANALYSIS
    # -----------------------------
    print("[INFO] Identifying critical periods...")
    
    # We'll define reference times from the earliest timestamp
    start_time = df['timestamp'].min()
    critical_periods = {
        'hypoxemia_1': df[
            (df['timestamp'] >= start_time + timedelta(minutes=55)) & 
            (df['timestamp'] <= start_time + timedelta(minutes=63))
        ],
        'hypoxemia_2': df[
            (df['timestamp'] >= start_time + timedelta(minutes=73)) & 
            (df['timestamp'] <= start_time + timedelta(minutes=76))
        ],
        'sustained_stress': df[
            (df['timestamp'] >= start_time + timedelta(minutes=82)) & 
            (df['timestamp'] <= start_time + timedelta(minutes=113))
        ],
    }
    
    # Auto-detection of low SpO2 segments
    df['low_SpO2'] = df['SpO2'] < 95
    # Each time low_SpO2 changes from True to False or vice versa, we count a new "segment"
    df['low_SpO2_segment'] = (df['low_SpO2'] != df['low_SpO2'].shift()).cumsum()
    low_SpO2_segments = df[df['low_SpO2']].groupby('low_SpO2_segment')
    
    # For each continuous low-SpO2 segment > N points, store as a critical period
    # (assuming ~10-second intervals, 20 points ~ 200 seconds = ~3.3 minutes, adjust as needed)
    for i, (segment_id, seg) in enumerate(low_SpO2_segments):
        if len(seg) > 20:
            critical_periods[f'detected_hypoxemia_{i+1}'] = seg

    # -----------------------------
    # 9. VISUALIZATIONS & SAVE PLOTS
    # -----------------------------
    print(f"[INFO] Creating output directory '{output_dir}' and saving plots...")
    os.makedirs(output_dir, exist_ok=True)
    
    # -- Main multi-plot figure
    plt.figure(figsize=(15, 10))

    # (A) SpO2 over time
    plt.subplot(3, 1, 1)
    plt.plot(df['timestamp'], df['SpO2'], 'b-', label='SpO2 (%)', alpha=0.7)
    if 'SpO2_smooth' in df.columns:
        plt.plot(df['timestamp'], df['SpO2_smooth'], 'b-', label='SpO2 (Smoothed)', linewidth=2)
    plt.axhline(y=95, color='r', linestyle='--', alpha=0.5, label='SpO2 Threshold (95%)')
    plt.legend(loc='upper right')
    plt.title('Oxygen Saturation (SpO2) Over Time', fontsize=14)
    plt.ylabel('SpO2 (%)')

    # (B) Pulse Rate & Workload Events
    plt.subplot(3, 1, 2)
    plt.plot(df['timestamp'], df['pulse_rate'], 'g-', label='Pulse Rate (bpm)', alpha=0.7)
    if 'pulse_rate_mean_30' in df.columns:
        plt.plot(df['timestamp'], df['pulse_rate_mean_30'], 'g-', label='Pulse Rate (30-pt MA)', linewidth=2)
    if not workload_events.empty:
        plt.scatter(workload_events['timestamp'], workload_events['pulse_rate'], 
                    color='red', label='Workload Events', alpha=0.5)
    plt.legend(loc='upper right')
    plt.title('Pulse Rate and Workload Events', fontsize=14)
    plt.ylabel('Pulse Rate (bpm)')

    # (C) Cognitive Load Index
    plt.subplot(3, 1, 3)
    plt.plot(df['timestamp'], df['cognitive_load_index'], 'm-', label='Cognitive Load Index', alpha=0.7)
    critical_mask = df['workload_level'] == 'Critical'
    if critical_mask.any():
        plt.scatter(
            df.loc[critical_mask, 'timestamp'],
            df.loc[critical_mask, 'cognitive_load_index'],
            color='red', label='Critical Load', alpha=0.7
        )
    plt.legend(loc='upper right')
    plt.title('Cognitive Load Index', fontsize=14)
    plt.xlabel('Time')
    plt.ylabel('Index Value')

    plt.tight_layout()
    main_plot_path = os.path.join(output_dir, 'clinical_analysis_main.png')
    plt.savefig(main_plot_path, dpi=300)
    plt.close()
    
    # -- Heatmap: feature correlations
    corr_features = [
        'SpO2', 'pulse_rate', 'HRV', 'cognitive_load_index',
        'SpO2_change', 'pulse_rate_mean_30'
    ]
    corr_features = [f for f in corr_features if f in df.columns]
    if len(corr_features) > 1:
        plt.figure(figsize=(12, 10))
        correlation_matrix = df[corr_features].corr()
        sns.heatmap(
            correlation_matrix, annot=True, cmap='coolwarm', 
            linewidths=0.5, square=False
        )
        plt.title('Feature Correlation Matrix', fontsize=14)
        plt.tight_layout()
        corr_plot_path = os.path.join(output_dir, 'feature_correlations.png')
        plt.savefig(corr_plot_path, dpi=300)
        plt.close()
    
    # -- Workload distribution
    plt.figure(figsize=(12, 6))
    sns.countplot(x='workload_level', data=df, palette='viridis')
    plt.title('Distribution of Workload Levels', fontsize=14)
    plt.xlabel('Workload Level')
    plt.ylabel('Count')
    plt.tight_layout()
    workload_dist_path = os.path.join(output_dir, 'workload_distribution.png')
    plt.savefig(workload_dist_path, dpi=300)
    plt.close()
    
    # -- Cluster visualization (pulse_rate vs. SpO2)
    if 'cluster' in df.columns and df['cluster'].notna().any():
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            data=df, x='pulse_rate', y='SpO2', 
            hue='cluster', palette='viridis', alpha=0.6
        )
        plt.title('Clustering of Physiological States', fontsize=14)
        plt.xlabel('Pulse Rate (bpm)')
        plt.ylabel('SpO2 (%)')
        plt.tight_layout()
        cluster_plot_path = os.path.join(output_dir, 'physiological_clusters.png')
        plt.savefig(cluster_plot_path, dpi=300)
        plt.close()
    
    # -- Critical periods: create a plot for each
    for period_name, period_data in critical_periods.items():
        if len(period_data) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(period_data['timestamp'], period_data['SpO2'], 'b-', label='SpO2')
            plt.plot(period_data['timestamp'], period_data['pulse_rate'], 'g-', label='Pulse Rate')
            plt.title(f'Critical Period: {period_name}', fontsize=14)
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend(loc='best')
            plt.tight_layout()
            critical_path = os.path.join(output_dir, f'critical_period_{period_name}.png')
            plt.savefig(critical_path, dpi=300)
            plt.close()
    
    # -----------------------------
    # 10. PREPARE THE REPORT
    # -----------------------------
    report = {
        'summary_stats': stats_summary,
        'workload_events': workload_events,
        'critical_periods': critical_periods,
        'cluster_summary': cluster_summary,
        'outliers': outliers
    }

    print("[INFO] Pipeline execution complete.")
    return df, report


def generate_clinical_report(report_dict, processed_df, output_file, include_plots=True):
    """
    Generate a comprehensive clinical report from the analysis results.
    
    Parameters
    ----------
    report_dict : dict
        Dictionary containing analysis results from the pipeline. Expected keys:
          - 'summary_stats': DataFrame of descriptive stats
          - 'workload_events': DataFrame subset with elevated/critical workload
          - 'critical_periods': dict of DataFrames for interesting intervals
          - 'cluster_summary': DataFrame summarizing cluster means
          - 'outliers': DataFrame of flagged outlier points
    processed_df : pandas.DataFrame
        The full processed DataFrame returned by `cognitive_workload_pipeline`.
    output_file : str
        Path to the output text (or markdown) file to be written.
    include_plots : bool, optional
        Whether to reference plots in the final report. Defaults to True.
        
    Notes
    -----
    This function writes out a plain-text (or simple Markdown) summary. 
    For more advanced reporting (PDF, HTML dashboards, etc.), consider
    a dedicated templating library (e.g., Jinja2 or nbconvert).
    """
    print(f"[INFO] Generating clinical report => {output_file}")
    
    # Gather pieces from the report dictionary
    summary_stats = report_dict.get('summary_stats')
    workload_events = report_dict.get('workload_events')
    critical_periods = report_dict.get('critical_periods', {})
    cluster_summary = report_dict.get('cluster_summary')
    outliers = report_dict.get('outliers')

    # Start building a textual report
    lines = []
    lines.append("# Clinical Analysis Report\n")
    lines.append("Generated by Cognitive Workload Pipeline\n\n")
    
    # Basic descriptive stats
    lines.append("## 1. Basic Descriptive Statistics\n")
    if isinstance(summary_stats, pd.DataFrame):
        lines.append("**Summary Stats (Selected Columns):**\n")
        lines.append(str(summary_stats))
        lines.append("\n")
    else:
        lines.append("No summary stats available.\n\n")
    
    # Outlier summary
    lines.append("## 2. Outlier Summary\n")
    if outliers is not None and not outliers.empty:
        lines.append(f"Total outliers detected: {len(outliers)}\n")
    else:
        lines.append("No outliers detected or outlier detection was skipped.\n")
    lines.append("\n")
    
    # Workload events
    lines.append("## 3. Workload Events\n")
    if workload_events is not None and not workload_events.empty:
        lines.append(f"Total Elevated+ Workload Events: {len(workload_events)}\n")
        # Show first few
        lines.append("Sample of Elevated Workload Events:\n")
        lines.append(str(workload_events.head(5)))
        lines.append("\n")
    else:
        lines.append("No elevated workload events detected.\n\n")

    # Cluster summary
    lines.append("## 4. Cluster Analysis\n")
    if cluster_summary is not None and not cluster_summary.empty:
        lines.append(f"Number of clusters found: {cluster_summary.shape[0]}\n")
        lines.append("Cluster Averages:\n")
        lines.append(str(cluster_summary))
        lines.append("\n")
    else:
        lines.append("Clustering was not performed or only one cluster was found.\n\n")
    
    # Critical periods
    lines.append("## 5. Critical Periods\n")
    if critical_periods:
        for name, data_slice in critical_periods.items():
            lines.append(f"### {name}\n")
            lines.append(f"Records in this period: {len(data_slice)}\n")
            if not data_slice.empty:
                # Show a brief time range
                t_min = data_slice['timestamp'].min()
                t_max = data_slice['timestamp'].max()
                lines.append(f"Time range: {t_min} to {t_max}\n\n")
            else:
                lines.append("*(Empty slice)*\n\n")
    else:
        lines.append("No critical periods identified.\n\n")

    # Reference to plots, if they exist
    if include_plots:
        lines.append("## 6. Generated Plots\n")
        lines.append("Please see the `.png` files in the output directory for:\n")
        lines.append("- Main time-series analysis\n- Feature correlations\n")
        lines.append("- Workload distribution\n- Physiological clusters\n- Critical period details\n\n")
    else:
        lines.append("*(Plots were not referenced in this report.)*\n")

    # Additional details
    lines.append("## 7. Additional Insights\n")
    lines.append("Further analysis or domain-specific metrics could be added here.\n")
    
    # Write everything out
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    
    print("[INFO] Clinical report generation complete.\n")
