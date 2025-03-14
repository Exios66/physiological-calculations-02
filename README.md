# Physiological Metrics Calculator

A comprehensive toolkit for analyzing physiological data to assess cognitive workload and other physiological metrics.

## Overview

This repository contains tools for processing and analyzing physiological data such as SpO2 (oxygen saturation), pulse rate, and other metrics. The primary focus is on cognitive workload assessment through physiological indicators.

## Features

- **Data Processing Pipeline**: Clean, transform, and analyze physiological time-series data
- **Feature Engineering**: Calculate derived metrics including:
  - Heart Rate Variability (HRV)
  - SpO2 variability and trends
  - Rolling statistics (mean, standard deviation)
  - Physiological load indices
- **Cognitive Workload Assessment**: Identify periods of elevated, high, and critical cognitive load
- **Visualization**: Generate plots and charts for data interpretation
- **Reporting**: Create comprehensive clinical reports from analysis results

## Repository Structure

```
physiological-calculations/
├── data/                  # Data directory
│   ├── pilot-data/        # Pilot study data
│   └── raw-data/          # Raw physiological data files
├── examples/              # Example scripts
│   ├── example_usage.py   # Basic usage example
│   └── README.md          # Examples documentation
├── src/                   # Source code
│   ├── python/            # Python implementation
│   │   └── generate_report.py  # Main analysis pipeline
│   └── js/                # JavaScript implementation (future)
├── CHANGELOG.md           # Detailed changelog
├── CONTRIBUTING.md        # Contribution guidelines
├── LICENSE                # MIT License
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Getting Started

### Prerequisites

- Python 3.8+
- Required Python packages:
  - numpy
  - pandas
  - scipy
  - matplotlib
  - seaborn
  - feature_engine
  - statsmodels
  - scikit-learn

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/physiological-calculations.git
   cd physiological-calculations
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

The main functionality is provided by the `cognitive_workload_pipeline` function in `src/python/generate_report.py`:

```python
from src.python.generate_report import cognitive_workload_pipeline, generate_clinical_report

# Process data and generate analysis
results_df, report_dict = cognitive_workload_pipeline('path/to/your/data.csv')

# Generate a clinical report
generate_clinical_report(report_dict, results_df, 'output/clinical_report.md')
```

For more detailed examples, see the [examples](examples/) directory.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute to this project.

```bash
