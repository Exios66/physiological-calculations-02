# Examples

This directory contains example scripts demonstrating how to use the Physiological Metrics Calculator.

## Contents

- `example_usage.py`: Basic example of using the cognitive workload pipeline to analyze physiological data and generate a report.

## Running the Examples

To run an example script:

```bash
cd examples
python example_usage.py
```

Note: You may need to modify the data file paths in the examples to point to your actual data files.

## Expected Output

When running the examples, you should expect:

1. Processing messages showing the progress of the analysis
2. A summary of the results printed to the console
3. Output files (reports, plots) saved to the `output` directory

## Troubleshooting

If you encounter errors:

- Ensure you have installed all required dependencies (`pip install -r ../requirements.txt`)
- Check that the data file paths are correct
- Verify that your data files have the expected format (CSV with required columns) 