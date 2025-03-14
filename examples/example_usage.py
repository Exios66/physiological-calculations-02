#!/usr/bin/env python3
"""
Example usage of the Physiological Metrics Calculator.

This script demonstrates how to use the cognitive workload pipeline
to analyze physiological data and generate a clinical report.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.python.generate_report import cognitive_workload_pipeline, generate_clinical_report

def main():
    """Run an example analysis using the cognitive workload pipeline."""
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Path to example data file
    # Replace with your actual data file path
    data_file = '../data/pilot-data/example_data.csv'
    
    print(f"Processing data from {data_file}...")
    
    try:
        # Run the cognitive workload pipeline
        processed_df, report_dict = cognitive_workload_pipeline(data_file, output_dir=output_dir)
        
        # Generate a clinical report
        report_file = os.path.join(output_dir, 'clinical_report.md')
        generate_clinical_report(report_dict, processed_df, report_file)
        
        print(f"Analysis complete. Report saved to {report_file}")
        
        # Display a summary of the results
        print("\nSummary of Results:")
        print(f"- Total records processed: {len(processed_df)}")
        
        if 'workload_events' in report_dict:
            workload_events = report_dict['workload_events']
            print(f"- Elevated workload events: {len(workload_events)}")
            
        if 'cluster_summary' in report_dict:
            cluster_summary = report_dict['cluster_summary']
            print(f"- Number of clusters identified: {len(cluster_summary)}")
        
    except FileNotFoundError:
        print(f"Error: Data file {data_file} not found.")
        print("Please update the data_file path to point to a valid CSV file.")
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main() 