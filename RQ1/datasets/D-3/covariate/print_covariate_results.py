#!/usr/bin/env python3
"""
print_covariate_results.py - Covariate Shift Results Printer
----------------------------------------------------------
Comprehensive script to print all covariate shift analysis results
including R² and p-values for both normalized and non-normalized data.
"""
import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings

DATASET_ROOT = Path(__file__).resolve().parents[1]
OVERFITTING_ROOT = DATASET_ROOT.parents[2] / "data" / "Overfitting"
DATA_ROOT = OVERFITTING_ROOT / "D-3"
RESULTS_ROOT = DATASET_ROOT / "results"

if str(DATASET_ROOT) not in sys.path:
    sys.path.append(str(DATASET_ROOT))

warnings.filterwarnings("ignore")

# Add parent directories to path

# Configuration
RESULTS_BASE_PATH = str(RESULTS_ROOT / "RQ1/CovariateShift")
DATA_TYPES = [
    'phone_usage',
    'mobility', 
    'physical_status',
    'sleep',
    'social_behavior'
]

class CovariateResultsPrinter:
    """Class to handle printing of covariate shift analysis results."""
    
    def __init__(self, results_path=RESULTS_BASE_PATH):
        self.results_path = Path(results_path)
        self.analysis_types = ['normalized', 'non_normalized']
        self.data_types = DATA_TYPES
        
    def find_analysis_files(self):
        """Find all analysis files in the results directory."""
        files = {
            'normalized': {},
            'non_normalized': {}
        }
        
        # Find summary files
        summary_files = list(self.results_path.glob("analysis_summary_*.json"))
        for file in summary_files:
            if 'normalized' in file.name:
                files['normalized']['summary'] = file
            elif 'non_normalized' in file.name:
                files['non_normalized']['summary'] = file
        
        # Find individual data type files
        for analysis_type in self.analysis_types:
            files[analysis_type]['data_types'] = {}
            for data_type in self.data_types:
                pattern = f"analysis_data_{data_type}_{analysis_type}_*.json"
                matching_files = list(self.results_path.glob(pattern))
                if matching_files:
                    # Get the most recent file
                    latest_file = max(matching_files, key=lambda x: x.stat().st_mtime)
                    files[analysis_type]['data_types'][data_type] = latest_file
        
        return files
    
    def load_json_data(self, file_path):
        """Load JSON data from file."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def load_pickle_data(self, file_path):
        """Load pickle data from file."""
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def format_p_value(self, p_value):
        """Format p-value with appropriate significance indicators."""
        if p_value is None or np.isnan(p_value):
            return "N/A"
        
        if p_value < 0.001:
            return f"{p_value:.6f} ***"
        elif p_value < 0.01:
            return f"{p_value:.6f} **"
        elif p_value < 0.05:
            return f"{p_value:.6f} *"
        else:
            return f"{p_value:.6f}"
    
    def format_r2(self, r2_value):
        """Format R² value."""
        if r2_value is None or np.isnan(r2_value):
            return "N/A"
        return f"{r2_value:.6f}"
    
    def print_header(self, title):
        """Print formatted header."""
        print("\n" + "="*80)
        print(f" {title}")
        print("="*80)
    
    def print_subheader(self, title):
        """Print formatted subheader."""
        print("\n" + "-"*60)
        print(f" {title}")
        print("-"*60)
    
    def print_summary_table(self, summary_data, analysis_type):
        """Print summary table for all data types."""
        if not summary_data:
            print(f"No summary data found for {analysis_type} analysis.")
            return
        
        print(f"\nSummary Table - {analysis_type.upper()} Analysis")
        print("="*80)
        
        # Create DataFrame for better formatting
        data = []
        for data_type, info in summary_data.get('data_type_summaries', {}).items():
            data.append({
                'Data Type': data_type.replace('_', ' ').title(),
                'N': info.get('N', 'N/A'),
                'k': info.get('k', 'N/A'),
                'Features': info.get('n_features', 'N/A'),
                'PERMANOVA R²': self.format_r2(info.get('permanova_r2')),
                'PERMDISP R²': self.format_r2(info.get('permdisp_r2'))
            })
        
        if data:
            df = pd.DataFrame(data)
            print(df.to_string(index=False))
        else:
            print("No data available for summary table.")
    
    def print_detailed_results(self, data_type, analysis_data, analysis_type):
        """Print detailed results for a specific data type."""
        if not analysis_data:
            print(f"No detailed data found for {data_type} ({analysis_type}).")
            return
        
        print(f"\nDetailed Results - {data_type.replace('_', ' ').title()} ({analysis_type.upper()})")
        print("="*80)
        
        # Metadata
        metadata = analysis_data.get('metadata', {})
        print(f"Data Type: {metadata.get('data_type', 'N/A')}")
        print(f"Analysis Type: {metadata.get('analysis_type', 'N/A')}")
        print(f"Timestamp: {metadata.get('timestamp', 'N/A')}")
        print(f"Total Samples (N): {metadata.get('N', 'N/A')}")
        print(f"Number of Groups (k): {metadata.get('k', 'N/A')}")
        print(f"Number of Features: {metadata.get('n_features', 'N/A')}")
        print(f"Groups: {metadata.get('groups', [])}")
        print(f"Group Counts: {metadata.get('group_counts', [])}")
        
        # Statistical Results
        stats = analysis_data.get('statistical_results', {})
        
        print(f"\nStatistical Results:")
        print("-" * 40)
        
        # PERMANOVA results
        permanova = stats.get('permanova', {})
        print(f"PERMANOVA:")
        print(f"  R²: {self.format_r2(permanova.get('r2'))}")
        print(f"  P-value: {self.format_p_value(permanova.get('p_value'))}")
        print(f"  F-statistic: {permanova.get('F_statistic', 'N/A')}")
        print(f"  Permutations: {permanova.get('permutations', 'N/A')}")
        
        # PERMDISP results
        permdisp = stats.get('permdisp', {})
        print(f"\nPERMDISP:")
        print(f"  R²: {self.format_r2(permdisp.get('r2'))}")
        print(f"  P-value: {self.format_p_value(permdisp.get('p_value'))}")
        print(f"  F-statistic: {permdisp.get('F_statistic', 'N/A')}")
        print(f"  Permutations: {permdisp.get('permutations', 'N/A')}")
        
        # Centroids information
        centroids = analysis_data.get('centroids', {})
        if centroids:
            print(f"\nCentroids Summary:")
            print("-" * 40)
            for group, info in centroids.items():
                print(f"  Group {group}:")
                print(f"    PC1: {info.get('PC1_mean', 'N/A'):.4f} ± {info.get('PC1_std', 'N/A'):.4f}")
                print(f"    PC2: {info.get('PC2_mean', 'N/A'):.4f} ± {info.get('PC2_std', 'N/A'):.4f}")
                print(f"    Samples: {info.get('n_samples', 'N/A')}")
    
    def print_comparison_table(self, files):
        """Print comparison table between normalized and non-normalized results."""
        print("\n" + "="*100)
        print(" COMPARISON TABLE: NORMALIZED vs NON-NORMALIZED RESULTS")
        print("="*100)
        
        comparison_data = []
        
        for data_type in self.data_types:
            row = {'Data Type': data_type.replace('_', ' ').title()}
            
            for analysis_type in self.analysis_types:
                if analysis_type in files and 'data_types' in files[analysis_type]:
                    data_file = files[analysis_type]['data_types'].get(data_type)
                    if data_file:
                        data = self.load_json_data(data_file)
                        if data and 'statistical_results' in data:
                            stats = data['statistical_results']
                            
                            # PERMANOVA
                            permanova = stats.get('permanova', {})
                            row[f'{analysis_type}_permanova_r2'] = self.format_r2(permanova.get('r2'))
                            row[f'{analysis_type}_permanova_p'] = self.format_p_value(permanova.get('p_value'))
                            
                            # PERMDISP
                            permdisp = stats.get('permdisp', {})
                            row[f'{analysis_type}_permdisp_r2'] = self.format_r2(permdisp.get('r2'))
                            row[f'{analysis_type}_permdisp_p'] = self.format_p_value(permdisp.get('p_value'))
                        else:
                            row[f'{analysis_type}_permanova_r2'] = 'N/A'
                            row[f'{analysis_type}_permanova_p'] = 'N/A'
                            row[f'{analysis_type}_permdisp_r2'] = 'N/A'
                            row[f'{analysis_type}_permdisp_p'] = 'N/A'
                    else:
                        row[f'{analysis_type}_permanova_r2'] = 'N/A'
                        row[f'{analysis_type}_permanova_p'] = 'N/A'
                        row[f'{analysis_type}_permdisp_r2'] = 'N/A'
                        row[f'{analysis_type}_permdisp_p'] = 'N/A'
                else:
                    row[f'{analysis_type}_permanova_r2'] = 'N/A'
                    row[f'{analysis_type}_permanova_p'] = 'N/A'
                    row[f'{analysis_type}_permdisp_r2'] = 'N/A'
                    row[f'{analysis_type}_permdisp_p'] = 'N/A'
            
            comparison_data.append(row)
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            
            # Reorder columns for better readability
            columns = ['Data Type']
            for analysis_type in self.analysis_types:
                columns.extend([
                    f'{analysis_type}_permanova_r2',
                    f'{analysis_type}_permanova_p',
                    f'{analysis_type}_permdisp_r2',
                    f'{analysis_type}_permdisp_p'
                ])
            
            df = df[columns]
            
            # Rename columns for display
            df.columns = [
                'Data Type',
                'Norm PERMANOVA R²', 'Norm PERMANOVA P', 'Norm PERMDISP R²', 'Norm PERMDISP P',
                'Non-Norm PERMANOVA R²', 'Non-Norm PERMANOVA P', 'Non-Norm PERMDISP R²', 'Non-Norm PERMDISP P'
            ]
            
            print(df.to_string(index=False))
        else:
            print("No comparison data available.")
    
    def print_significance_summary(self, files):
        """Print summary of significant results."""
        print("\n" + "="*80)
        print(" SIGNIFICANCE SUMMARY")
        print("="*80)
        
        significance_data = {
            'normalized': {'permanova': [], 'permdisp': []},
            'non_normalized': {'permanova': [], 'permdisp': []}
        }
        
        for analysis_type in self.analysis_types:
            if analysis_type in files and 'data_types' in files[analysis_type]:
                for data_type in self.data_types:
                    data_file = files[analysis_type]['data_types'].get(data_type)
                    if data_file:
                        data = self.load_json_data(data_file)
                        if data and 'statistical_results' in data:
                            stats = data['statistical_results']
                            
                            # Check PERMANOVA significance
                            permanova_p = stats.get('permanova', {}).get('p_value')
                            if permanova_p is not None and permanova_p < 0.05:
                                significance_data[analysis_type]['permanova'].append({
                                    'data_type': data_type,
                                    'p_value': permanova_p,
                                    'r2': stats.get('permanova', {}).get('r2')
                                })
                            
                            # Check PERMDISP significance
                            permdisp_p = stats.get('permdisp', {}).get('p_value')
                            if permdisp_p is not None and permdisp_p < 0.05:
                                significance_data[analysis_type]['permdisp'].append({
                                    'data_type': data_type,
                                    'p_value': permdisp_p,
                                    'r2': stats.get('permdisp', {}).get('r2')
                                })
        
        for analysis_type in self.analysis_types:
            print(f"\n{analysis_type.upper()} Analysis:")
            print("-" * 40)
            
            # PERMANOVA significant results
            permanova_sig = significance_data[analysis_type]['permanova']
            if permanova_sig:
                print(f"Significant PERMANOVA results (p < 0.05):")
                for result in permanova_sig:
                    print(f"  {result['data_type'].replace('_', ' ').title()}: p={result['p_value']:.6f}, R²={result['r2']:.6f}")
            else:
                print("No significant PERMANOVA results found.")
            
            # PERMDISP significant results
            permdisp_sig = significance_data[analysis_type]['permdisp']
            if permdisp_sig:
                print(f"\nSignificant PERMDISP results (p < 0.05):")
                for result in permdisp_sig:
                    print(f"  {result['data_type'].replace('_', ' ').title()}: p={result['p_value']:.6f}, R²={result['r2']:.6f}")
            else:
                print("No significant PERMDISP results found.")
    
    def print_all_results(self):
        """Print all covariate shift results."""
        print("\n" + "="*100)
        print(" COVARIATE SHIFT ANALYSIS RESULTS")
        print("="*100)
        print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Results directory: {self.results_path}")
        
        # Find all analysis files
        files = self.find_analysis_files()
        
        # Print summary tables
        for analysis_type in self.analysis_types:
            if analysis_type in files and 'summary' in files[analysis_type]:
                summary_file = files[analysis_type]['summary']
                summary_data = self.load_json_data(summary_file)
                self.print_summary_table(summary_data, analysis_type)
        
        # Print detailed results for each data type
        for analysis_type in self.analysis_types:
            if analysis_type in files and 'data_types' in files[analysis_type]:
                self.print_subheader(f"DETAILED RESULTS - {analysis_type.upper()}")
                
                for data_type in self.data_types:
                    data_file = files[analysis_type]['data_types'].get(data_type)
                    if data_file:
                        data = self.load_json_data(data_file)
                        self.print_detailed_results(data_type, data, analysis_type)
        
        # Print comparison table
        self.print_comparison_table(files)
        
        # Print significance summary
        self.print_significance_summary(files)
        
        # Print footer
        print("\n" + "="*100)
        print(" END OF COVARIATE SHIFT ANALYSIS RESULTS")
        print("="*100)
        print("\nLegend:")
        print("  *** p < 0.001 (highly significant)")
        print("  **  p < 0.01 (very significant)")
        print("  *   p < 0.05 (significant)")
        print("  N/A Not available or not computed")

def main():
    """Main function to run the results printer."""
    printer = CovariateResultsPrinter()
    printer.print_all_results()

if __name__ == "__main__":
    main() 
