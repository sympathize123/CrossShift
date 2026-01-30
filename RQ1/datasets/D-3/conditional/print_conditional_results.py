#!/usr/bin/env python3
"""
print_conditional_results.py - Conditional Shift Results Printer
----------------------------------------------------------------
Script to print and format conditional shift analysis results from JSON files.
Handles both normalized and non-normalized data, with separate results for each label (0.0 and 1.0).
"""
import os
import sys
import json
import pandas as pd
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

class ConditionalShiftResultsPrinter:
    """Print and format conditional shift analysis results."""
    
    def __init__(self):
        self.results_path = Path(str(RESULTS_ROOT / "RQ1/ConditionalShift"))
        self.data_types = ['phone_usage', 'mobility', 'physical_status', 'sleep', 'social_behavior']
        self.labels = [0.0, 1.0]
        self.analysis_types = ['normalized', 'non_normalized']
        
    def load_summary_files(self):
        """Load all summary files from the results directory."""
        summary_files = list(self.results_path.glob("analysis_summary_*.json"))
        if not summary_files:
            print("No summary files found!")
            return {}
        
        summaries = {}
        for file_path in summary_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    analysis_type = data['metadata']['analysis_type']
                    timestamp = data['metadata']['timestamp']
                    summaries[f"{analysis_type}_{timestamp}"] = {
                        'file_path': file_path,
                        'data': data
                    }
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        return summaries
    
    def load_detailed_files(self, data_type, analysis_type):
        """Load detailed analysis files for a specific data type and analysis type."""
        pattern = f"analysis_data_{data_type}_{analysis_type}_label_*.json"
        detailed_files = list(self.results_path.glob(pattern))
        
        detailed_data = {}
        for file_path in detailed_files:
            try:
                # Extract label from filename
                filename = file_path.name
                if "label_0.0" in filename:
                    label = 0.0
                elif "label_1.0" in filename:
                    label = 1.0
                else:
                    continue
                
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    detailed_data[label] = data
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        return detailed_data
    
    def format_r2_value(self, value):
        """Format R² value for display."""
        if value is None or pd.isna(value):
            return "N/A"
        return f"{value:.6f}"
    
    def format_p_value(self, value):
        """Format p-value for display."""
        if value is None or pd.isna(value):
            return "N/A"
        if value < 0.001:
            return "< 0.001"
        return f"{value:.3f}"
    
    def print_summary_table(self, summaries):
        """Print summary table of all conditional shift results."""
        print("\n" + "="*120)
        print(" CONDITIONAL SHIFT ANALYSIS - SUMMARY TABLE")
        print("="*120)
        
        for analysis_key, summary_info in summaries.items():
            data = summary_info['data']
            analysis_type = data['metadata']['analysis_type']
            timestamp = data['metadata']['timestamp']
            
            print(f"\nAnalysis Type: {analysis_type.upper()}")
            print(f"Timestamp: {timestamp}")
            print("-" * 120)
            print(f"{'Data Type':<20} {'Label':<8} {'N':<8} {'k':<6} {'Features':<10} {'PERMANOVA R²':<15} {'PERMANOVA p':<12} {'PERMDISP R²':<15} {'PERMDISP p':<12}")
            print("-" * 120)
            
            for data_type in self.data_types:
                if data_type in data['data_type_summaries']:
                    type_data = data['data_type_summaries'][data_type]
                    
                    # Load detailed data for p-values
                    detailed_data = self.load_detailed_files(data_type, analysis_type)
                    
                    for label_key, label_data in type_data.items():
                        label = label_key.replace('label_', '')
                        permanova_r2 = self.format_r2_value(label_data.get('permanova_r2'))
                        permdisp_r2 = self.format_r2_value(label_data.get('permdisp_r2'))
                        
                        # Get p-values from detailed data
                        label_float = float(label)
                        if label_float in detailed_data:
                            stats = detailed_data[label_float]['statistical_results']
                            permanova_p = self.format_p_value(stats['permanova']['p_value'])
                            permdisp_p = self.format_p_value(stats['permdisp']['p_value'])
                        else:
                            permanova_p = "N/A"
                            permdisp_p = "N/A"
                        
                        print(f"{data_type:<20} {label:<8} {label_data['N']:<8} {label_data['k']:<6} "
                              f"{label_data['n_features']:<10} {permanova_r2:<15} {permanova_p:<12} {permdisp_r2:<15} {permdisp_p:<12}")
            
            print("-" * 120)
    
    def print_detailed_results(self, summaries):
        """Print detailed results for each data type and label."""
        print("\n" + "="*120)
        print(" CONDITIONAL SHIFT ANALYSIS - DETAILED RESULTS")
        print("="*120)
        
        for analysis_key, summary_info in summaries.items():
            data = summary_info['data']
            analysis_type = data['metadata']['analysis_type']
            timestamp = data['metadata']['timestamp']
            
            print(f"\n{'='*60}")
            print(f" ANALYSIS TYPE: {analysis_type.upper()}")
            print(f" TIMESTAMP: {timestamp}")
            print("="*60)
            
            for data_type in self.data_types:
                if data_type in data['data_type_summaries']:
                    print(f"\n{data_type.upper().replace('_', ' ')}:")
                    print("-" * 60)
                    
                    # Load detailed data
                    detailed_data = self.load_detailed_files(data_type, analysis_type)
                    
                    for label in self.labels:
                        if label in detailed_data:
                            label_data = detailed_data[label]
                            stats = label_data['statistical_results']
                            
                            print(f"\nLabel {int(label)} (Stress = {int(label)}):")
                            print(f"  N: {label_data['metadata']['N']}")
                            print(f"  k: {label_data['metadata']['k']}")
                            print(f"  Features: {label_data['metadata']['n_features']}")
                            print(f"  Features used: {label_data['metadata']['features_used']}")
                            
                            print(f"\n  PERMANOVA:")
                            permanova = stats['permanova']
                            print(f"    R²: {self.format_r2_value(permanova['r2'])}")
                            print(f"    P-value: {self.format_p_value(permanova['p_value'])}")
                            print(f"    F-statistic: {permanova['F_statistic']:.6f}" if permanova['F_statistic'] else "    F-statistic: N/A")
                            
                            print(f"\n  PERMDISP:")
                            permdisp = stats['permdisp']
                            print(f"    R²: {self.format_r2_value(permdisp['r2'])}")
                            print(f"    P-value: {self.format_p_value(permdisp['p_value'])}")
                            print(f"    F-statistic: {permdisp['F_statistic']:.6f}" if permdisp['F_statistic'] else "    F-statistic: N/A")
                            
                            if label_data['centroids']:
                                print(f"\n  Centroids (first 3 groups):")
                                for i, (group, info) in enumerate(list(label_data['centroids'].items())[:3]):
                                    print(f"    Group {group}: PC1={info['PC1_mean']:.4f}±{info['PC1_std']:.4f}, "
                                          f"PC2={info['PC2_mean']:.4f}±{info['PC2_std']:.4f}, n={info['n_samples']}")
    
    def print_comparison_table(self, summaries):
        """Print comparison table between labels for each data type."""
        print("\n" + "="*120)
        print(" CONDITIONAL SHIFT ANALYSIS - LABEL COMPARISON")
        print("="*120)
        
        for analysis_key, summary_info in summaries.items():
            data = summary_info['data']
            analysis_type = data['metadata']['analysis_type']
            timestamp = data['metadata']['timestamp']
            
            print(f"\nAnalysis Type: {analysis_type.upper()}")
            print(f"Timestamp: {timestamp}")
            print("-" * 120)
            print(f"{'Data Type':<20} {'Metric':<15} {'Label 0':<15} {'Label 1':<15} {'Difference':<15} {'Ratio':<10}")
            print("-" * 120)
            
            for data_type in self.data_types:
                if data_type in data['data_type_summaries']:
                    type_data = data['data_type_summaries'][data_type]
                    
                    if 'label_0' in type_data and 'label_1' in type_data:
                        label_0 = type_data['label_0']
                        label_1 = type_data['label_1']
                        
                        # Sample size comparison
                        n_diff = label_0['N'] - label_1['N']
                        n_ratio = label_0['N'] / label_1['N'] if label_1['N'] > 0 else float('inf')
                        print(f"{data_type:<20} {'N':<15} {label_0['N']:<15} {label_1['N']:<15} {n_diff:<15} {n_ratio:.2f}")
                        
                        # Load detailed data for p-values
                        detailed_data = self.load_detailed_files(data_type, analysis_type)
                        
                        # PERMANOVA R² comparison
                        permanova_0 = label_0.get('permanova_r2')
                        permanova_1 = label_1.get('permanova_r2')
                        if permanova_0 is not None and permanova_1 is not None:
                            perm_diff = permanova_0 - permanova_1
                            perm_ratio = permanova_0 / permanova_1 if permanova_1 > 0 else float('inf')
                            print(f"{'':<20} {'PERMANOVA R²':<15} {self.format_r2_value(permanova_0):<15} "
                                  f"{self.format_r2_value(permanova_1):<15} {perm_diff:.6f} {perm_ratio:.2f}")
                        
                        # PERMANOVA p-value comparison
                        if 0.0 in detailed_data and 1.0 in detailed_data:
                            perm_p_0 = detailed_data[0.0]['statistical_results']['permanova']['p_value']
                            perm_p_1 = detailed_data[1.0]['statistical_results']['permanova']['p_value']
                            if perm_p_0 is not None and perm_p_1 is not None:
                                perm_p_diff = perm_p_0 - perm_p_1
                                perm_p_ratio = perm_p_0 / perm_p_1 if perm_p_1 > 0 else float('inf')
                                print(f"{'':<20} {'PERMANOVA p':<15} {self.format_p_value(perm_p_0):<15} "
                                      f"{self.format_p_value(perm_p_1):<15} {perm_p_diff:.6f} {perm_p_ratio:.2f}")
                        
                        # PERMDISP R² comparison
                        permdisp_0 = label_0.get('permdisp_r2')
                        permdisp_1 = label_1.get('permdisp_r2')
                        if permdisp_0 is not None and permdisp_1 is not None:
                            disp_diff = permdisp_0 - permdisp_1
                            disp_ratio = permdisp_0 / permdisp_1 if permdisp_1 > 0 else float('inf')
                            print(f"{'':<20} {'PERMDISP R²':<15} {self.format_r2_value(permdisp_0):<15} "
                                  f"{self.format_r2_value(permdisp_1):<15} {disp_diff:.6f} {disp_ratio:.2f}")
                        
                        # PERMDISP p-value comparison
                        if 0.0 in detailed_data and 1.0 in detailed_data:
                            disp_p_0 = detailed_data[0.0]['statistical_results']['permdisp']['p_value']
                            disp_p_1 = detailed_data[1.0]['statistical_results']['permdisp']['p_value']
                            if disp_p_0 is not None and disp_p_1 is not None:
                                disp_p_diff = disp_p_0 - disp_p_1
                                disp_p_ratio = disp_p_0 / disp_p_1 if disp_p_1 > 0 else float('inf')
                                print(f"{'':<20} {'PERMDISP p':<15} {self.format_p_value(disp_p_0):<15} "
                                      f"{self.format_p_value(disp_p_1):<15} {disp_p_diff:.6f} {disp_p_ratio:.2f}")
                        
                        print("-" * 120)
    
    def print_significance_table(self, summaries):
        """Print significance table showing which results are statistically significant."""
        print("\n" + "="*120)
        print(" CONDITIONAL SHIFT ANALYSIS - STATISTICAL SIGNIFICANCE")
        print("="*120)
        
        significance_levels = {
            'p < 0.001': '***',
            'p < 0.01': '**',
            'p < 0.05': '*',
            'p >= 0.05': 'ns'
        }
        
        for analysis_key, summary_info in summaries.items():
            data = summary_info['data']
            analysis_type = data['metadata']['analysis_type']
            timestamp = data['metadata']['timestamp']
            
            print(f"\nAnalysis Type: {analysis_type.upper()}")
            print(f"Timestamp: {timestamp}")
            print("-" * 120)
            print(f"{'Data Type':<20} {'Label':<8} {'PERMANOVA':<15} {'PERMDISP':<15}")
            print("-" * 120)
            
            for data_type in self.data_types:
                if data_type in data['data_type_summaries']:
                    type_data = data['data_type_summaries'][data_type]
                    
                    for label_key, label_data in type_data.items():
                        label = label_key.replace('label_', '')
                        
                        # Load detailed data for p-values
                        detailed_data = self.load_detailed_files(data_type, analysis_type)
                        if float(label) in detailed_data:
                            stats = detailed_data[float(label)]['statistical_results']
                            
                            # PERMANOVA significance
                            permanova_p = stats['permanova']['p_value']
                            if permanova_p is not None:
                                if permanova_p < 0.001:
                                    perm_sig = '***'
                                elif permanova_p < 0.01:
                                    perm_sig = '**'
                                elif permanova_p < 0.05:
                                    perm_sig = '*'
                                else:
                                    perm_sig = 'ns'
                            else:
                                perm_sig = 'N/A'
                            
                            # PERMDISP significance
                            permdisp_p = stats['permdisp']['p_value']
                            if permdisp_p is not None:
                                if permdisp_p < 0.001:
                                    disp_sig = '***'
                                elif permdisp_p < 0.01:
                                    disp_sig = '**'
                                elif permdisp_p < 0.05:
                                    disp_sig = '*'
                                else:
                                    disp_sig = 'ns'
                            else:
                                disp_sig = 'N/A'
                            
                            print(f"{data_type:<20} {label:<8} {perm_sig:<15} {disp_sig:<15}")
            
            print("-" * 120)
            print("Legend: *** p < 0.001, ** p < 0.01, * p < 0.05, ns p >= 0.05, N/A = Not Available")
    
    def print_all_results(self):
        """Print all conditional shift results in different formats."""
        print("Loading conditional shift results...")
        summaries = self.load_summary_files()
        
        if not summaries:
            print("No summary files found in the results directory.")
            return
        
        print(f"Found {len(summaries)} summary file(s):")
        for key in summaries.keys():
            print(f"  - {key}")
        
        # Print different types of tables
        self.print_summary_table(summaries)
        self.print_detailed_results(summaries)
        self.print_comparison_table(summaries)
        self.print_significance_table(summaries)
        
        print(f"\n" + "="*120)
        print(" CONDITIONAL SHIFT RESULTS PRINTING COMPLETE")
        print("="*120)
        print("Results include:")
        print("- Summary table with all data types and labels")
        print("- Detailed results with statistical values")
        print("- Label comparison showing differences between stress levels")
        print("- Statistical significance indicators")
        print("="*120)

def main():
    """Main function to print conditional shift results."""
    printer = ConditionalShiftResultsPrinter()
    printer.print_all_results()

if __name__ == "__main__":
    main() 
