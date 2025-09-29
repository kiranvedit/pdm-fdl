"""
Statistical Analysis Module for Federated Learning Experiments

This module contains functions for statistical analysis and comparison of federated learning
experiments, including aggregation, significance testing, and performance comparison.

Functions:
- compute_statistical_aggregation: Aggregate results from multiple experimental runs
- display_aggregated_results_table: Display aggregated results in a formatted table
- run_final_statistical_analysis: Comprehensive statistical analysis of results
- detailed_individual_ttest_analysis: Individual t-test analysis for each configuration
"""

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import warnings
warnings.filterwarnings('ignore')


def compute_statistical_aggregation(all_runs_results, run_summaries):
    """
    Compute statistical aggregation across multiple experimental runs.
    
    Args:
        all_runs_results: Dictionary containing results from all runs
        run_summaries: List of run summary statistics
    
    Returns:
        dict: Aggregated results with means, standard deviations, and statistical summaries
    """
    print("="*80)
    print("COMPUTING STATISTICAL AGGREGATION ACROSS ALL RUNS")
    print("="*80)
    
    # Organize data by configuration
    config_data = {}
    
    for run_name, run_data in all_runs_results.items():
        for exp_key, exp_result in run_data.items():
            if exp_result.get('status') == 'success':
                # Parse configuration
                parts = exp_key.split('_')
                if len(parts) >= 4:
                    model = parts[0]
                    algorithm = parts[1]
                    distribution = parts[2]
                    server_type = parts[3]
                    
                    config_key = f"{model}_{algorithm}_{distribution}"
                    
                    if config_key not in config_data:
                        config_data[config_key] = {
                            'standard': {'accuracy': [], 'f1': [], 'precision': [], 'recall': [], 'auc': [], 'training_time': [], 'inference_time': []},
                            'secure': {'accuracy': [], 'f1': [], 'precision': [], 'recall': [], 'auc': [], 'training_time': [], 'inference_time': []}
                        }
                    
                    # Extract metrics with fallback for missing timing metrics
                    accuracy = exp_result.get('final_accuracy', 0.0)
                    f1 = exp_result.get('final_f1', 0.0)
                    precision = exp_result.get('precision', 0.0)
                    recall = exp_result.get('recall', 0.0)
                    auc = exp_result.get('auc', 0.0)
                    training_time = exp_result.get('training_time', 0.0)
                    inference_time = exp_result.get('inference_time', 0.0)
                    
                    # Store data
                    config_data[config_key][server_type]['accuracy'].append(accuracy)
                    config_data[config_key][server_type]['f1'].append(f1)
                    config_data[config_key][server_type]['precision'].append(precision)
                    config_data[config_key][server_type]['recall'].append(recall)
                    config_data[config_key][server_type]['auc'].append(auc)
                    config_data[config_key][server_type]['training_time'].append(training_time)
                    config_data[config_key][server_type]['inference_time'].append(inference_time)
    
    # Compute aggregated statistics
    aggregated_results = {}
    
    for config_key, server_data in config_data.items():
        aggregated_results[config_key] = {}
        
        for server_type in ['standard', 'secure']:
            if server_data[server_type]['accuracy']:  # If data exists
                metrics = {}
                
                for metric_name in ['accuracy', 'f1', 'precision', 'recall', 'auc', 'training_time', 'inference_time']:
                    values = server_data[server_type][metric_name]
                    if values:  # Only compute if values exist
                        metrics[metric_name] = {
                            'mean': np.mean(values),
                            'std': np.std(values, ddof=1) if len(values) > 1 else 0.0,
                            'min': np.min(values),
                            'max': np.max(values),
                            'count': len(values)
                        }
                    else:
                        # Default values for missing metrics
                        metrics[metric_name] = {
                            'mean': 0.0,
                            'std': 0.0,
                            'min': 0.0,
                            'max': 0.0,
                            'count': 0
                        }
                
                aggregated_results[config_key][server_type] = metrics
    
    # Print summary
    print(f"Aggregated results for {len(aggregated_results)} configurations")
    print(f"Available configurations:")
    for config in sorted(aggregated_results.keys()):
        standard_count = aggregated_results[config].get('standard', {}).get('accuracy', {}).get('count', 0)
        secure_count = aggregated_results[config].get('secure', {}).get('accuracy', {}).get('count', 0)
        print(f"  {config}: {standard_count} standard runs, {secure_count} secure runs")
    
    return aggregated_results


def display_aggregated_results_table(aggregated_results):
    """
    Display aggregated results in a well-formatted table.
    
    Args:
        aggregated_results: Results from compute_statistical_aggregation
    
    Returns:
        pd.DataFrame: Formatted results table
    """
    import pandas as pd
    
    print("\n" + "="*120)
    print("FINAL AGGREGATED RESULTS TABLE - STATISTICAL SUMMARY")
    print("="*120)
    
    # Prepare data for table
    table_data = []
    
    for config_key in sorted(aggregated_results.keys()):
        parts = config_key.split('_')
        model = parts[0]
        algorithm = parts[1]
        distribution = parts[2]
        
        config_data = aggregated_results[config_key]
        
        # Standard results
        if 'standard' in config_data:
            std_data = config_data['standard']
            table_data.append({
                'Model': model.upper(),
                'Algorithm': algorithm.upper(),
                'Distribution': distribution.upper(),
                'Server_Type': 'STANDARD',
                'Accuracy_Mean': f"{std_data['accuracy']['mean']:.4f}",
                'Accuracy_Std': f"{std_data['accuracy']['std']:.4f}",
                'F1_Mean': f"{std_data['f1']['mean']:.4f}",
                'F1_Std': f"{std_data['f1']['std']:.4f}",
                'Precision_Mean': f"{std_data['precision']['mean']:.4f}",
                'Recall_Mean': f"{std_data['recall']['mean']:.4f}",
                'AUC_Mean': f"{std_data['auc']['mean']:.4f}",
                'Training_Time_Mean': f"{std_data['training_time']['mean']:.2f}s",
                'Inference_Time_Mean': f"{std_data['inference_time']['mean']:.4f}s",
                'Runs': std_data['accuracy']['count']
            })
        
        # Secure results  
        if 'secure' in config_data:
            sec_data = config_data['secure']
            table_data.append({
                'Model': model.upper(),
                'Algorithm': algorithm.upper(),
                'Distribution': distribution.upper(),
                'Server_Type': 'SECURE',
                'Accuracy_Mean': f"{sec_data['accuracy']['mean']:.4f}",
                'Accuracy_Std': f"{sec_data['accuracy']['std']:.4f}",
                'F1_Mean': f"{sec_data['f1']['mean']:.4f}",
                'F1_Std': f"{sec_data['f1']['std']:.4f}",
                'Precision_Mean': f"{sec_data['precision']['mean']:.4f}",
                'Recall_Mean': f"{sec_data['recall']['mean']:.4f}",
                'AUC_Mean': f"{sec_data['auc']['mean']:.4f}",
                'Training_Time_Mean': f"{sec_data['training_time']['mean']:.2f}s",
                'Inference_Time_Mean': f"{sec_data['inference_time']['mean']:.4f}s",
                'Runs': sec_data['accuracy']['count']
            })
    
    # Create DataFrame
    df_aggregated = pd.DataFrame(table_data)
    
    # Set pandas display options for better formatting
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 15)
    
    # Display table
    print(df_aggregated.to_string(index=False, max_cols=None, max_colwidth=15))
    
    # Reset options
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.max_colwidth')
    
    return df_aggregated


def run_final_statistical_analysis(aggregated_results):
    """
    Run comprehensive statistical analysis on aggregated results from multiple runs.
    
    Args:
        aggregated_results: Aggregated results from compute_statistical_aggregation
    """
    print("FINAL STATISTICAL ANALYSIS - AGGREGATED RESULTS")
    print("="*80)
    
    # Prepare data for analysis
    standard_accuracies = []
    secure_accuracies = []
    standard_f1_scores = []
    secure_f1_scores = []
    
    configurations = []
    
    for config_key, server_data in aggregated_results.items():
        if 'standard' in server_data and 'secure' in server_data:
            # Both standard and secure data available
            standard_acc = server_data['standard']['accuracy']['mean']
            secure_acc = server_data['secure']['accuracy']['mean']
            standard_f1 = server_data['standard']['f1']['mean']
            secure_f1 = server_data['secure']['f1']['mean']
            
            standard_accuracies.append(standard_acc)
            secure_accuracies.append(secure_acc)
            standard_f1_scores.append(standard_f1)
            secure_f1_scores.append(secure_f1)
            configurations.append(config_key)
    
    print(f"Configurations with both standard and secure data: {len(configurations)}")
    
    # Overall comparison
    if standard_accuracies and secure_accuracies:
        print(f"\nOVERALL PERFORMANCE COMPARISON (Aggregated from multiple runs)")
        print("-" * 60)
        
        std_acc_mean = np.mean(standard_accuracies)
        std_acc_std = np.std(standard_accuracies)
        sec_acc_mean = np.mean(secure_accuracies)
        sec_acc_std = np.std(secure_accuracies)
        
        std_f1_mean = np.mean(standard_f1_scores)
        std_f1_std = np.std(standard_f1_scores)
        sec_f1_mean = np.mean(secure_f1_scores)
        sec_f1_std = np.std(secure_f1_scores)
        
        print(f"Standard Federated Learning:")
        print(f"  Mean Accuracy: {std_acc_mean:.6f} ± {std_acc_std:.6f}")
        print(f"  Mean F1-Score: {std_f1_mean:.6f} ± {std_f1_std:.6f}")
        
        print(f"\nSecure Federated Learning:")
        print(f"  Mean Accuracy: {sec_acc_mean:.6f} ± {sec_acc_std:.6f}")
        print(f"  Mean F1-Score: {sec_f1_mean:.6f} ± {sec_f1_std:.6f}")
        
        # Performance impact
        acc_impact = ((std_acc_mean - sec_acc_mean) / std_acc_mean) * 100
        f1_impact = ((std_f1_mean - sec_f1_mean) / std_f1_mean) * 100
        
        print(f"\nSecurity Impact Analysis:")
        print(f"  Accuracy degradation: {acc_impact:.2f}%")
        print(f"  F1-Score degradation: {f1_impact:.2f}%")
        
        # Statistical significance testing
        t_stat_acc, p_val_acc = ttest_ind(standard_accuracies, secure_accuracies)
        t_stat_f1, p_val_f1 = ttest_ind(standard_f1_scores, secure_f1_scores)
        
        print(f"\nStatistical Significance Testing:")
        print(f"  Accuracy - t-statistic: {t_stat_acc:.4f}, p-value: {p_val_acc:.6f}")
        print(f"  F1-Score - t-statistic: {t_stat_f1:.4f}, p-value: {p_val_f1:.6f}")
        print(f"  Accuracy difference is {'significant' if p_val_acc < 0.05 else 'not significant'} (α=0.05)")
        print(f"  F1-Score difference is {'significant' if p_val_f1 < 0.05 else 'not significant'} (α=0.05)")
        
        # Effect size (Cohen's d)
        def cohens_d(group1, group2):
            n1, n2 = len(group1), len(group2)
            s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
            pooled_std = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
            return (np.mean(group1) - np.mean(group2)) / pooled_std
        
        effect_size_acc = cohens_d(standard_accuracies, secure_accuracies)
        effect_size_f1 = cohens_d(standard_f1_scores, secure_f1_scores)
        
        def effect_size_interpretation(d):
            abs_d = abs(d)
            if abs_d < 0.2:
                return "negligible"
            elif abs_d < 0.5:
                return "small"
            elif abs_d < 0.8:
                return "medium"
            else:
                return "large"
        
        print(f"\nEffect Size Analysis (Cohen's d):")
        print(f"  Accuracy effect size: {effect_size_acc:.4f} ({effect_size_interpretation(effect_size_acc)})")
        print(f"  F1-Score effect size: {effect_size_f1:.4f} ({effect_size_interpretation(effect_size_f1)})")
        
        return {
            'standard_accuracy': {'mean': std_acc_mean, 'std': std_acc_std},
            'secure_accuracy': {'mean': sec_acc_mean, 'std': sec_acc_std},
            'standard_f1': {'mean': std_f1_mean, 'std': std_f1_std},
            'secure_f1': {'mean': sec_f1_mean, 'std': sec_f1_std},
            'accuracy_degradation': acc_impact,
            'f1_degradation': f1_impact,
            'significance_tests': {
                'accuracy': {'t_stat': t_stat_acc, 'p_value': p_val_acc},
                'f1': {'t_stat': t_stat_f1, 'p_value': p_val_f1}
            },
            'effect_sizes': {
                'accuracy': effect_size_acc,
                'f1': effect_size_f1
            }
        }
    else:
        print("❌ Insufficient data for statistical analysis")
        return None


def detailed_individual_ttest_analysis(final_results):
    """
    Perform detailed t-test analysis for each experiment configuration comparing
    standard vs secure federated learning.
    
    Args:
        final_results: Dictionary containing experimental results
    
    Returns:
        pd.DataFrame: T-test results for each configuration
    """
    print("="*100)
    print("DETAILED T-TEST ANALYSIS - INDIVIDUAL EXPERIMENT CONFIGURATIONS")  
    print("="*100)
    print("Comparing Standard vs Secure for each of the 24 experiment combinations")
    print("(3 models × 4 algorithms × 2 distributions = 24 combinations)")
    print("="*100)
    
    # Collect all individual results
    all_individual_results = []
    
    for run_name, run_data in final_results.items():
        for exp_key, exp_result in run_data.items():
            if exp_result.get('status') == 'success':
                # Handle potential missing timing metrics gracefully
                training_time = exp_result.get('training_time', 0.0)
                inference_time = exp_result.get('inference_time', 0.0)
                
                all_individual_results.append({
                    'run': run_name,
                    'experiment': exp_key,
                    'model': exp_result['model'],
                    'algorithm': exp_result['algorithm'], 
                    'distribution': exp_result['distribution'],
                    'server_type': exp_result['server_type'],
                    'accuracy': exp_result['final_accuracy'],
                    'f1_score': exp_result['final_f1'],
                    'precision': exp_result['precision'],
                    'recall': exp_result['recall'],
                    'auc': exp_result['auc'],
                    'training_time': training_time,
                    'inference_time': inference_time
                })
    
    # Convert to DataFrame for easier analysis
    df_individual = pd.DataFrame(all_individual_results)
    
    if df_individual.empty:
        print("❌ No successful experiments found for analysis")
        return pd.DataFrame()
    
    print(f"✅ Found {len(df_individual)} successful experiment results")
    print(f"   Standard experiments: {len(df_individual[df_individual['server_type'] == 'standard'])}")
    print(f"   Secure experiments: {len(df_individual[df_individual['server_type'] == 'secure'])}")
    
    # Group by configuration (model + algorithm + distribution)
    ttest_results = []
    
    # Available metrics for comparison
    metrics = ['accuracy', 'f1_score', 'precision', 'recall', 'auc']
    
    # Check if timing metrics are available
    if df_individual['training_time'].sum() > 0:
        metrics.append('training_time')
    if df_individual['inference_time'].sum() > 0:
        metrics.append('inference_time')
    
    print(f"\nAnalyzing metrics: {metrics}")
    
    for model in df_individual['model'].unique():
        for algorithm in df_individual['algorithm'].unique():
            for distribution in df_individual['distribution'].unique():
                
                # Filter data for this configuration
                config_mask = ((df_individual['model'] == model) & 
                             (df_individual['algorithm'] == algorithm) & 
                             (df_individual['distribution'] == distribution))
                
                config_data = df_individual[config_mask]
                
                if len(config_data) == 0:
                    continue
                
                standard_data = config_data[config_data['server_type'] == 'standard']
                secure_data = config_data[config_data['server_type'] == 'secure']
                
                if len(standard_data) == 0 or len(secure_data) == 0:
                    print(f"⚠️  Skipping {model}-{algorithm}-{distribution}: Missing standard or secure data")
                    continue
                
                config_name = f"{model}-{algorithm}-{distribution}"
                print(f"\n📊 Analyzing: {config_name}")
                print(f"   Standard runs: {len(standard_data)}, Secure runs: {len(secure_data)}")
                
                # Perform t-tests for each metric
                for metric in metrics:
                    standard_vals = standard_data[metric].values
                    secure_vals = secure_data[metric].values
                    
                    # Skip if insufficient data
                    if len(standard_vals) < 2 or len(secure_vals) < 2:
                        continue
                    
                    # Perform t-test
                    t_stat, p_value = ttest_ind(standard_vals, secure_vals)
                    
                    # Calculate effect size (Cohen's d)
                    pooled_std = np.sqrt(((len(standard_vals)-1)*np.var(standard_vals, ddof=1) + 
                                        (len(secure_vals)-1)*np.var(secure_vals, ddof=1)) / 
                                       (len(standard_vals)+len(secure_vals)-2))
                    
                    cohens_d = (np.mean(standard_vals) - np.mean(secure_vals)) / pooled_std if pooled_std > 0 else 0
                    
                    # Performance degradation
                    standard_mean = np.mean(standard_vals)
                    secure_mean = np.mean(secure_vals)
                    
                    if metric in ['training_time', 'inference_time']:
                        # For timing metrics, higher is worse (degradation is positive when secure is slower)
                        degradation = ((secure_mean - standard_mean) / standard_mean) * 100 if standard_mean > 0 else 0
                    else:
                        # For performance metrics, lower is worse (degradation is positive when secure is lower)
                        degradation = ((standard_mean - secure_mean) / standard_mean) * 100 if standard_mean > 0 else 0
                    
                    # Effect size interpretation
                    def effect_size_desc(d):
                        abs_d = abs(d)
                        if abs_d < 0.2: return "negligible"
                        elif abs_d < 0.5: return "small"
                        elif abs_d < 0.8: return "medium"
                        else: return "large"
                    
                    ttest_results.append({
                        'configuration': config_name,
                        'model': model,
                        'algorithm': algorithm,
                        'distribution': distribution,
                        'metric': metric,
                        'standard_mean': standard_mean,
                        'standard_std': np.std(standard_vals, ddof=1),
                        'secure_mean': secure_mean,
                        'secure_std': np.std(secure_vals, ddof=1),
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'is_significant': 'Yes' if p_value < 0.05 else 'No',
                        'effect_size': cohens_d,
                        'effect_size_desc': effect_size_desc(cohens_d),
                        'degradation': degradation,
                        'standard_n': len(standard_vals),
                        'secure_n': len(secure_vals)
                    })
    
    # Convert to DataFrame
    df_ttests = pd.DataFrame(ttest_results)
    
    if df_ttests.empty:
        print("❌ No valid t-test comparisons could be performed")
        return df_ttests
    
    # Summary statistics
    print(f"\n📈 T-TEST ANALYSIS SUMMARY")
    print("="*50)
    
    total_comparisons = len(df_ttests)
    significant_results = df_ttests[df_ttests['is_significant'] == 'Yes']
    total_significant = len(significant_results)
    
    print(f"Total comparisons performed: {total_comparisons}")
    print(f"Statistically significant results: {total_significant} ({(total_significant/total_comparisons)*100:.1f}%)")
    
    # Significance by metric
    print(f"\nSignificance rate by metric:")
    for metric in df_ttests['metric'].unique():
        metric_data = df_ttests[df_ttests['metric'] == metric]
        metric_significant = len(metric_data[metric_data['is_significant'] == 'Yes'])
        metric_total = len(metric_data)
        print(f"  {metric}: {metric_significant}/{metric_total} ({(metric_significant/metric_total)*100:.1f}%)")
    
    # Average effect sizes and degradation
    print(f"\nAverage effects:")
    avg_effect_size = df_ttests['effect_size'].abs().mean()
    avg_degradation = df_ttests['degradation'].mean()
    avg_p_value = df_ttests['p_value'].mean()
    
    print(f"  Average absolute effect size: {avg_effect_size:.3f}")
    print(f"  Average performance degradation: {avg_degradation:.2f}%")
    print(f"  Average p-value: {avg_p_value:.4f}")
    
    # Display most significant results
    print(f"\n🔍 TOP 10 MOST SIGNIFICANT DIFFERENCES:")
    print("-" * 50)
    top_significant = df_ttests.nsmallest(10, 'p_value')[
        ['configuration', 'metric', 'degradation', 'p_value', 'effect_size_desc']
    ]
    print(top_significant.to_string(index=False))
    
    return df_ttests
