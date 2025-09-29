"""
Results Collection and Management
================================
Handles saving, loading, and organizing experiment results.
"""

import json
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd


class ResultsCollector:
    """
    Manages experiment results storage and retrieval.
    Supports both JSON and binary formats for different use cases.
    """
    
    def __init__(self, base_results_dir: str):
        self.base_results_dir = Path(base_results_dir)
        self.base_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.base_results_dir / "experiments").mkdir(exist_ok=True)
        (self.base_results_dir / "batches").mkdir(exist_ok=True)
        (self.base_results_dir / "metrics").mkdir(exist_ok=True)
        (self.base_results_dir / "summaries").mkdir(exist_ok=True)
    
    def save_experiment_results(self, experiment_id: str, results: Dict[str, Any]) -> str:
        """
        Save experiment results to file.
        
        Args:
            experiment_id: Unique experiment identifier
            results: Experiment results dictionary
            
        Returns:
            str: Path to saved results file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{experiment_id}_{timestamp}.json"
        filepath = self.base_results_dir / "experiments" / filename
        
        # Add metadata
        results_with_meta = {
            'metadata': {
                'experiment_id': experiment_id,
                'saved_at': timestamp,
                'version': '1.0'
            },
            'results': results
        }
        
        # Save as JSON
        with open(filepath, 'w') as f:
            json.dump(results_with_meta, f, indent=2, default=str)
        
        return str(filepath)
    
    def save_batch_results(self, batch_id: str, batch_results: Dict[str, Any]) -> str:
        """Save batch experiment results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{batch_id}_{timestamp}.json"
        filepath = self.base_results_dir / "batches" / filename
        
        with open(filepath, 'w') as f:
            json.dump(batch_results, f, indent=2, default=str)
        
        return str(filepath)
    
    def load_experiment_results(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Load the most recent results for an experiment."""
        experiment_files = list((self.base_results_dir / "experiments").glob(f"{experiment_id}_*.json"))
        
        if not experiment_files:
            return None
        
        # Get most recent file
        latest_file = max(experiment_files, key=lambda f: f.stat().st_mtime)
        
        with open(latest_file, 'r') as f:
            data = json.load(f)
        
        return data.get('results', data)
    
    def create_summary_report(self, experiment_ids: List[str]) -> Dict[str, Any]:
        """Create a summary report for multiple experiments."""
        summary = {
            'report_id': f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'generated_at': datetime.now().isoformat(),
            'experiments': [],
            'comparison': {}
        }
        
        experiment_data = []
        
        for exp_id in experiment_ids:
            results = self.load_experiment_results(exp_id)
            if results:
                exp_summary = {
                    'experiment_id': exp_id,
                    'algorithm': results.get('algorithm', 'unknown'),
                    'model_type': results.get('model_type', 'unknown'),
                    'total_rounds': len(results.get('round_results', [])),
                    'final_accuracy': results.get('final_metrics', {}).get('final_accuracy', 0),
                    'best_accuracy': results.get('final_metrics', {}).get('best_accuracy', 0),
                    'total_time': self._calculate_total_time(results),
                    'datasites_count': len(results.get('datasites', []))
                }
                experiment_data.append(exp_summary)
                summary['experiments'].append(exp_summary)
        
        # Add comparison metrics
        if experiment_data:
            summary['comparison'] = {
                'best_experiment': max(experiment_data, key=lambda x: x['final_accuracy'])['experiment_id'],
                'worst_experiment': min(experiment_data, key=lambda x: x['final_accuracy'])['experiment_id'],
                'avg_accuracy': sum(exp['final_accuracy'] for exp in experiment_data) / len(experiment_data),
                'avg_rounds': sum(exp['total_rounds'] for exp in experiment_data) / len(experiment_data)
            }
        
        # Save summary
        summary_file = self.base_results_dir / "summaries" / f"{summary['report_id']}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return summary
    
    def export_to_csv(self, experiment_ids: List[str], output_file: str = None) -> str:
        """Export experiment results to CSV for analysis."""
        if output_file is None:
            output_file = f"experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        csv_path = self.base_results_dir / output_file
        
        # Collect data for CSV
        rows = []
        for exp_id in experiment_ids:
            results = self.load_experiment_results(exp_id)
            if results:
                # Basic experiment info
                base_row = {
                    'experiment_id': exp_id,
                    'algorithm': results.get('algorithm', ''),
                    'model_type': results.get('model_type', ''),
                    'max_rounds': results.get('max_rounds', 0),
                    'datasites_count': len(results.get('datasites', [])),
                }
                
                # Final metrics
                final_metrics = results.get('final_metrics', {})
                base_row.update({
                    'final_accuracy': final_metrics.get('final_accuracy', 0),
                    'final_loss': final_metrics.get('final_loss', 0),
                    'best_accuracy': final_metrics.get('best_accuracy', 0),
                    'total_communication_time': final_metrics.get('total_communication_time', 0),
                    'total_training_time': final_metrics.get('total_training_time', 0)
                })
                
                # Round-by-round data
                round_results = results.get('round_results', [])
                for i, round_result in enumerate(round_results):
                    row = base_row.copy()
                    row.update({
                        'round_num': i + 1,
                        'round_participants': round_result.get('num_participants', 0),
                        'round_time': round_result.get('round_time', 0)
                    })
                    
                    # Average client metrics for this round
                    client_updates = round_result.get('client_updates', [])
                    if client_updates:
                        row.update({
                            'round_avg_accuracy': sum(u.get('accuracy', 0) for u in client_updates) / len(client_updates),
                            'round_avg_loss': sum(u.get('loss', 0) for u in client_updates) / len(client_updates),
                            'round_avg_training_time': sum(u.get('training_time', 0) for u in client_updates) / len(client_updates)
                        })
                    
                    rows.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        
        return str(csv_path)
    
    def _calculate_total_time(self, results: Dict[str, Any]) -> float:
        """Calculate total experiment time."""
        start_time = results.get('start_time')
        end_time = results.get('end_time')
        
        if start_time and end_time:
            if isinstance(start_time, str):
                start_time = datetime.fromisoformat(start_time)
            if isinstance(end_time, str):
                end_time = datetime.fromisoformat(end_time)
            
            return (end_time - start_time).total_seconds()
        
        return 0.0
    
    def list_experiments(self) -> List[str]:
        """List all available experiment IDs."""
        experiment_files = list((self.base_results_dir / "experiments").glob("*.json"))
        
        experiment_ids = set()
        for file in experiment_files:
            # Extract experiment ID from filename (before first underscore)
            exp_id = file.stem.split('_')[0]
            experiment_ids.add(exp_id)
        
        return sorted(list(experiment_ids))
    
    def cleanup_old_results(self, days_old: int = 30) -> int:
        """Clean up results older than specified days."""
        import time
        
        cutoff_time = time.time() - (days_old * 24 * 60 * 60)
        removed_count = 0
        
        for results_dir in ["experiments", "batches", "metrics"]:
            dir_path = self.base_results_dir / results_dir
            for file in dir_path.glob("*.json"):
                if file.stat().st_mtime < cutoff_time:
                    file.unlink()
                    removed_count += 1
        
        return removed_count
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        stats = {
            'total_experiments': len(list((self.base_results_dir / "experiments").glob("*.json"))),
            'total_batches': len(list((self.base_results_dir / "batches").glob("*.json"))),
            'total_summaries': len(list((self.base_results_dir / "summaries").glob("*.json"))),
            'total_size_mb': 0
        }
        
        # Calculate total size
        for file in self.base_results_dir.rglob("*"):
            if file.is_file():
                stats['total_size_mb'] += file.stat().st_size / (1024 * 1024)
        
        stats['total_size_mb'] = round(stats['total_size_mb'], 2)
        
        return stats
