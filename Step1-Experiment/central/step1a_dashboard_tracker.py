#!/usr/bin/env python3
"""
Web Dashboard Tracker for Central Model Training
Provides real-time tracking and visualization of training experiments
"""

import os
import json
import time
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ExperimentDashboard:
    """Real-time experiment tracking and dashboard generation"""
    
    def __init__(self, experiment_name: str, output_dir: str = "results/dashboard"):
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        self.experiment_id = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "data"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
        
        # Initialize tracking data
        self.experiments = []
        self.training_logs = {}
        self.hyperparameter_results = []
        self.model_comparisons = []
        
        # Dashboard file paths
        self.dashboard_path = os.path.join(output_dir, "dashboard.html")
        self.data_path = os.path.join(output_dir, "data", f"{self.experiment_id}.json")
        
        logger.info(f"📊 Dashboard initialized: {self.experiment_id}")
    
    def log_experiment_start(self, model_type: str, hyperparams: Dict, data_info: Dict):
        """Log the start of a new experiment"""
        experiment = {
            'experiment_id': self.experiment_id,
            'model_type': model_type,
            'hyperparameters': hyperparams,
            'data_info': data_info,
            'start_time': datetime.now().isoformat(),
            'status': 'running',
            'metrics': {}
        }
        
        self.experiments.append(experiment)
        self._save_data()
        logger.info(f"🚀 Experiment started: {model_type}")
    
    def log_training_epoch(self, model_type: str, epoch: int, metrics: Dict):
        """Log training metrics for each epoch"""
        if model_type not in self.training_logs:
            self.training_logs[model_type] = {
                'epochs': [],
                'train_loss': [],
                'train_acc': [],
                'val_loss': [],
                'val_acc': [],
                'learning_rate': [],
                'training_time': []
            }
        
        logs = self.training_logs[model_type]
        logs['epochs'].append(epoch)
        logs['train_loss'].append(metrics.get('train_loss', 0))
        logs['train_acc'].append(metrics.get('train_acc', 0))
        logs['val_loss'].append(metrics.get('val_loss', 0))
        logs['val_acc'].append(metrics.get('val_acc', 0))
        logs['learning_rate'].append(metrics.get('learning_rate', 0))
        logs['training_time'].append(metrics.get('training_time', 0))
        
        # Update dashboard every 5 epochs
        if epoch % 5 == 0:
            self._update_training_plots()
    
    def log_hyperparameter_result(self, model_type: str, hyperparams: Dict, metrics: Dict):
        """Log hyperparameter tuning results"""
        result = {
            'model_type': model_type,
            'hyperparameters': hyperparams,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        self.hyperparameter_results.append(result)
        self._save_data()
        self._update_hyperparameter_plots()
    
    def log_experiment_completion(self, model_type: str, final_metrics: Dict, best_hyperparams: Dict):
        """Log experiment completion"""
        for exp in self.experiments:
            if exp['model_type'] == model_type and exp['status'] == 'running':
                exp['status'] = 'completed'
                exp['end_time'] = datetime.now().isoformat()
                exp['metrics'] = final_metrics
                exp['best_hyperparameters'] = best_hyperparams
                break
        
        self._save_data()
        self._generate_full_dashboard()
        logger.info(f"✅ Experiment completed: {model_type}")
    
    def add_model_comparison(self, comparison_data: Dict):
        """Add model comparison results"""
        comparison_data['timestamp'] = datetime.now().isoformat()
        self.model_comparisons.append(comparison_data)
        self._save_data()
        self._update_comparison_plots()
    
    def _save_data(self):
        """Save all tracking data to JSON"""
        data = {
            'experiment_id': self.experiment_id,
            'experiment_name': self.experiment_name,
            'experiments': self.experiments,
            'training_logs': self.training_logs,
            'hyperparameter_results': self.hyperparameter_results,
            'model_comparisons': self.model_comparisons,
            'last_updated': datetime.now().isoformat()
        }
        
        with open(self.data_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def _update_training_plots(self):
        """Update training progress plots"""
        if not self.training_logs:
            return
        
        # Create training progress plots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Training Loss', 'Training Accuracy', 'Validation Loss', 'Validation Accuracy'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, (model_type, logs) in enumerate(self.training_logs.items()):
            color = colors[i % len(colors)]
            
            # Training Loss
            fig.add_trace(
                go.Scatter(x=logs['epochs'], y=logs['train_loss'], 
                          mode='lines+markers', name=f'{model_type} Train Loss',
                          line=dict(color=color)),
                row=1, col=1
            )
            
            # Training Accuracy
            fig.add_trace(
                go.Scatter(x=logs['epochs'], y=logs['train_acc'], 
                          mode='lines+markers', name=f'{model_type} Train Acc',
                          line=dict(color=color)),
                row=1, col=2
            )
            
            # Validation Loss
            fig.add_trace(
                go.Scatter(x=logs['epochs'], y=logs['val_loss'], 
                          mode='lines+markers', name=f'{model_type} Val Loss',
                          line=dict(color=color, dash='dash')),
                row=2, col=1
            )
            
            # Validation Accuracy
            fig.add_trace(
                go.Scatter(x=logs['epochs'], y=logs['val_acc'], 
                          mode='lines+markers', name=f'{model_type} Val Acc',
                          line=dict(color=color, dash='dash')),
                row=2, col=2
            )
        
        fig.update_layout(
            height=600,
            title_text=f"Training Progress - {self.experiment_name}",
            showlegend=True
        )
        
        # Save plot
        plot_path = os.path.join(self.output_dir, "plots", "training_progress.html")
        pyo.plot(fig, filename=plot_path, auto_open=False)
    
    def _update_hyperparameter_plots(self):
        """Update hyperparameter tuning plots"""
        if not self.hyperparameter_results:
            return
        
        # Create hyperparameter comparison plots for each model type
        model_types = list(set([r['model_type'] for r in self.hyperparameter_results]))
        
        for model_type in model_types:
            model_results = [r for r in self.hyperparameter_results if r['model_type'] == model_type]
            
            if len(model_results) < 2:
                continue
            
            # Create DataFrame for easier plotting
            df_data = []
            for result in model_results:
                row = {**result['hyperparameters'], **result['metrics']}
                row['model_type'] = result['model_type']
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            
            # Create hyperparameter vs accuracy plot
            if 'accuracy' in df.columns:
                fig = px.parallel_coordinates(
                    df, 
                    dimensions=[col for col in df.columns if col not in ['model_type', 'accuracy']],
                    color='accuracy',
                    title=f'Hyperparameter Tuning Results - {model_type}'
                )
                
                plot_path = os.path.join(self.output_dir, "plots", f"hyperparams_{model_type}.html")
                pyo.plot(fig, filename=plot_path, auto_open=False)
    
    def _update_comparison_plots(self):
        """Update model comparison plots"""
        if not self.model_comparisons:
            return
        
        # Create model comparison plots
        latest_comparison = self.model_comparisons[-1]
        
        if 'model_metrics' in latest_comparison:
            metrics_df = pd.DataFrame(latest_comparison['model_metrics'])
            
            # Create bar chart comparing models
            fig = go.Figure()
            
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            for metric in metrics:
                if metric in metrics_df.columns:
                    fig.add_trace(go.Bar(
                        name=metric.capitalize(),
                        x=metrics_df['model'],
                        y=metrics_df[metric]
                    ))
            
            fig.update_layout(
                barmode='group',
                title='Model Performance Comparison',
                xaxis_title='Model',
                yaxis_title='Score'
            )
            
            plot_path = os.path.join(self.output_dir, "plots", "model_comparison.html")
            pyo.plot(fig, filename=plot_path, auto_open=False)
    
    def _generate_full_dashboard(self):
        """Generate complete HTML dashboard"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>PDM Central Training Dashboard - {self.experiment_name}</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
        .section {{ background-color: white; padding: 20px; margin-bottom: 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #ecf0f1; border-radius: 3px; }}
        .status-running {{ color: #f39c12; }}
        .status-completed {{ color: #27ae60; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
        iframe {{ width: 100%; height: 400px; border: none; }}
    </style>
    <script>
        function refreshPage() {{
            location.reload();
        }}
        // Auto-refresh every 30 seconds
        setInterval(refreshPage, 30000);
    </script>
</head>
<body>
    <div class="header">
        <h1>🏭 PDM Central Training Dashboard</h1>
        <h2>{self.experiment_name}</h2>
        <p>Experiment ID: {self.experiment_id}</p>
        <p>Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <button onclick="refreshPage()" style="background-color: #3498db; color: white; border: none; padding: 10px 20px; border-radius: 3px; cursor: pointer;">🔄 Refresh</button>
    </div>
    
    <div class="section">
        <h3>📊 Experiment Summary</h3>
        <div class="grid">
        """
        
        # Add experiment summaries
        for exp in self.experiments:
            status_class = f"status-{exp['status']}"
            html_content += f"""
            <div class="metric">
                <h4>{exp['model_type']}</h4>
                <p>Status: <span class="{status_class}">{exp['status'].upper()}</span></p>
                <p>Started: {exp['start_time'][:16]}</p>
                {"<p>Completed: " + exp.get('end_time', '')[:16] + "</p>" if exp.get('end_time') else ""}
            </div>
            """
        
        html_content += """
        </div>
    </div>
    
    <div class="section">
        <h3>📈 Training Progress</h3>
        <iframe src="plots/training_progress.html"></iframe>
    </div>
    
    <div class="section">
        <h3>🔧 Hyperparameter Tuning</h3>
        <div class="grid">
        """
        
        # Add hyperparameter plots
        model_types = list(set([r['model_type'] for r in self.hyperparameter_results]))
        for model_type in model_types:
            html_content += f"""
            <div>
                <h4>{model_type}</h4>
                <iframe src="plots/hyperparams_{model_type}.html"></iframe>
            </div>
            """
        
        html_content += """
        </div>
    </div>
    
    <div class="section">
        <h3>🏆 Model Comparison</h3>
        <iframe src="plots/model_comparison.html"></iframe>
    </div>
    
    <div class="section">
        <h3>📋 Detailed Results</h3>
        <pre id="detailed-results"></pre>
    </div>
    
    <script>
        // Load detailed results
        fetch('data/""" + f"{self.experiment_id}.json" + """')
            .then(response => response.json())
            .then(data => {
                document.getElementById('detailed-results').textContent = JSON.stringify(data, null, 2);
            })
            .catch(error => console.error('Error loading data:', error));
    </script>
    
</body>
</html>
        """
        
        # Save dashboard
        with open(self.dashboard_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"📊 Dashboard updated: {self.dashboard_path}")
    
    def get_dashboard_url(self) -> str:
        """Get the dashboard URL"""
        return f"file://{os.path.abspath(self.dashboard_path)}"

class MetricsCollector:
    """Collect and format metrics for dashboard"""
    
    @staticmethod
    def calculate_training_metrics(outputs, targets, loss_value, training_time):
        """Calculate training metrics"""
        predictions = outputs.argmax(dim=1)
        accuracy = (predictions == targets).float().mean().item()
        
        return {
            'train_loss': loss_value,
            'train_acc': accuracy,
            'training_time': training_time
        }
    
    @staticmethod
    def calculate_validation_metrics(outputs, targets, loss_value):
        """Calculate validation metrics"""
        predictions = outputs.argmax(dim=1)
        accuracy = (predictions == targets).float().mean().item()
        
        return {
            'val_loss': loss_value,
            'val_acc': accuracy
        }
    
    @staticmethod
    def calculate_comprehensive_metrics(y_true, y_pred, y_pred_proba=None):
        """Calculate comprehensive evaluation metrics"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }
        
        if y_pred_proba is not None and len(np.unique(y_true)) == 2:
            metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
        
        return metrics
