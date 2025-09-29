"""
Dashboard Tracker for ML Experiments
Real-time web dashboard for tracking machine learning experiments and metrics
"""

import os
import json
import time
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.utils

class MetricsCollector:
    """Collects and manages experiment metrics"""
    
    def __init__(self):
        self.experiments = {}
        self.current_experiment = None
        self.metrics_history = []
        
    def start_experiment(self, name: str, config: Dict[str, Any], metadata: Dict[str, Any] = None):
        """Start a new experiment"""
        self.current_experiment = {
            'name': name,
            'config': config,
            'metadata': metadata or {},
            'start_time': datetime.now(),
            'metrics': [],
            'epochs': [],
            'status': 'running'
        }
        self.experiments[name] = self.current_experiment
        
    def log_metric(self, metric_name: str, value: float, epoch: int = None, step: int = None):
        """Log a metric value"""
        if self.current_experiment is None:
            return
            
        metric_entry = {
            'name': metric_name,
            'value': value,
            'timestamp': datetime.now(),
            'epoch': epoch,
            'step': step
        }
        
        self.current_experiment['metrics'].append(metric_entry)
        self.metrics_history.append({
            'experiment': self.current_experiment['name'],
            **metric_entry
        })
        
    def log_epoch_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Log multiple metrics for an epoch"""
        epoch_data = {
            'epoch': epoch,
            'timestamp': datetime.now(),
            'metrics': metrics
        }
        
        if self.current_experiment:
            self.current_experiment['epochs'].append(epoch_data)
            
        # Log individual metrics
        for name, value in metrics.items():
            self.log_metric(name, value, epoch=epoch)
    
    def finish_experiment(self, final_metrics: Dict[str, float] = None):
        """Finish the current experiment"""
        if self.current_experiment:
            self.current_experiment['end_time'] = datetime.now()
            self.current_experiment['status'] = 'completed'
            self.current_experiment['duration'] = (
                self.current_experiment['end_time'] - self.current_experiment['start_time']
            ).total_seconds()
            
            if final_metrics:
                self.current_experiment['final_metrics'] = final_metrics
                
        self.current_experiment = None

class ExperimentDashboard:
    """Web dashboard for experiment visualization"""
    
    def __init__(self, port: int = 8050):
        self.port = port
        self.metrics_collector = MetricsCollector()
        self.dashboard_data = {
            'experiments': [],
            'model_comparisons': [],
            'hyperparameter_results': {},
            'real_time_metrics': []
        }
        self.dashboard_html = None
        self.auto_refresh = True
        
    def log_experiment_start(self, name: str, config: Dict[str, Any], metadata: Dict[str, Any] = None):
        """Start logging a new experiment"""
        self.metrics_collector.start_experiment(name, config, metadata)
        
    def log_metric(self, metric_name: str, value: float, epoch: int = None):
        """Log a single metric"""
        self.metrics_collector.log_metric(metric_name, value, epoch)
        
        # Add to real-time metrics for dashboard
        self.dashboard_data['real_time_metrics'].append({
            'experiment': self.metrics_collector.current_experiment['name'] if self.metrics_collector.current_experiment else 'unknown',
            'metric': metric_name,
            'value': value,
            'epoch': epoch,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last 1000 metrics to prevent memory issues
        if len(self.dashboard_data['real_time_metrics']) > 1000:
            self.dashboard_data['real_time_metrics'] = self.dashboard_data['real_time_metrics'][-1000:]
    
    def log_epoch_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Log multiple metrics for an epoch"""
        self.metrics_collector.log_epoch_metrics(epoch, metrics)
        
        # Log each metric individually for real-time tracking
        for name, value in metrics.items():
            self.log_metric(name, value, epoch)
    
    def log_experiment_completion(self, name: str, final_metrics: Dict[str, float], config: Dict[str, Any]):
        """Complete an experiment and log final results"""
        self.metrics_collector.finish_experiment(final_metrics)
        
        # Add to experiments list
        experiment_data = {
            'name': name,
            'final_metrics': final_metrics,
            'config': config,
            'completion_time': datetime.now().isoformat()
        }
        self.dashboard_data['experiments'].append(experiment_data)
    
    def add_hyperparameter_results(self, model_name: str, results: List[Dict[str, Any]]):
        """Add hyperparameter tuning results"""
        self.dashboard_data['hyperparameter_results'][model_name] = results
    
    def add_model_comparison(self, comparison_data: Dict[str, Any]):
        """Add model comparison data"""
        self.dashboard_data['model_comparisons'].append(comparison_data)
    
    def create_metrics_plot(self) -> str:
        """Create a plot of training metrics"""
        if not self.dashboard_data['real_time_metrics']:
            return "<p>No metrics data available</p>"
        
        df = pd.DataFrame(self.dashboard_data['real_time_metrics'])
        
        if df.empty:
            return "<p>No metrics data available</p>"
        
        # Create subplots for different metrics
        unique_metrics = df['metric'].unique()
        fig = make_subplots(
            rows=len(unique_metrics), cols=1,
            subplot_titles=unique_metrics,
            shared_xaxes=True
        )
        
        for i, metric in enumerate(unique_metrics, 1):
            metric_data = df[df['metric'] == metric]
            
            for exp_name in metric_data['experiment'].unique():
                exp_data = metric_data[metric_data['experiment'] == exp_name]
                fig.add_trace(
                    go.Scatter(
                        x=exp_data['epoch'],
                        y=exp_data['value'],
                        mode='lines+markers',
                        name=f"{exp_name}_{metric}",
                        showlegend=(i == 1)  # Only show legend for first subplot
                    ),
                    row=i, col=1
                )
        
        fig.update_layout(
            title="Training Metrics Over Time",
            height=300 * len(unique_metrics),
            showlegend=True
        )
        
        return fig.to_html(include_plotlyjs='cdn')
    
    def create_model_comparison_plot(self) -> str:
        """Create model comparison visualization"""
        if not self.dashboard_data['experiments']:
            return "<p>No experiment data available</p>"
        
        df = pd.DataFrame(self.dashboard_data['experiments'])
        
        if df.empty or 'final_metrics' not in df.columns:
            return "<p>No final metrics available</p>"
        
        # Extract metrics
        metrics_data = []
        for _, row in df.iterrows():
            for metric_name, metric_value in row['final_metrics'].items():
                metrics_data.append({
                    'model': row['name'],
                    'metric': metric_name,
                    'value': metric_value
                })
        
        metrics_df = pd.DataFrame(metrics_data)
        
        if metrics_df.empty:
            return "<p>No metrics data to display</p>"
        
        # Create bar plot
        fig = px.bar(
            metrics_df, 
            x='model', 
            y='value', 
            color='metric',
            title="Model Performance Comparison",
            barmode='group'
        )
        
        fig.update_layout(
            xaxis_title="Model",
            yaxis_title="Metric Value",
            height=500
        )
        
        return fig.to_html(include_plotlyjs='cdn')
    
    def create_hyperparameter_heatmap(self, model_name: str) -> str:
        """Create hyperparameter optimization heatmap"""
        if model_name not in self.dashboard_data['hyperparameter_results']:
            return f"<p>No hyperparameter data for {model_name}</p>"
        
        results = self.dashboard_data['hyperparameter_results'][model_name]
        
        if not results:
            return f"<p>No hyperparameter results for {model_name}</p>"
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        if df.empty:
            return f"<p>No hyperparameter data to display for {model_name}</p>"
        
        # Create correlation heatmap of hyperparameters vs performance
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) < 2:
            return f"<p>Insufficient numeric data for heatmap for {model_name}</p>"
        
        correlation_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(
            correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            color_continuous_scale='RdBu',
            title=f"Hyperparameter Correlation Heatmap - {model_name}"
        )
        
        return fig.to_html(include_plotlyjs='cdn')
    
    def generate_dashboard_html(self) -> str:
        """Generate complete dashboard HTML"""
        metrics_plot = self.create_metrics_plot()
        model_comparison = self.create_model_comparison_plot()
        
        # Generate hyperparameter heatmaps for each model
        heatmaps_html = ""
        for model_name in self.dashboard_data['hyperparameter_results'].keys():
            heatmap = self.create_hyperparameter_heatmap(model_name)
            heatmaps_html += f"<div class='heatmap-section'><h3>{model_name} Hyperparameters</h3>{heatmap}</div>"
        
        # Current experiment status
        current_exp_html = ""
        if self.metrics_collector.current_experiment:
            exp = self.metrics_collector.current_experiment
            current_exp_html = f"""
            <div class="current-experiment">
                <h3>Current Experiment: {exp['name']}</h3>
                <p>Status: {exp['status']}</p>
                <p>Started: {exp['start_time'].strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Configuration: {json.dumps(exp['config'], indent=2)}</p>
            </div>
            """
        
        # Experiments summary table
        experiments_table = ""
        if self.dashboard_data['experiments']:
            experiments_table = "<h3>Completed Experiments</h3><table border='1'><tr><th>Name</th><th>Completion Time</th><th>Final Metrics</th></tr>"
            for exp in self.dashboard_data['experiments']:
                metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in exp['final_metrics'].items()])
                experiments_table += f"<tr><td>{exp['name']}</td><td>{exp['completion_time']}</td><td>{metrics_str}</td></tr>"
            experiments_table += "</table>"
        
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ML Experiment Dashboard</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                .section {{ background-color: white; padding: 20px; margin-bottom: 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .current-experiment {{ background-color: #e8f5e8; padding: 15px; border-radius: 5px; border-left: 4px solid #27ae60; }}
                .heatmap-section {{ margin-bottom: 30px; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 8px; text-align: left; }}
                th {{ background-color: #34495e; color: white; }}
                .timestamp {{ color: #7f8c8d; font-size: 0.9em; }}
                .refresh-note {{ text-align: center; color: #7f8c8d; font-style: italic; }}
            </style>
            {f'<meta http-equiv="refresh" content="30">' if self.auto_refresh else ''}
        </head>
        <body>
            <div class="header">
                <h1>🔬 ML Experiment Dashboard</h1>
                <p>Real-time tracking of machine learning experiments</p>
                <div class="timestamp">Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
            </div>
            
            {current_exp_html}
            
            <div class="section">
                <h2>📊 Training Metrics</h2>
                {metrics_plot}
            </div>
            
            <div class="section">
                <h2>🏆 Model Comparison</h2>
                {model_comparison}
            </div>
            
            <div class="section">
                <h2>🎛️ Hyperparameter Analysis</h2>
                {heatmaps_html if heatmaps_html else '<p>No hyperparameter data available</p>'}
            </div>
            
            <div class="section">
                <h2>📋 Experiments Summary</h2>
                {experiments_table if experiments_table else '<p>No completed experiments</p>'}
            </div>
            
            <div class="refresh-note">
                {'Dashboard auto-refreshes every 30 seconds' if self.auto_refresh else 'Manual refresh required'}
            </div>
        </body>
        </html>
        """
        
        return html_template
    
    def save_dashboard(self, filepath: str = "dashboard.html"):
        """Save dashboard to HTML file"""
        html_content = self.generate_dashboard_html()
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.dashboard_html = filepath
        return filepath
    
    def get_dashboard_url(self) -> str:
        """Get dashboard URL"""
        if self.dashboard_html:
            return f"file://{os.path.abspath(self.dashboard_html)}"
        return "Dashboard not yet created"
    
    def export_data(self, filepath: str = "experiment_data.json"):
        """Export all dashboard data to JSON"""
        export_data = {
            'dashboard_data': self.dashboard_data,
            'experiments': self.metrics_collector.experiments,
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        return filepath

# Factory function for easy dashboard creation
def create_dashboard(port: int = 8050, auto_refresh: bool = True) -> ExperimentDashboard:
    """Create and return a new experiment dashboard"""
    dashboard = ExperimentDashboard(port)
    dashboard.auto_refresh = auto_refresh
    return dashboard

if __name__ == "__main__":
    # Example usage
    dashboard = create_dashboard()
    
    # Simulate experiment
    dashboard.log_experiment_start(
        "test_experiment",
        {"learning_rate": 0.001, "batch_size": 32},
        {"dataset": "test_data"}
    )
    
    # Simulate training metrics
    for epoch in range(10):
        metrics = {
            "train_loss": 1.0 - epoch * 0.1,
            "val_loss": 1.2 - epoch * 0.08,
            "train_acc": epoch * 0.1,
            "val_acc": epoch * 0.08
        }
        dashboard.log_epoch_metrics(epoch, metrics)
    
    # Complete experiment
    dashboard.log_experiment_completion(
        "test_experiment",
        {"final_accuracy": 0.95, "final_loss": 0.05},
        {"learning_rate": 0.001, "batch_size": 32}
    )
    
    # Save dashboard
    dashboard_file = dashboard.save_dashboard("test_dashboard.html")
    print(f"Dashboard saved to: {dashboard_file}")
    print(f"Open in browser: {dashboard.get_dashboard_url()}")
