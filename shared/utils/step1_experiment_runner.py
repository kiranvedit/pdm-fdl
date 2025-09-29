#!/usr/bin/env python3
"""
Experiment Runner for Federated Learning Predictive Maintenance
Main orchestration script for running complete data preparation experiments
"""

import os
import sys
import logging
import argparse
import traceback
from datetime import datetime
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from step1_data_preparation import DataPreprocessor, DataConfig
from step1_data_utils import DataLoader, DataValidator, DataVisualizer, validate_and_visualize
from step1_config import ConfigManager, create_default_config

# Setup logging
def setup_logging(log_dir: str, log_level: str = 'INFO') -> None:
    """Setup comprehensive logging"""
    os.makedirs(log_dir, exist_ok=True)
    
    # Create timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"experiment_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger

class ExperimentRunner:
    """Main experiment orchestration class"""
    
    def __init__(self, config_path: str = None):
        self.start_time = datetime.now()
        
        # Load or create configuration
        if config_path and os.path.exists(config_path):
            self.config_manager = ConfigManager(config_path)
        else:
            config_path = config_path or "/workspaces/pdm-fdl/shared/utils/step1_config.yaml"
            self.config_manager = create_default_config(config_path)
        
        # Setup logging
        self.logger = setup_logging(
            self.config_manager.experiment_config.logs_dir,
            self.config_manager.experiment_config.log_level
        )
        
        # Initialize components
        self.data_preprocessor = None
        self.data_loader = None
        self.results = {}
        
        self.logger.info("ExperimentRunner initialized")
        self.logger.info(f"Experiment: {self.config_manager.experiment_config.experiment_name}")
    
    def setup_environment(self) -> bool:
        """Setup experiment environment and validate configuration"""
        try:
            self.logger.info("Setting up experiment environment...")
            
            # Validate configuration
            if not self.config_manager.validate_config():
                self.logger.error("Configuration validation failed")
                return False
            
            # Setup directories
            self.config_manager.setup_experiment_directories()
            
            # Save configuration for reference
            config_save_path = os.path.join(
                self.config_manager.experiment_config.results_dir,
                "experiment_config.yaml"
            )
            self.config_manager.save_config(config_save_path)
            
            self.logger.info("Environment setup completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting up environment: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
    
    def run_data_preparation(self) -> bool:
        """Run complete data preparation pipeline"""
        try:
            self.logger.info("Starting data preparation pipeline...")
            
            # Create data configuration from experiment config
            data_config = DataConfig(
                data_path=os.path.join(self.config_manager.experiment_config.data_dir.replace("processed_data", ""), "../shared/data/ai4i2020.csv"),
                output_dir=self.config_manager.experiment_config.data_dir,
                train_ratio=0.6,
                val_ratio=0.2,
                test_ratio=0.2,
                normalize_method='minmax',
                handle_imbalance=True,
                imbalance_method='smote',
                create_sequences=True,
                sequence_length=10,
                num_clients=self.config_manager.federated_config.num_clients,
                non_iid_alpha=0.5,
                random_seed=self.config_manager.experiment_config.random_seed
            )
            
            # Initialize preprocessor
            self.data_preprocessor = DataPreprocessor(data_config)
            
            # Run preprocessing pipeline
            self.logger.info("Loading and cleaning data...")
            self.data_preprocessor.load_data()
            self.data_preprocessor.clean_data()
            
            self.logger.info("Engineering features...")
            self.data_preprocessor.engineer_features()
            
            self.logger.info("Preparing features and targets...")
            self.data_preprocessor.prepare_features_and_targets()
            
            self.logger.info("Splitting data...")
            data_splits = self.data_preprocessor.split_data()
            
            # Skip federated splits - training split will be used for all clients
            self.logger.info("Skipping federated splits - training split will be used for all clients")
            federated_splits = {}
            
            self.logger.info("Saving processed data to CSV format...")
            self.data_preprocessor.save_processed_data(data_splits)
            
            # Store results
            self.results['data_preparation'] = {
                'status': 'completed',
                'data_splits': {k: {kk: vv.shape for kk, vv in v.items()} for k, v in data_splits.items()},
                'federated_approach': 'training_split_shared',
                'completion_time': datetime.now()
            }
            
            self.logger.info("Data preparation completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in data preparation: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.results['data_preparation'] = {
                'status': 'failed',
                'error': str(e),
                'completion_time': datetime.now()
            }
            return False
    
    def run_data_validation(self) -> bool:
        """Run comprehensive data validation"""
        try:
            self.logger.info("Starting data validation...")
            
            # Initialize data loader
            self.data_loader = DataLoader(self.config_manager.experiment_config.data_dir)
            
            # Load and validate tabular data
            tabular_data = self.data_loader.load_tabular_data()
            tabular_valid = DataValidator.validate_data_splits(tabular_data)
            
            # Load and validate federated data
            try:
                federated_data = self.data_loader.load_federated_data()
                federated_valid = DataValidator.validate_federated_data(federated_data)
            except FileNotFoundError:
                self.logger.warning("No federated data found")
                federated_data = None
                federated_valid = False
            
            # Load sequence data if available
            sequence_valid = True
            try:
                sequence_data = self.data_loader.load_sequence_data()
                if sequence_data:
                    sequence_valid = DataValidator.validate_data_splits(
                        sequence_data, 
                        ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test']
                    )
            except:
                sequence_data = None
                sequence_valid = False
                self.logger.info("No sequence data found - skipping sequence validation")
            
            # Overall validation status
            validation_success = tabular_valid and (federated_valid or federated_data is None)
            
            # Store validation results
            self.results['data_validation'] = {
                'status': 'completed' if validation_success else 'failed',
                'tabular_data_valid': tabular_valid,
                'federated_data_valid': federated_valid,
                'sequence_data_valid': sequence_valid,
                'tabular_data_shapes': {k: v.shape for k, v in tabular_data.items()},
                'federated_clients': len(federated_data) if federated_data else 0,
                'completion_time': datetime.now()
            }
            
            if validation_success:
                self.logger.info("Data validation completed successfully")
            else:
                self.logger.error("Data validation failed")
            
            return validation_success
            
        except Exception as e:
            self.logger.error(f"Error in data validation: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.results['data_validation'] = {
                'status': 'failed',
                'error': str(e),
                'completion_time': datetime.now()
            }
            return False
    
    def generate_data_visualizations(self) -> bool:
        """Generate comprehensive data visualizations"""
        try:
            self.logger.info("Generating data visualizations...")
            
            if not self.data_loader:
                self.data_loader = DataLoader(self.config_manager.experiment_config.data_dir)
            
            # Initialize visualizer
            visualizer = DataVisualizer()
            
            # Load data
            tabular_data = self.data_loader.load_tabular_data()
            feature_names = self.data_loader.get_feature_names()
            
            # Generate visualizations
            self.logger.info("Creating data distribution plots...")
            visualizer.plot_data_distribution(tabular_data, "Training Data Distribution")
            
            self.logger.info("Creating feature distribution plots...")
            visualizer.plot_feature_distributions(tabular_data['X_train'], feature_names)
            
            # Federated client distributions
            try:
                federated_data = self.data_loader.load_federated_data()
                if federated_data:
                    self.logger.info("Creating federated client distribution plots...")
                    visualizer.plot_federated_distribution(federated_data)
            except:
                self.logger.info("Skipping federated visualizations - data not available")
            
            # Sequence data visualizations
            try:
                sequence_data = self.data_loader.load_sequence_data()
                if sequence_data:
                    self.logger.info("Creating sequence data sample plots...")
                    visualizer.plot_sequence_data_sample(
                        sequence_data['X_train'],
                        sequence_data['y_train'],
                        feature_names,
                        n_samples=3
                    )
            except:
                self.logger.info("Skipping sequence visualizations - data not available")
            
            self.results['data_visualization'] = {
                'status': 'completed',
                'completion_time': datetime.now()
            }
            
            self.logger.info("Data visualizations generated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.results['data_visualization'] = {
                'status': 'failed',
                'error': str(e),
                'completion_time': datetime.now()
            }
            return False
    
    def generate_experiment_report(self) -> bool:
        """Generate comprehensive experiment report"""
        try:
            self.logger.info("Generating experiment report...")
            
            # Calculate execution time
            execution_time = datetime.now() - self.start_time
            
            # Create experiment summary
            experiment_summary = self.config_manager.create_experiment_summary()
            experiment_summary['execution_time'] = str(execution_time)
            experiment_summary['results'] = self.results
            
            # Save experiment summary
            import json
            report_path = os.path.join(
                self.config_manager.experiment_config.results_dir,
                "experiment_report.json"
            )
            
            with open(report_path, 'w') as f:
                json.dump(experiment_summary, f, indent=2, default=str)
            
            # Generate markdown report
            markdown_report = self._generate_markdown_report(experiment_summary)
            markdown_path = os.path.join(
                self.config_manager.experiment_config.results_dir,
                "experiment_report.md"
            )
            
            with open(markdown_path, 'w') as f:
                f.write(markdown_report)
            
            self.logger.info(f"Experiment report saved to: {report_path}")
            self.logger.info(f"Markdown report saved to: {markdown_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
    
    def _generate_markdown_report(self, summary: dict) -> str:
        """Generate markdown experiment report"""
        report = f"""# {summary['experiment_info']['name']}

## Experiment Overview
- **Description**: {summary['experiment_info']['description']}
- **Version**: {summary['experiment_info']['version']}
- **Timestamp**: {summary['experiment_info']['timestamp']}
- **Execution Time**: {summary['execution_time']}

## Configuration
- **Model Types**: {', '.join(summary['model_types'])}
- **Number of Clients**: {summary['federated_setup']['num_clients']}
- **Communication Rounds**: {summary['federated_setup']['num_rounds']}
- **Local Epochs**: {summary['federated_setup']['local_epochs']}
- **Aggregation Method**: {summary['federated_setup']['aggregation_method']}

## Training Configuration
- **Epochs**: {summary['training_setup']['epochs']}
- **Batch Size**: {summary['training_setup']['batch_size']}
- **Learning Rate**: {summary['training_setup']['learning_rate']}
- **Optimizer**: {summary['training_setup']['optimizer']}

## Results Summary

### Data Preparation
"""
        
        if 'data_preparation' in summary['results']:
            dp_result = summary['results']['data_preparation']
            report += f"- **Status**: {dp_result['status']}\n"
            if dp_result['status'] == 'completed':
                report += f"- **Federated Clients**: {dp_result['federated_clients']}\n"
                report += f"- **Data Splits**:\n"
                for split_type, shapes in dp_result['data_splits'].items():
                    report += f"  - {split_type}: {shapes}\n"
        
        report += "\n### Data Validation\n"
        if 'data_validation' in summary['results']:
            dv_result = summary['results']['data_validation']
            report += f"- **Status**: {dv_result['status']}\n"
            report += f"- **Tabular Data Valid**: {dv_result['tabular_data_valid']}\n"
            report += f"- **Federated Data Valid**: {dv_result['federated_data_valid']}\n"
            report += f"- **Sequence Data Valid**: {dv_result['sequence_data_valid']}\n"
        
        report += "\n### Data Visualization\n"
        if 'data_visualization' in summary['results']:
            viz_result = summary['results']['data_visualization']
            report += f"- **Status**: {viz_result['status']}\n"
        
        report += f"\n## Evaluation Metrics\n"
        report += f"- {', '.join(summary['evaluation_metrics'])}\n"
        
        report += f"\n## Random Seed\n{summary['random_seed']}\n"
        
        return report
    
    def run_complete_experiment(self) -> bool:
        """Run the complete data preparation experiment"""
        self.logger.info("="*80)
        self.logger.info("STARTING COMPLETE DATA PREPARATION EXPERIMENT")
        self.logger.info("="*80)
        
        success = True
        
        # Step 1: Setup environment
        if not self.setup_environment():
            self.logger.error("Failed to setup environment")
            return False
        
        # Step 2: Data preparation
        if not self.run_data_preparation():
            self.logger.error("Data preparation failed")
            success = False
        
        # Step 3: Data validation
        if success and not self.run_data_validation():
            self.logger.error("Data validation failed")
            success = False
        
        # Step 4: Generate visualizations
        if success:
            self.generate_data_visualizations()  # Non-critical, continue even if fails
        
        # Step 5: Generate report
        self.generate_experiment_report()
        
        # Final summary
        execution_time = datetime.now() - self.start_time
        self.logger.info("="*80)
        if success:
            self.logger.info("EXPERIMENT COMPLETED SUCCESSFULLY")
        else:
            self.logger.error("EXPERIMENT COMPLETED WITH ERRORS")
        self.logger.info(f"Total execution time: {execution_time}")
        self.logger.info("="*80)
        
        return success

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Run federated learning data preparation experiment")
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--data-only', action='store_true', help='Run only data preparation')
    parser.add_argument('--validate-only', action='store_true', help='Run only data validation')
    parser.add_argument('--visualize-only', action='store_true', help='Run only data visualization')
    
    args = parser.parse_args()
    
    try:
        # Initialize experiment runner
        runner = ExperimentRunner(args.config)
        
        if args.data_only:
            success = runner.setup_environment() and runner.run_data_preparation()
        elif args.validate_only:
            success = runner.run_data_validation()
        elif args.visualize_only:
            success = runner.generate_data_visualizations()
        else:
            success = runner.run_complete_experiment()
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
