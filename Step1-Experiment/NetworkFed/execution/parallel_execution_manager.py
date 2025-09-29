"""
Parallel Execution Manager
=========================
Manages parallel federated learning execution across multiple datasites.
Handles fault tolerance, adaptive waiting, and dynamic datasite management.
"""

import time
import threading
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
import queue

from monitoring.heartbeat_manager import get_heartbeat_manager


class ParallelExecutionManager:
    """
    Manages parallel execution of federated learning rounds.
    Provides fault tolerance and adaptive waiting based on heartbeat status.
    """
    
    def __init__(self, heartbeat_manager=None, max_wait_per_round: int = 600, check_interval: int = 30):
        """
        Initialize parallel execution manager.
        
        Args:
            heartbeat_manager: Heartbeat manager instance (can be client or direct manager)
            max_wait_per_round: Maximum time to wait for round completion (seconds)
            check_interval: How often to check datasite status (seconds)
        """
        self.max_wait_per_round = max_wait_per_round
        self.check_interval = check_interval
        self.heartbeat_manager = heartbeat_manager or get_heartbeat_manager()
        self.logger = logging.getLogger(__name__)
        
        # Execution state
        self.current_round = 0
        self.active_datasites = []
        self.round_results = {}
        self.execution_stats = {
            'rounds_completed': 0,
            'total_datasites_used': 0,
            'datasite_failures': 0,
            'adaptive_waits': 0
        }
    
    def execute_parallel_round(self, 
                              training_function, 
                              datasite_configs: List[Dict],
                              round_number: int,
                              **training_kwargs) -> Tuple[bool, Dict[str, Any]]:
        """
        Execute federated learning round in parallel across available datasites.
        
        Args:
            training_function: Function to execute on each datasite
            datasite_configs: List of datasite configurations
            round_number: Current round number
            **training_kwargs: Additional arguments for training function
            
        Returns:
            Tuple of (success, results_dict)
        """
        self.current_round = round_number
        
        # Step 1: Check available datasites
        datasite_names = [config['site_id'] for config in datasite_configs]
        available_datasites = self.heartbeat_manager.get_available_datasites(datasite_names)
        
        if len(available_datasites) < 2:
            self.logger.error(f"Round {round_number}: Insufficient datasites available: {available_datasites}")
            return False, {}
        
        # Filter configs to only available datasites
        active_configs = [config for config in datasite_configs if config['site_id'] in available_datasites]
        self.active_datasites = available_datasites.copy()
        
        self.logger.info(f"Round {round_number}: Starting parallel execution on {len(active_configs)} datasites: {available_datasites}")
        
        # Step 2: Execute training in parallel
        round_results = {}
        completed_datasites = set()
        failed_datasites = set()
        
        with ThreadPoolExecutor(max_workers=len(active_configs)) as executor:
            # Submit all training tasks
            future_to_datasite = {}
            for config in active_configs:
                future = executor.submit(
                    self._safe_training_execution,
                    training_function,
                    config,
                    round_number,
                    **training_kwargs
                )
                future_to_datasite[future] = config['site_id']
            
            # Step 3: Adaptive waiting with heartbeat monitoring
            start_time = time.time()
            last_completion_time = start_time
            
            while future_to_datasite and (time.time() - start_time) < self.max_wait_per_round:
                # Wait for completions with timeout
                try:
                    completed_futures = []
                    for future in as_completed(future_to_datasite.keys(), timeout=self.check_interval):
                        completed_futures.append(future)
                    
                    # Process completed futures
                    for future in completed_futures:
                        datasite_name = future_to_datasite[future]
                        try:
                            result = future.result()
                            if result['success']:
                                round_results[datasite_name] = result['data']
                                completed_datasites.add(datasite_name)
                                last_completion_time = time.time()
                                self.logger.info(f"Round {round_number}: {datasite_name} completed successfully")
                            else:
                                failed_datasites.add(datasite_name)
                                self.logger.warning(f"Round {round_number}: {datasite_name} failed: {result.get('error', 'Unknown error')}")
                        except Exception as e:
                            failed_datasites.add(datasite_name)
                            self.logger.error(f"Round {round_number}: {datasite_name} exception: {e}")
                        
                        # Remove completed future
                        del future_to_datasite[future]
                
                except TimeoutError:
                    # No completions in check_interval, check datasite health
                    remaining_datasites = list(future_to_datasite.values())
                    healthy, still_available = self.heartbeat_manager.are_datasites_healthy(
                        remaining_datasites, required_ratio=0.5
                    )
                    
                    if not healthy:
                        self.logger.warning(f"Round {round_number}: Too many datasites unhealthy, proceeding with completed results")
                        break
                    
                    # Check if we've been waiting too long since last completion
                    time_since_last_completion = time.time() - last_completion_time
                    if time_since_last_completion > (self.max_wait_per_round * 0.5):
                        self.logger.warning(f"Round {round_number}: Long wait since last completion, checking if we should proceed")
                        if len(completed_datasites) >= 2:
                            self.logger.info(f"Round {round_number}: Proceeding with {len(completed_datasites)} completed datasites")
                            break
                    
                    self.execution_stats['adaptive_waits'] += 1
                    self.logger.info(f"Round {round_number}: Adaptive wait - remaining: {remaining_datasites}, available: {still_available}")
            
            # Cancel remaining futures
            for future in future_to_datasite.keys():
                future.cancel()
        
        # Step 4: Evaluate round success
        total_completed = len(completed_datasites)
        total_failed = len(failed_datasites)
        
        self.logger.info(f"Round {round_number} summary: Completed: {total_completed}, Failed: {total_failed}")
        
        if total_completed >= 2:
            self.execution_stats['rounds_completed'] += 1
            self.execution_stats['total_datasites_used'] += total_completed
            self.execution_stats['datasite_failures'] += total_failed
            
            return True, {
                'round_number': round_number,
                'results': round_results,
                'completed_datasites': list(completed_datasites),
                'failed_datasites': list(failed_datasites),
                'execution_time': time.time() - start_time
            }
        else:
            self.logger.error(f"Round {round_number}: Insufficient completions for aggregation")
            return False, {}
    
    def _safe_training_execution(self, training_function, datasite_config: Dict, 
                                round_number: int, **kwargs) -> Dict[str, Any]:
        """
        Safely execute training function with error handling.
        
        Args:
            training_function: Function to execute
            datasite_config: Datasite configuration
            round_number: Current round number
            **kwargs: Additional arguments
            
        Returns:
            Dict with success status and result data
        """
        datasite_name = datasite_config['site_id']
        
        try:
            # Execute training function
            result = training_function(datasite_config, round_number, **kwargs)
            
            return {
                'success': True,
                'data': result,
                'datasite': datasite_name,
                'round': round_number,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Training execution failed for {datasite_name} in round {round_number}: {e}")
            return {
                'success': False,
                'error': str(e),
                'datasite': datasite_name,
                'round': round_number,
                'timestamp': datetime.now().isoformat()
            }
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            **self.execution_stats,
            'current_round': self.current_round,
            'active_datasites': self.active_datasites.copy()
        }
    
    def wait_for_datasites_ready(self, required_datasites: List[str], 
                                minimum_count: int = 2, max_wait: int = 300) -> bool:
        """
        Wait for minimum number of datasites to be ready for experiment.
        
        Args:
            required_datasites: List of required datasite names
            minimum_count: Minimum number of datasites needed
            max_wait: Maximum wait time in seconds
            
        Returns:
            True if minimum datasites ready, False otherwise
        """
        self.logger.info(f"Waiting for minimum {minimum_count} datasites from: {required_datasites}")
        
        start_time = time.time()
        while time.time() - start_time < max_wait:
            available = self.heartbeat_manager.get_available_datasites(required_datasites)
            
            if len(available) >= minimum_count:
                self.logger.info(f"Datasites ready for experiment: {available}")
                return True
            
            self.logger.info(f"Waiting... Available: {available} (need {minimum_count})")
            time.sleep(10)
        
        self.logger.error(f"Timeout waiting for datasites. Available: {self.heartbeat_manager.get_available_datasites(required_datasites)}")
        return False
