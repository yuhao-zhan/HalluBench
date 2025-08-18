#!/usr/bin/env python3
"""
Parallel processing script for evaluating multiple JSON files in deerflow/ directory.
Optimized for GPU memory efficiency and parallel execution.
"""

import argparse
import asyncio
import json
import multiprocessing as mp
import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import subprocess
import psutil
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
import queue

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(processName)s] %(message)s'
)
logger = logging.getLogger(__name__)

class GPUMemoryMonitor:
    """Monitor GPU memory usage across processes."""
    
    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start GPU memory monitoring in a separate thread."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop GPU memory monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
    
    def _monitor_loop(self):
        """Monitor GPU memory usage every 30 seconds."""
        while self.monitoring:
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    for i, line in enumerate(result.stdout.strip().split('\n')):
                        if line.strip():
                            used, total = map(int, line.split(', '))
                            usage_percent = (used / total) * 100
                            if usage_percent > 85:
                                logger.warning(f"GPU {i} memory high: {usage_percent:.1f}% ({used}MB/{total}MB)")
                            elif usage_percent > 95:
                                logger.error(f"GPU {i} memory critical: {usage_percent:.1f}% - may cause OOM!")
            except Exception as e:
                logger.debug(f"GPU monitoring error: {e}")
            
            time.sleep(30)


class ProcessManager:
    """Manage parallel processing with GPU memory constraints."""
    
    def __init__(self, max_workers: int = None, max_gpu_memory_percent: float = 90.0):
        self.max_workers = max_workers or min(mp.cpu_count(), 4)
        self.max_gpu_memory_percent = max_gpu_memory_percent
        self.active_processes = {}
        self.gpu_monitor = GPUMemoryMonitor()
        
    def get_gpu_memory_usage(self) -> List[float]:
        """Get current GPU memory usage percentages."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                usage_percentages = []
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        used, total = map(int, line.split(', '))
                        usage_percentages.append((used / total) * 100)
                return usage_percentages
        except Exception as e:
            logger.warning(f"Could not get GPU memory usage: {e}")
        return [0.0]  # Fallback if nvidia-smi not available
    
    def can_start_new_process(self) -> bool:
        """Check if we can start a new process based on GPU memory."""
        gpu_usage = self.get_gpu_memory_usage()
        max_usage = max(gpu_usage) if gpu_usage else 0
        return max_usage < self.max_gpu_memory_percent
    
    def wait_for_gpu_memory(self, timeout: int = 300):
        """Wait for GPU memory to become available."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.can_start_new_process():
                return True
            logger.info(f"Waiting for GPU memory... Current max usage: {max(self.get_gpu_memory_usage()):.1f}%")
            time.sleep(10)
        return False


def process_single_file(file_path: str, gpu_id: int = None, log_dir: str = "../log/train_gemini") -> Dict[str, Any]:
    """
    Process a single JSON file using the evaluate.py script.
    
    Args:
        file_path: Path to the JSON file to process
        gpu_id: Optional GPU ID to use (for CUDA_VISIBLE_DEVICES)
        log_dir: Directory to store log files
        
    Returns:
        Dictionary containing processing results and metadata
    """
    start_time = time.time()
    process_id = os.getpid()
    
    try:
        # Set GPU device if specified
        env = os.environ.copy()
        if gpu_id is not None:
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            logger.info(f"[PID {process_id}] Processing {os.path.basename(file_path)} on GPU {gpu_id}")
        else:
            logger.info(f"[PID {process_id}] Processing {os.path.basename(file_path)} on available GPUs")
        
        # Set process title for monitoring
        try:
            import setproctitle
            setproctitle.setproctitle(f'Yuhao_evaluate_{os.path.basename(file_path)}')
        except ImportError:
            pass
        
        # Run evaluate.py as subprocess
        cmd = [sys.executable, 'evaluate.py', file_path]
        logger.info(f"[PID {process_id}] Running command: {' '.join(cmd)}")
        
        # Create log directory if it doesn't exist
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)
        
        # Generate log file name from the JSON file name
        json_basename = os.path.basename(file_path)
        task_id = os.path.splitext(json_basename)[0]  # Remove .json extension
        log_file_path = log_dir_path / f"{task_id}.txt"
        
        logger.info(f"[PID {process_id}] Logging output to: {log_file_path}")
        
        # Open log file for writing
        with open(log_file_path, 'w', encoding='utf-8') as log_file:
            # Write command header
            log_file.write(f"Command: {' '.join(cmd)}\n")
            log_file.write(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write("="*80 + "\n\n")
            
            # Run the subprocess and capture output
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=3600  # 1 hour timeout per file
            )
            
            # Write stdout and stderr to log file
            if result.stdout:
                log_file.write("STDOUT:\n")
                log_file.write(result.stdout)
                log_file.write("\n")
            
            if result.stderr:
                log_file.write("STDERR:\n")
                log_file.write(result.stderr)
                log_file.write("\n")
            
            # Write completion info
            log_file.write("\n" + "="*80 + "\n")
            log_file.write(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"Return code: {result.returncode}\n")
            log_file.write(f"Duration: {time.time() - start_time:.1f} seconds\n")
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            logger.info(f"[PID {process_id}] ✅ Successfully processed {os.path.basename(file_path)} in {duration:.1f}s")
            return {
                'file_path': file_path,
                'status': 'success',
                'duration': duration,
                'stdout': result.stdout[-1000:],  # Last 1000 chars for debugging
                'stderr': result.stderr[-1000:] if result.stderr else '',
                'gpu_id': gpu_id
            }
        else:
            logger.error(f"[PID {process_id}] ❌ Failed to process {os.path.basename(file_path)}")
            logger.error(f"[PID {process_id}] Error output: {result.stderr}")
            return {
                'file_path': file_path,
                'status': 'error',
                'duration': duration,
                'stdout': result.stdout[-1000:],
                'stderr': result.stderr[-1000:] if result.stderr else '',
                'error_code': result.returncode,
                'gpu_id': gpu_id
            }
            
    except subprocess.TimeoutExpired:
        logger.error(f"[PID {process_id}] ⏰ Timeout processing {os.path.basename(file_path)}")
        return {
            'file_path': file_path,
            'status': 'timeout',
            'duration': time.time() - start_time,
            'error': 'Process timeout after 1 hour',
            'gpu_id': gpu_id
        }
    except Exception as e:
        logger.error(f"[PID {process_id}] 💥 Exception processing {os.path.basename(file_path)}: {str(e)}")
        return {
            'file_path': file_path,
            'status': 'exception',
            'duration': time.time() - start_time,
            'error': str(e),
            'gpu_id': gpu_id
        }


def get_json_files(directory: str) -> List[str]:
    """Get list of JSON files to process."""
    json_files = []
    directory_path = Path(directory)
    
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    for file_path in directory_path.glob("*.json"):
        if file_path.is_file():
            json_files.append(str(file_path))
    
    json_files.sort()  # Process in alphabetical order
    logger.info(f"Found {len(json_files)} JSON files to process")
    return json_files


def filter_processed_files(json_files: List[str], results_dir: str = "../results/train_gemini/") -> List[str]:
    """Filter out files that have already been processed successfully."""
    unprocessed_files = []
    results_path = Path(results_dir)
    
    if not results_path.exists():
        results_path.mkdir(parents=True, exist_ok=True)
        return json_files
    
    for json_file in json_files:
        base_name = os.path.basename(json_file)
        result_file = results_path / f"results_{base_name}"
        
        if result_file.exists():
            try:
                # Check if result file is complete and valid
                with open(result_file, 'r', encoding='utf-8') as f:
                    result_data = json.load(f)
                
                # Check if processing was completed successfully
                summary = result_data.get('summary', {})
                total_iterations = summary.get('total_iterations', 0)
                processed_iterations = summary.get('processed_iterations', 0)
                total_paragraphs = summary.get('total_paragraphs', 0)
                processed_paragraphs = summary.get('processed_paragraphs', 0)
                
                if (total_iterations > 0 and processed_iterations == total_iterations and
                    total_paragraphs > 0 and processed_paragraphs == total_paragraphs):
                    logger.info(f"⏭️  Skipping already processed file: {base_name}")
                    continue
                else:
                    logger.info(f"📝 Re-processing incomplete file: {base_name}")
                    
            except Exception as e:
                logger.warning(f"⚠️  Could not validate result file for {base_name}: {e}")
        
        unprocessed_files.append(json_file)
    
    logger.info(f"📊 {len(unprocessed_files)} files need processing, {len(json_files) - len(unprocessed_files)} already completed")
    return unprocessed_files


def smart_gpu_assignment(num_files: int, max_workers: int) -> List[Optional[int]]:
    """
    Assign GPU IDs to processes intelligently.
    
    Args:
        num_files: Number of files to process
        max_workers: Maximum number of parallel workers
        
    Returns:
        List of GPU IDs to assign to each worker (None for automatic assignment)
    """
    try:
        # Get number of available GPUs
        result = subprocess.run(
            ['nvidia-smi', '--list-gpus'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            num_gpus = len(result.stdout.strip().split('\n'))
            logger.info(f"🎮 Detected {num_gpus} GPUs")
            
            # If we have enough GPUs, assign one per worker
            if num_gpus >= max_workers:
                gpu_assignments = list(range(max_workers))
                logger.info(f"🎯 Assigning dedicated GPUs: {gpu_assignments}")
                return gpu_assignments
            else:
                # Round-robin assignment for more workers than GPUs
                gpu_assignments = [i % num_gpus for i in range(max_workers)]
                logger.info(f"🔄 Round-robin GPU assignment: {gpu_assignments}")
                return gpu_assignments
        else:
            logger.warning("Could not detect GPUs, using automatic assignment")
            return [None] * max_workers
            
    except Exception as e:
        logger.warning(f"GPU detection failed: {e}, using automatic assignment")
        return [None] * max_workers


def save_progress_report(results: List[Dict[str, Any]], output_file: str):
    """Save processing progress report."""
    total_files = len(results)
    successful = len([r for r in results if r['status'] == 'success'])
    failed = len([r for r in results if r['status'] in ['error', 'timeout', 'exception']])
    
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_files': total_files,
        'successful': successful,
        'failed': failed,
        'success_rate': f"{(successful/total_files*100):.1f}%" if total_files > 0 else "0.0%",
        'total_duration': sum(r.get('duration', 0) for r in results),
        'results': results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    logger.info(f"📊 Progress report saved to: {output_file}")


async def process_files_parallel(
    json_files: List[str],
    max_workers: int = 4,
    batch_size: int = None,
    max_gpu_memory_percent: float = 85.0,
    log_dir: str = "../log/train_gemini"
) -> List[Dict[str, Any]]:
    """
    Process JSON files in parallel with GPU memory management.
    
    Args:
        json_files: List of JSON file paths to process
        max_workers: Maximum number of parallel workers
        batch_size: Number of files to process in each batch (None for all at once)
        max_gpu_memory_percent: Maximum GPU memory usage before throttling
        log_dir: Directory to store log files
        
    Returns:
        List of processing results
    """
    manager = ProcessManager(max_workers, max_gpu_memory_percent)
    manager.gpu_monitor.start_monitoring()
    
    all_results = []
    
    try:
        # Determine batch size
        if batch_size is None:
            batch_size = len(json_files)
        
        # Process files in batches
        for batch_start in range(0, len(json_files), batch_size):
            batch_end = min(batch_start + batch_size, len(json_files))
            batch_files = json_files[batch_start:batch_end]
            
            logger.info(f"🚀 Processing batch {batch_start//batch_size + 1}: files {batch_start+1}-{batch_end}")
            
            # Get GPU assignments for this batch
            gpu_assignments = smart_gpu_assignment(len(batch_files), max_workers)
            
            # Process batch with ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit tasks with GPU assignments
                future_to_file = {}
                for i, file_path in enumerate(batch_files):
                    # Wait for GPU memory if needed
                    if not manager.wait_for_gpu_memory():
                        logger.warning("GPU memory timeout, proceeding anyway...")
                    
                    gpu_id = gpu_assignments[i % len(gpu_assignments)] if gpu_assignments[0] is not None else None
                    future = executor.submit(process_single_file, file_path, gpu_id, log_dir)
                    future_to_file[future] = file_path
                
                # Collect results as they complete
                batch_results = []
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        result = future.result()
                        batch_results.append(result)
                        
                        # Log progress
                        completed = len(batch_results)
                        total_batch = len(batch_files)
                        logger.info(f"📈 Batch progress: {completed}/{total_batch} files completed")
                        
                    except Exception as e:
                        logger.error(f"❌ Error processing {os.path.basename(file_path)}: {e}")
                        batch_results.append({
                            'file_path': file_path,
                            'status': 'exception',
                            'error': str(e),
                            'duration': 0
                        })
            
            all_results.extend(batch_results)
            
            # Save intermediate progress
            progress_file = f"../results/train_gemini/progress_report_batch_{batch_start//batch_size + 1}.json"
            save_progress_report(all_results, progress_file)
            
            # Memory cleanup between batches
            if batch_end < len(json_files):
                logger.info("🧹 Memory cleanup between batches...")
                time.sleep(5)  # Allow memory to clear
    
    finally:
        manager.gpu_monitor.stop_monitoring()
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Parallel processing script for deerflow JSON files'
    )
    parser.add_argument(
        'directory',
        default='/data2/yuhaoz/DeepResearch/HalluBench/data/train/close-source/gemini/json',
        nargs='?',
        help='Directory containing JSON files to process (default: deerflow/)'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=4,
        help='Maximum number of parallel workers (default: 2)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Number of files to process in each batch (default: all at once)'
    )
    parser.add_argument(
        '--max-gpu-memory',
        type=float,
        default=85.0,
        help='Maximum GPU memory usage percentage before throttling (default: 85.0)'
    )
    parser.add_argument(
        '--skip-processed',
        action='store_true',
        help='Skip files that have already been processed successfully'
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default='../log/train_gemini',
        help='Directory to store log files (default: ../log/train_gemini)'
    )
    
    args = parser.parse_args()
    
    # Validate directory
    if not os.path.exists(args.directory):
        logger.error(f"❌ Directory not found: {args.directory}")
        sys.exit(1)
    
    # Change to scripts directory to run evaluate.py
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    if not os.path.exists('evaluate.py'):
        logger.error("❌ evaluate.py not found in current directory")
        sys.exit(1)
    
    logger.info(f"🎯 Starting parallel processing with {args.max_workers} workers")
    logger.info(f"📁 Processing directory: {args.directory}")
    logger.info(f"🎮 GPU memory limit: {args.max_gpu_memory}%")
    
    # Create log directory
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"📝 Log files will be saved to: {log_dir.absolute()}")
    logger.info(f"📝 Each JSON file will have its own log file: <task_id>.txt")
    
    start_time = time.time()
    
    try:
        # Get list of JSON files
        json_files = get_json_files(args.directory)
        json_files = json_files[:1]
        
        if not json_files:
            logger.warning("⚠️  No JSON files found in directory")
            return
        
        # Filter already processed files if requested
        if args.skip_processed:
            json_files = filter_processed_files(json_files)
            
            if not json_files:
                logger.info("🎉 All files have already been processed!")
                return
        
        # Process files in parallel
        results = asyncio.run(process_files_parallel(
            json_files,
            max_workers=args.max_workers,
            batch_size=args.batch_size,
            max_gpu_memory_percent=args.max_gpu_memory,
            log_dir=args.log_dir
        ))
        
        # Calculate final statistics
        end_time = time.time()
        total_duration = end_time - start_time
        
        successful = len([r for r in results if r['status'] == 'success'])
        failed = len([r for r in results if r['status'] != 'success'])
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🎉 PROCESSING COMPLETED")
        logger.info(f"{'='*60}")
        logger.info(f"📊 Total files: {len(results)}")
        logger.info(f"✅ Successful: {successful}")
        logger.info(f"❌ Failed: {failed}")
        logger.info(f"📈 Success rate: {(successful/len(results)*100):.1f}%")
        logger.info(f"⏱️  Total time: {total_duration/60:.1f} minutes")
        logger.info(f"🚀 Average time per file: {total_duration/len(results):.1f} seconds")
        
        # Save final report
        final_report_file = "../results/train_gemini/final_processing_report.json"
        save_progress_report(results, final_report_file)
        
        # Show failed files
        failed_files = [r for r in results if r['status'] != 'success']
        if failed_files:
            logger.warning(f"\n❌ Failed files ({len(failed_files)}):")
            for result in failed_files:
                file_name = os.path.basename(result['file_path'])
                status = result['status']
                error = result.get('error', result.get('stderr', 'Unknown error'))
                logger.warning(f"  - {file_name}: {status} - {error[:100]}")
        
    except KeyboardInterrupt:
        logger.info("\n⏹️  Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()