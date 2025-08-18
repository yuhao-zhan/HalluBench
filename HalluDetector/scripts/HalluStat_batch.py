import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Any, Tuple
import os
from collections import defaultdict
import pandas as pd
import glob

class BatchHallucinationStatistics:
    """
    Batch processing of hallucination detection results across multiple JSON files.
    
    Analyzes:
    1. H1: Action contradictions with observation memory
    2. h2: Fact fabrication (neutral judgments)
    3. h3: Fact contradiction 
    4. h4: Noise domination
    
    Creates overall statistics across all files in a directory.
    """
    
    def __init__(self, results_directory: str):
        """Initialize with directory containing multiple results JSON files."""
        self.results_directory = results_directory
        self.all_files_stats = []
        self.overall_summary = {}
    
    def _get_json_files(self) -> List[str]:
        """Get all JSON files in the results directory."""
        pattern = os.path.join(self.results_directory, "*.json")
        json_files = glob.glob(pattern)
        # Filter out non-result files (like progress reports)
        result_files = [f for f in json_files if "results_" in f and not f.endswith("_report.json")]
        return result_files
    
    def _load_single_file(self, file_path: str) -> Dict[str, Any]:
        """Load results from a single JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return {}
    
    def _analyze_single_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a single JSON file and return its statistics."""
        data = self._load_single_file(file_path)
        if not data:
            return {}
        
        # Extract filename for identification
        filename = os.path.basename(file_path)
        
        # Initialize counters for this file
        file_stats = {
            'filename': filename,
            'h1_count': 0,
            'h2_count': 0,
            'h3_count': 0,
            'h4_count': 0,
            'total_actions': 0,
            'total_claims': 0,
            'entailments': 0
        }
        
        # Analyze chain of research iterations
        chain_results = data.get('chain_of_research_results', [])
        for iteration in chain_results:
            # Analyze action judgments (H1)
            action_judgments = iteration.get('action_judgments', [])
            for judgment in action_judgments:
                file_stats['total_actions'] += 1
                
                if isinstance(judgment, dict):
                    if judgment.get('judgment') == 'contradiction':
                        file_stats['h1_count'] += 1
                elif isinstance(judgment, str) and judgment == 'contradiction':
                    file_stats['h1_count'] += 1
            
            # Analyze claim results (h2, h3)
            claim_results = iteration.get('claim_results', [])
            for claim_result in claim_results:
                if isinstance(claim_result, dict):
                    file_stats['total_claims'] += 1
                    final_judgment = claim_result.get('final_judgment', 'unknown')
                    
                    if final_judgment == 'neutral':
                        file_stats['h2_count'] += 1
                    elif final_judgment == 'contradiction':
                        file_stats['h3_count'] += 1
                    elif final_judgment == 'entailment':
                        file_stats['entailments'] += 1
            
            # Count noise domination (h4)
            noise_results = iteration.get('noise_results', [])
            file_stats['h4_count'] += len(noise_results)
        
        # Analyze report paragraphs
        report_results = data.get('report_results', [])
        for paragraph in report_results:
            claim_results = paragraph.get('claim_results', [])
            noise_results = paragraph.get('noise_results', [])
            
            for claim_result in claim_results:
                if isinstance(claim_result, dict):
                    file_stats['total_claims'] += 1
                    final_judgment = claim_result.get('final_judgment', 'unknown')
                    
                    if final_judgment == 'neutral':
                        file_stats['h2_count'] += 1
                    elif final_judgment == 'contradiction':
                        file_stats['h3_count'] += 1
                    elif final_judgment == 'entailment':
                        file_stats['entailments'] += 1
            
            file_stats['h4_count'] += len(noise_results)
        
        # Calculate ratios for this file
        file_stats['h1_ratio'] = file_stats['h1_count'] / file_stats['total_actions'] if file_stats['total_actions'] > 0 else 0
        total_claim_hallucinations = file_stats['h2_count'] + file_stats['h3_count'] + file_stats['h4_count']
        file_stats['claim_hallucination_ratio'] = total_claim_hallucinations / file_stats['total_claims'] if file_stats['total_claims'] > 0 else 0
        file_stats['overall_hallucination_ratio'] = (file_stats['h1_count'] + total_claim_hallucinations) / (file_stats['total_actions'] + file_stats['total_claims']) if (file_stats['total_actions'] + file_stats['total_claims']) > 0 else 0
        
        return file_stats
    
    def process_all_files(self) -> None:
        """Process all JSON files in the directory."""
        json_files = self._get_json_files()
        print(f"Found {len(json_files)} result files to process...")
        
        for file_path in json_files:
            print(f"Processing: {os.path.basename(file_path)}")
            file_stats = self._analyze_single_file(file_path)
            if file_stats:
                self.all_files_stats.append(file_stats)
        
        print(f"Successfully processed {len(self.all_files_stats)} files")
        self._calculate_overall_summary()
    
    def _calculate_overall_summary(self) -> None:
        """Calculate overall statistics across all files."""
        if not self.all_files_stats:
            return
        
        # Calculate averages across all files
        total_files = len(self.all_files_stats)
        
        avg_h1_ratio = np.mean([stats['h1_ratio'] for stats in self.all_files_stats])
        avg_claim_hallucination_ratio = np.mean([stats['claim_hallucination_ratio'] for stats in self.all_files_stats])
        avg_overall_hallucination_ratio = np.mean([stats['overall_hallucination_ratio'] for stats in self.all_files_stats])
        
        # Calculate standard deviations
        std_h1_ratio = np.std([stats['h1_ratio'] for stats in self.all_files_stats])
        std_claim_hallucination_ratio = np.std([stats['claim_hallucination_ratio'] for stats in self.all_files_stats])
        std_overall_hallucination_ratio = np.std([stats['overall_hallucination_ratio'] for stats in self.all_files_stats])
        
        # Calculate medians
        median_h1_ratio = np.median([stats['h1_ratio'] for stats in self.all_files_stats])
        median_claim_hallucination_ratio = np.median([stats['claim_hallucination_ratio'] for stats in self.all_files_stats])
        median_overall_hallucination_ratio = np.median([stats['overall_hallucination_ratio'] for stats in self.all_files_stats])
        
        self.overall_summary = {
            'total_files_processed': total_files,
            'averages': {
                'h1_ratio': avg_h1_ratio,
                'claim_hallucination_ratio': avg_claim_hallucination_ratio,
                'overall_hallucination_ratio': avg_overall_hallucination_ratio
            },
            'standard_deviations': {
                'h1_ratio': std_h1_ratio,
                'claim_hallucination_ratio': std_claim_hallucination_ratio,
                'overall_hallucination_ratio': std_overall_hallucination_ratio
            },
            'medians': {
                'h1_ratio': median_h1_ratio,
                'claim_hallucination_ratio': median_claim_hallucination_ratio,
                'overall_hallucination_ratio': median_overall_hallucination_ratio
            }
        }
    
    def print_summary_statistics(self) -> None:
        """Print summary statistics across all files."""
        if not self.overall_summary:
            print("No statistics available. Run process_all_files() first.")
            return
        
        print("=" * 80)
        print("BATCH HALLUCINATION STATISTICS SUMMARY")
        print("=" * 80)
        print(f"Total files processed: {self.overall_summary['total_files_processed']}")
        
        print(f"\nAVERAGE HALLUCINATION RATIOS:")
        print(f"H1 Ratio (Action Contradictions): {self.overall_summary['averages']['h1_ratio']:.4f} ± {self.overall_summary['standard_deviations']['h1_ratio']:.4f}")
        print(f"Claim Hallucination Ratio: {self.overall_summary['averages']['claim_hallucination_ratio']:.4f} ± {self.overall_summary['standard_deviations']['claim_hallucination_ratio']:.4f}")
        print(f"Overall Hallucination Ratio: {self.overall_summary['averages']['overall_hallucination_ratio']:.4f} ± {self.overall_summary['standard_deviations']['overall_hallucination_ratio']:.4f}")
        
        print(f"\nMEDIAN HALLUCINATION RATIOS:")
        print(f"H1 Ratio: {self.overall_summary['medians']['h1_ratio']:.4f}")
        print(f"Claim Hallucination Ratio: {self.overall_summary['medians']['claim_hallucination_ratio']:.4f}")
        print(f"Overall Hallucination Ratio: {self.overall_summary['medians']['overall_hallucination_ratio']:.4f}")
    
    def create_overall_visualization(self, output_dir: str = "batch_hallucination_analysis") -> None:
        """Create a single comprehensive visualization showing hallucination distribution across all files."""
        if not self.all_files_stats:
            print("No data available for visualization. Run process_all_files() first.")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create a large figure with multiple subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Overall Hallucination Statistics Distribution Across All Files', fontsize=20, fontweight='bold')
        
        # Extract data for plotting
        filenames = [stats['filename'].replace('results_', '').replace('.json', '') for stats in self.all_files_stats]
        h1_ratios = [stats['h1_ratio'] for stats in self.all_files_stats]
        claim_hallucination_ratios = [stats['claim_hallucination_ratio'] for stats in self.all_files_stats]
        overall_hallucination_ratios = [stats['overall_hallucination_ratio'] for stats in self.all_files_stats]
        
        # 1. Overall Hallucination Ratio Distribution (Main plot)
        ax1.hist(overall_hallucination_ratios, bins=20, alpha=0.7, color='red', edgecolor='black')
        ax1.axvline(np.mean(overall_hallucination_ratios), color='blue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(overall_hallucination_ratios):.4f}')
        ax1.axvline(np.median(overall_hallucination_ratios), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(overall_hallucination_ratios):.4f}')
        ax1.set_xlabel('Overall Hallucination Ratio', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Distribution of Overall Hallucination Ratios', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. H1 vs Claim Hallucination Ratios Scatter Plot
        ax2.scatter(h1_ratios, claim_hallucination_ratios, alpha=0.6, s=50)
        ax2.set_xlabel('H1 Ratio (Action Contradictions)', fontsize=12)
        ax2.set_ylabel('Claim Hallucination Ratio', fontsize=12)
        ax2.set_title('H1 vs Claim Hallucination Ratios', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        correlation = np.corrcoef(h1_ratios, claim_hallucination_ratios)[0, 1]
        ax2.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=ax2.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 3. Box Plot of All Ratios
        ratio_data = [h1_ratios, claim_hallucination_ratios, overall_hallucination_ratios]
        labels = ['H1 Ratio\n(Actions)', 'Claim\nHallucination', 'Overall\nHallucination']
        bp = ax3.boxplot(ratio_data, labels=labels, patch_artist=True)
        
        # Color the boxes
        colors = ['lightcoral', 'lightblue', 'lightgreen']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax3.set_ylabel('Ratio Value', fontsize=12)
        ax3.set_title('Box Plot of All Hallucination Ratios', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Top 10 Files by Overall Hallucination Ratio
        # Sort files by overall hallucination ratio
        sorted_stats = sorted(self.all_files_stats, key=lambda x: x['overall_hallucination_ratio'], reverse=True)
        top_10_files = sorted_stats[:10]
        top_10_names = [stats['filename'].replace('results_', '').replace('.json', '') for stats in top_10_files]
        top_10_ratios = [stats['overall_hallucination_ratio'] for stats in top_10_files]
        
        bars = ax4.barh(range(len(top_10_names)), top_10_ratios, color='orange', alpha=0.7)
        ax4.set_yticks(range(len(top_10_names)))
        ax4.set_yticklabels(top_10_names, fontsize=10)
        ax4.set_xlabel('Overall Hallucination Ratio', fontsize=12)
        ax4.set_title('Top 10 Files by Hallucination Ratio', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, ratio) in enumerate(zip(bars, top_10_ratios)):
            ax4.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{ratio:.3f}', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        # Save the comprehensive figure
        output_file = f"{output_dir}/overall_hallucination_distribution.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Overall visualization saved to: {output_file}")
    
    def save_batch_report(self, output_file: str = "batch_hallucination_report.json") -> None:
        """Save batch statistics report to JSON."""
        # Sort individual file stats by overall hallucination ratio in descending order
        sorted_file_stats = sorted(self.all_files_stats, key=lambda x: x['overall_hallucination_ratio'], reverse=True)
        
        report = {
            'overall_summary': self.overall_summary,
            'individual_file_stats': sorted_file_stats
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"📄 Batch report saved to: {output_file}")
    
    def run_batch_analysis(self, output_dir: str = "../hallucination_analysis") -> None:
        """Run complete batch hallucination analysis."""
        print("🔍 Starting batch hallucination analysis...")
        
        # Process all files
        self.process_all_files()
        
        # Print summary statistics
        self.print_summary_statistics()
        
        # Create overall visualization
        print("\n📊 Creating overall visualization...")
        self.create_overall_visualization(output_dir)
        
        # Save batch report
        report_file = f"{output_dir}/batch_hallucination_report.json"
        self.save_batch_report(report_file)
        
        print(f"\n✅ Batch analysis finished! Results saved to: {output_dir}/")


# Example usage
def analyze_batch_hallucinations(results_directory: str):
    """
    Main function to analyze hallucinations from multiple evaluation result files.
    
    Args:
        results_directory: Path to directory containing results JSON files
    """
    analyzer = BatchHallucinationStatistics(results_directory)
    analyzer.run_batch_analysis()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python HalluStat_batch.py <results_directory_path>")
        print("Example: python HalluStat_batch.py ../results/deerflow_mind2web2")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    analyze_batch_hallucinations(results_dir)