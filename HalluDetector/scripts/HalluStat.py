import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Any, Tuple
import os
from collections import defaultdict
import pandas as pd

class HallucinationStatistics:
    """
    Comprehensive statistical analysis for hallucination detection across the evaluation pipeline.
    
    Analyzes:
    1. H1: Action contradictions with observation memory
    2. h2: Fact fabrication (neutral judgments)
    3. h3: Fact contradiction 
    4. h4: Noise domination
    """
    
    def __init__(self, results_file_path: str):
        """Initialize with results from the evaluation pipeline."""
        self.results_file_path = results_file_path
        self.data = self._load_results()
        self.stats = self._initialize_stats()
    
    def _load_results(self) -> Dict[str, Any]:
        """Load results from the evaluation pipeline JSON file."""
        try:
            with open(self.results_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading results file: {e}")
            return {}
    
    def _initialize_stats(self) -> Dict[str, Any]:
        """Initialize statistics structure."""
        return {
            'iteration_stats': {
                'h1_count': 0,  # Action contradictions
                'h2_count': 0,  # Fact fabrication
                'h3_count': 0,  # Fact contradiction
                'h4_count': 0,  # Noise domination
                'total_actions': 0,
                'total_claims': 0,
                'entailments': 0
            },
            'report_stats': {
                'h2_count': 0,  # Fact fabrication
                'h3_count': 0,  # Fact contradiction
                'h4_count': 0,  # Noise domination
                'total_claims': 0,
                'entailments': 0
            },
            'overall_stats': {},
            'detailed_breakdown': {
                'iterations': [],
                'paragraphs': []
            }
        }
    
    def analyze_iterations(self) -> None:
        """Analyze hallucinations in chain of research iterations."""
        chain_results = self.data.get('chain_of_research_results', [])
        
        for i, iteration in enumerate(chain_results):
            iteration_breakdown = {
                'iteration_id': i + 1,
                'h1_count': 0,
                'h2_count': 0,
                'h3_count': 0,
                'h4_count': 0,
                'total_actions': 0,
                'total_claims': 0,
                'entailments': 0
            }
            
            # Analyze action judgments (H1)
            action_judgments = iteration.get('action_judgments', [])
            for judgment in action_judgments:
                self.stats['iteration_stats']['total_actions'] += 1
                iteration_breakdown['total_actions'] += 1
                
                # Handle both string and dict formats for judgment
                if isinstance(judgment, dict):
                    if judgment.get('judgment') == 'contradiction':
                        self.stats['iteration_stats']['h1_count'] += 1
                        iteration_breakdown['h1_count'] += 1
                elif isinstance(judgment, str) and judgment == 'contradiction':
                    self.stats['iteration_stats']['h1_count'] += 1
                    iteration_breakdown['h1_count'] += 1
            
            # Analyze claim results (h2, h3, h4)
            claim_results = iteration.get('claim_results', [])
            noise_results = iteration.get('noise_results', [])
            
            for claim_result in claim_results:
                if isinstance(claim_result, dict):
                    self.stats['iteration_stats']['total_claims'] += 1
                    iteration_breakdown['total_claims'] += 1
                    
                    final_judgment = claim_result.get('final_judgment', 'unknown')
                    
                    if final_judgment == 'neutral':
                        self.stats['iteration_stats']['h2_count'] += 1
                        iteration_breakdown['h2_count'] += 1
                    elif final_judgment == 'contradiction':
                        self.stats['iteration_stats']['h3_count'] += 1
                        iteration_breakdown['h3_count'] += 1
                    elif final_judgment == 'entailment':
                        self.stats['iteration_stats']['entailments'] += 1
                        iteration_breakdown['entailments'] += 1
            
            # Count noise domination (h4)
            h4_count = len(noise_results)
            self.stats['iteration_stats']['h4_count'] += h4_count
            iteration_breakdown['h4_count'] = h4_count
            
            self.stats['detailed_breakdown']['iterations'].append(iteration_breakdown)
    
    def analyze_report(self) -> None:
        """Analyze hallucinations in report paragraphs."""
        report_results = self.data.get('report_results', [])
        
        for i, paragraph in enumerate(report_results):
            paragraph_breakdown = {
                'paragraph_id': i + 1,
                'h2_count': 0,
                'h3_count': 0,
                'h4_count': 0,
                'total_claims': 0,
                'entailments': 0
            }
            
            # Analyze claim results
            claim_results = paragraph.get('claim_results', [])
            noise_results = paragraph.get('noise_results', [])
            
            for claim_result in claim_results:
                if isinstance(claim_result, dict):
                    self.stats['report_stats']['total_claims'] += 1
                    paragraph_breakdown['total_claims'] += 1
                    
                    final_judgment = claim_result.get('final_judgment', 'unknown')
                    
                    if final_judgment == 'neutral':
                        self.stats['report_stats']['h2_count'] += 1
                        paragraph_breakdown['h2_count'] += 1
                    elif final_judgment == 'contradiction':
                        self.stats['report_stats']['h3_count'] += 1
                        paragraph_breakdown['h3_count'] += 1
                    elif final_judgment == 'entailment':
                        self.stats['report_stats']['entailments'] += 1
                        paragraph_breakdown['entailments'] += 1
            
            # Count noise domination (h4)
            h4_count = len(noise_results)
            self.stats['report_stats']['h4_count'] += h4_count
            paragraph_breakdown['h4_count'] = h4_count
            
            self.stats['detailed_breakdown']['paragraphs'].append(paragraph_breakdown)
    
    def calculate_ratios(self) -> None:
        """Calculate all hallucination ratios."""
        # Iteration ratios
        iter_stats = self.stats['iteration_stats']
        
        # H1 ratio: H1 count / total actions
        h1_ratio = iter_stats['h1_count'] / iter_stats['total_actions'] if iter_stats['total_actions'] > 0 else 0
        
        # Overall claim hallucination ratio for iterations
        total_claim_hallucinations_iter = iter_stats['h2_count'] + iter_stats['h3_count'] + iter_stats['h4_count']
        iter_claim_ratio = total_claim_hallucinations_iter / iter_stats['total_claims'] if iter_stats['total_claims'] > 0 else 0
        
        # Individual claim hallucination ratios for iterations
        h2_ratio_iter = iter_stats['h2_count'] / total_claim_hallucinations_iter if total_claim_hallucinations_iter > 0 else 0
        h3_ratio_iter = iter_stats['h3_count'] / total_claim_hallucinations_iter if total_claim_hallucinations_iter > 0 else 0
        h4_ratio_iter = iter_stats['h4_count'] / total_claim_hallucinations_iter if total_claim_hallucinations_iter > 0 else 0
        
        # Report ratios
        report_stats = self.stats['report_stats']
        
        # Overall claim hallucination ratio for report
        total_claim_hallucinations_report = report_stats['h2_count'] + report_stats['h3_count'] + report_stats['h4_count']
        report_claim_ratio = total_claim_hallucinations_report / report_stats['total_claims'] if report_stats['total_claims'] > 0 else 0
        
        # Individual claim hallucination ratios for report
        h2_ratio_report = report_stats['h2_count'] / total_claim_hallucinations_report if total_claim_hallucinations_report > 0 else 0
        h3_ratio_report = report_stats['h3_count'] / total_claim_hallucinations_report if total_claim_hallucinations_report > 0 else 0
        h4_ratio_report = report_stats['h4_count'] / total_claim_hallucinations_report if total_claim_hallucinations_report > 0 else 0
        
        # Overall statistics (combining iterations and report)
        total_actions = iter_stats['total_actions']
        total_claims = iter_stats['total_claims'] + report_stats['total_claims']
        total_h1 = iter_stats['h1_count']
        total_h2 = iter_stats['h2_count'] + report_stats['h2_count']
        total_h3 = iter_stats['h3_count'] + report_stats['h3_count']
        total_h4 = iter_stats['h4_count'] + report_stats['h4_count']
        total_claim_hallucinations = total_h2 + total_h3 + total_h4
        
        # Overall ratios
        overall_h1_ratio = total_h1 / total_actions if total_actions > 0 else 0
        overall_claim_ratio = total_claim_hallucinations / total_claims if total_claims > 0 else 0
        overall_h2_ratio = total_h2 / total_claim_hallucinations if total_claim_hallucinations > 0 else 0
        overall_h3_ratio = total_h3 / total_claim_hallucinations if total_claim_hallucinations > 0 else 0
        overall_h4_ratio = total_h4 / total_claim_hallucinations if total_claim_hallucinations > 0 else 0
        
        self.stats['overall_stats'] = {
            # Counts
            'total_actions': total_actions,
            'total_claims': total_claims,
            'h1_count': total_h1,
            'h2_count': total_h2,
            'h3_count': total_h3,
            'h4_count': total_h4,
            'total_claim_hallucinations': total_claim_hallucinations,
            
            # Iteration ratios
            'iteration_h1_ratio': h1_ratio,
            'iteration_claim_hallucination_ratio': iter_claim_ratio,
            'iteration_h2_ratio': h2_ratio_iter,
            'iteration_h3_ratio': h3_ratio_iter,
            'iteration_h4_ratio': h4_ratio_iter,
            
            # Report ratios
            'report_claim_hallucination_ratio': report_claim_ratio,
            'report_h2_ratio': h2_ratio_report,
            'report_h3_ratio': h3_ratio_report,
            'report_h4_ratio': h4_ratio_report,
            
            # Overall ratios
            'overall_h1_ratio': overall_h1_ratio,
            'overall_claim_hallucination_ratio': overall_claim_ratio,
            'overall_h2_ratio': overall_h2_ratio,
            'overall_h3_ratio': overall_h3_ratio,
            'overall_h4_ratio': overall_h4_ratio
        }
    
    def print_statistics(self) -> None:
        """Print comprehensive statistics."""
        stats = self.stats['overall_stats']
        
        print("=" * 80)
        print("GLOBAL HALLUCINATION STATISTICS")
        print("=" * 80)
        
        print(f"\nOVERALL COUNTS:")
        print(f"Total Actions: {stats['total_actions']}")
        print(f"Total Claims: {stats['total_claims']}")
        print(f"H1 (Action Contradictions): {stats['h1_count']}")
        print(f"h2 (Fact Fabrication): {stats['h2_count']}")
        print(f"h3 (Fact Contradiction): {stats['h3_count']}")
        print(f"h4 (Noise Domination): {stats['h4_count']}")
        print(f"Total Claim Hallucinations: {stats['total_claim_hallucinations']}")
        
        print(f"\nOVERALL HALLUCINATION RATIOS:")
        print(f"H1 Ratio (H1/Total Actions): {stats['overall_h1_ratio']:.4f} ({stats['overall_h1_ratio']*100:.2f}%)")
        print(f"Overall Claim Hallucination Ratio: {stats['overall_claim_hallucination_ratio']:.4f} ({stats['overall_claim_hallucination_ratio']*100:.2f}%)")
        
        print(f"\nHALLUCINATION TYPE DISTRIBUTION:")
        print(f"h2 Ratio (among claim hallucinations): {stats['overall_h2_ratio']:.4f} ({stats['overall_h2_ratio']*100:.2f}%)")
        print(f"h3 Ratio (among claim hallucinations): {stats['overall_h3_ratio']:.4f} ({stats['overall_h3_ratio']*100:.2f}%)")
        print(f"h4 Ratio (among claim hallucinations): {stats['overall_h4_ratio']:.4f} ({stats['overall_h4_ratio']*100:.2f}%)")
        
        print(f"\nITERATION-SPECIFIC RATIOS:")
        print(f"Iteration H1 Ratio: {stats['iteration_h1_ratio']:.4f} ({stats['iteration_h1_ratio']*100:.2f}%)")
        print(f"Iteration Claim Hallucination Ratio: {stats['iteration_claim_hallucination_ratio']:.4f} ({stats['iteration_claim_hallucination_ratio']*100:.2f}%)")
        print(f"Iteration h2 Ratio: {stats['iteration_h2_ratio']:.4f} ({stats['iteration_h2_ratio']*100:.2f}%)")
        print(f"Iteration h3 Ratio: {stats['iteration_h3_ratio']:.4f} ({stats['iteration_h3_ratio']*100:.2f}%)")
        print(f"Iteration h4 Ratio: {stats['iteration_h4_ratio']:.4f} ({stats['iteration_h4_ratio']*100:.2f}%)")
        
        print(f"\nREPORT-SPECIFIC RATIOS:")
        print(f"Report Claim Hallucination Ratio: {stats['report_claim_hallucination_ratio']:.4f} ({stats['report_claim_hallucination_ratio']*100:.2f}%)")
        print(f"Report h2 Ratio: {stats['report_h2_ratio']:.4f} ({stats['report_h2_ratio']*100:.2f}%)")
        print(f"Report h3 Ratio: {stats['report_h3_ratio']:.4f} ({stats['report_h3_ratio']*100:.2f}%)")
        print(f"Report h4 Ratio: {stats['report_h4_ratio']:.4f} ({stats['report_h4_ratio']*100:.2f}%)")
    
    def create_visualizations(self, output_dir: str = "hallucination_plots") -> None:
        """Create comprehensive visualizations of hallucination statistics."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Overall Hallucination Distribution Pie Chart
        self._plot_overall_distribution(output_dir)
        
        # 2. Hallucination Type Distribution
        self._plot_hallucination_types(output_dir)
        
        # 3. Iteration vs Report Comparison
        self._plot_iteration_vs_report(output_dir)
        
        # 4. Per-iteration/paragraph breakdown
        self._plot_detailed_breakdown(output_dir)
        
        # 5. Hallucination Ratios Bar Chart
        self._plot_ratio_comparison(output_dir)
        
        print(f"\n📊 All visualizations saved to: {output_dir}/")
    
    def _plot_overall_distribution(self, output_dir: str) -> None:
        """Plot overall distribution of all hallucination types."""
        stats = self.stats['overall_stats']
        
        # Prepare data
        labels = ['Correct Actions', 'H1 (Action Contradiction)', 
                 'Correct Claims', 'h2 (Fact Fabrication)', 
                 'h3 (Fact Contradiction)', 'h4 (Noise Domination)']
        
        correct_actions = stats['total_actions'] - stats['h1_count']
        correct_claims = stats['total_claims'] - stats['total_claim_hallucinations']
        
        values = [correct_actions, stats['h1_count'], 
                 correct_claims, stats['h2_count'], 
                 stats['h3_count'], stats['h4_count']]
        
        colors = ['lightgreen', 'red', 'lightblue', 'orange', 'purple', 'brown']
        
        plt.figure(figsize=(12, 8))
        wedges, texts, autotexts = plt.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Overall Distribution of Actions and Claims', fontsize=16, fontweight='bold')
        
        # Enhance text readability
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/overall_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_hallucination_types(self, output_dir: str) -> None:
        """Plot distribution of hallucination types."""
        stats = self.stats['overall_stats']
        
        # Hallucination types only
        labels = ['H1\n(Action Contradiction)', 'h2\n(Fact Fabrication)', 
                 'h3\n(Fact Contradiction)', 'h4\n(Noise Domination)']
        values = [stats['h1_count'], stats['h2_count'], stats['h3_count'], stats['h4_count']]
        colors = ['red', 'orange', 'purple', 'brown']
        
        plt.figure(figsize=(10, 8))
        bars = plt.bar(labels, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        plt.title('Distribution of Hallucination Types', fontsize=16, fontweight='bold')
        plt.ylabel('Count', fontsize=12)
        plt.xlabel('Hallucination Type', fontsize=12)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(values),
                    f'{value}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/hallucination_types.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_iteration_vs_report(self, output_dir: str) -> None:
        """Plot comparison between iterations and report."""
        iter_stats = self.stats['iteration_stats']
        report_stats = self.stats['report_stats']
        
        # Calculate ratios
        iter_h2_ratio = iter_stats['h2_count'] / iter_stats['total_claims'] if iter_stats['total_claims'] > 0 else 0
        iter_h3_ratio = iter_stats['h3_count'] / iter_stats['total_claims'] if iter_stats['total_claims'] > 0 else 0
        iter_h4_ratio = iter_stats['h4_count'] / iter_stats['total_claims'] if iter_stats['total_claims'] > 0 else 0
        
        report_h2_ratio = report_stats['h2_count'] / report_stats['total_claims'] if report_stats['total_claims'] > 0 else 0
        report_h3_ratio = report_stats['h3_count'] / report_stats['total_claims'] if report_stats['total_claims'] > 0 else 0
        report_h4_ratio = report_stats['h4_count'] / report_stats['total_claims'] if report_stats['total_claims'] > 0 else 0
        
        # Create grouped bar chart
        categories = ['h2 (Fact Fabrication)', 'h3 (Fact Contradiction)', 'h4 (Noise Domination)']
        iteration_ratios = [iter_h2_ratio, iter_h3_ratio, iter_h4_ratio]
        report_ratios = [report_h2_ratio, report_h3_ratio, report_h4_ratio]
        
        x = np.arange(len(categories))
        width = 0.35
        
        plt.figure(figsize=(12, 8))
        bars1 = plt.bar(x - width/2, iteration_ratios, width, label='Chain of Research', color='lightcoral', alpha=0.8)
        bars2 = plt.bar(x + width/2, report_ratios, width, label='Final Report', color='lightblue', alpha=0.8)
        
        plt.xlabel('Hallucination Type', fontsize=12)
        plt.ylabel('Ratio (Hallucinations/Total Claims)', fontsize=12)
        plt.title('Hallucination Ratios: Chain of Research vs Final Report', fontsize=16, fontweight='bold')
        plt.xticks(x, categories)
        plt.legend()
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/iteration_vs_report.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_detailed_breakdown(self, output_dir: str) -> None:
        """Plot detailed breakdown per iteration/paragraph."""
        # Iteration breakdown
        iter_data = self.stats['detailed_breakdown']['iterations']
        if iter_data:
            iter_ids = [item['iteration_id'] for item in iter_data]
            iter_h1 = [item['h1_count'] for item in iter_data]
            iter_h2 = [item['h2_count'] for item in iter_data]
            iter_h3 = [item['h3_count'] for item in iter_data]
            iter_h4 = [item['h4_count'] for item in iter_data]
            
            plt.figure(figsize=(15, 6))
            width = 0.2
            x = np.arange(len(iter_ids))
            
            plt.bar(x - 1.5*width, iter_h1, width, label='H1', color='red', alpha=0.8)
            plt.bar(x - 0.5*width, iter_h2, width, label='h2', color='orange', alpha=0.8)
            plt.bar(x + 0.5*width, iter_h3, width, label='h3', color='purple', alpha=0.8)
            plt.bar(x + 1.5*width, iter_h4, width, label='h4', color='brown', alpha=0.8)
            
            plt.xlabel('Iteration ID', fontsize=12)
            plt.ylabel('Hallucination Count', fontsize=12)
            plt.title('Hallucination Breakdown by Iteration', fontsize=16, fontweight='bold')
            plt.xticks(x, iter_ids)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{output_dir}/iteration_breakdown.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Paragraph breakdown (first 20 paragraphs for readability)
        para_data = self.stats['detailed_breakdown']['paragraphs'][:20]
        if para_data:
            para_ids = [item['paragraph_id'] for item in para_data]
            para_h2 = [item['h2_count'] for item in para_data]
            para_h3 = [item['h3_count'] for item in para_data]
            para_h4 = [item['h4_count'] for item in para_data]
            
            plt.figure(figsize=(15, 6))
            width = 0.25
            x = np.arange(len(para_ids))
            
            plt.bar(x - width, para_h2, width, label='h2', color='orange', alpha=0.8)
            plt.bar(x, para_h3, width, label='h3', color='purple', alpha=0.8)
            plt.bar(x + width, para_h4, width, label='h4', color='brown', alpha=0.8)
            
            plt.xlabel('Paragraph ID (First 20)', fontsize=12)
            plt.ylabel('Hallucination Count', fontsize=12)
            plt.title('Hallucination Breakdown by Report Paragraph', fontsize=16, fontweight='bold')
            plt.xticks(x, para_ids)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{output_dir}/paragraph_breakdown.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_ratio_comparison(self, output_dir: str) -> None:
        """Plot comparison of different hallucination ratios."""
        stats = self.stats['overall_stats']
        
        # Prepare data
        ratio_types = ['H1 Ratio\n(Actions)', 'Overall Claim\nHallucination', 
                      'h2 Ratio\n(among claim hall.)', 'h3 Ratio\n(among claim hall.)', 
                      'h4 Ratio\n(among claim hall.)']
        ratios = [stats['overall_h1_ratio'], stats['overall_claim_hallucination_ratio'],
                 stats['overall_h2_ratio'], stats['overall_h3_ratio'], stats['overall_h4_ratio']]
        
        colors = ['red', 'darkred', 'orange', 'purple', 'brown']
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(ratio_types, ratios, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        plt.title('Hallucination Ratios Comparison', fontsize=16, fontweight='bold')
        plt.ylabel('Ratio', fontsize=12)
        plt.xlabel('Ratio Type', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for bar, ratio in zip(bars, ratios):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{ratio:.3f}\n({ratio*100:.1f}%)', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/ratio_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_detailed_report(self, output_file: str = "hallucination_report.json") -> None:
        """Save detailed statistics report to JSON."""
        report = {
            'summary': self.stats['overall_stats'],
            'iteration_details': self.stats['iteration_stats'],
            'report_details': self.stats['report_stats'],
            'breakdown_by_iteration': self.stats['detailed_breakdown']['iterations'],
            'breakdown_by_paragraph': self.stats['detailed_breakdown']['paragraphs']
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"📄 Detailed report saved to: {output_file}")
    
    def run_complete_analysis(self, output_dir: str = "../hallucination_analysis") -> None:
        """Run complete hallucination analysis."""
        print("🔍 Starting hallucination analysis...")
        
        # Run analysis
        self.analyze_iterations()
        self.analyze_report()
        self.calculate_ratios()
        
        # Print statistics
        self.print_statistics()
        
        # Create visualizations
        print("\n📊 Creating visualizations...")
        self.create_visualizations(output_dir)
        
        # Save detailed report
        report_file = f"{output_dir}/hallucination_report.json"
        self.save_detailed_report(report_file)
        
        print(f"\n✅ Complete analysis finished! Results saved to: {output_dir}/")


# Example usage
def analyze_hallucinations(results_file_path: str):
    """
    Main function to analyze hallucinations from evaluation results.
    
    Args:
        results_file_path: Path to the results JSON file from evaluate.py
    """
    analyzer = HallucinationStatistics(results_file_path)
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python hallucination_stats.py <results_file_path>")
        sys.exit(1)
    
    results_file = sys.argv[1]
    analyze_hallucinations(results_file)