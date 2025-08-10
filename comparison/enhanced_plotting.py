#!/usr/bin/env python3
"""
Enhanced plotting functions for comprehensive model and data ablation analysis.

This module provides advanced visualization capabilities for understanding:
1. Impact of different data sources on model performance
2. Model comparison across various metrics
3. Feature importance and ablation studies
4. Time series analysis and prediction quality
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for consistent, publication-ready plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class EnhancedPlottingManager:
    """Enhanced plotting manager for comprehensive model analysis."""
    
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Color palettes for different chart types
        import matplotlib.cm as cm
        self.model_colors = cm.get_cmap('tab10')(np.linspace(0, 1, 12))
        self.data_colors = {
            'price': '#1f77b4',      # Blue
            'technical': '#ff7f0e',   # Orange  
            'news': '#2ca02c',        # Green
            'economic': '#d62728',    # Red
            'events': '#9467bd',      # Purple
            'temporal': '#8c564b',    # Brown
            'baseline': '#e377c2',    # Pink
            'all_features': '#7f7f7f' # Gray
        }
        
    def create_data_impact_analysis(self, baseline_results: Dict[str, Any], save_name: str = "data_impact_analysis"):
        """
        Create comprehensive analysis showing impact of different data sources.
        Similar to the ablation study heatmaps shown in the user's images.
        """
        print("📊 Creating data impact analysis...")
        
        # Convert results to DataFrame
        summary_df = self._results_to_dataframe(baseline_results)
        if summary_df.empty:
            print("⚠️  No baseline results to plot")
            return
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Data Source Impact Analysis - Comprehensive Model Comparison', 
                     fontsize=16, fontweight='bold')
        
        # 1. R² Score Impact Heatmap (Top Left)
        self._plot_r2_impact_heatmap(summary_df, axes[0, 0])
        
        # 2. Feature Count vs Performance (Top Right)  
        self._plot_feature_count_vs_performance(summary_df, axes[0, 1])
        
        # 3. Data Source Contribution Analysis (Bottom Left)
        self._plot_data_source_contribution(summary_df, axes[1, 0])
        
        # 4. Model Ranking by Data Combination (Bottom Right)
        self._plot_model_ranking_by_data(summary_df, axes[1, 1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'{save_name}.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.save_dir, f'{save_name}.pdf'), bbox_inches='tight')
        plt.close()
        
        print(f"✅ Data impact analysis saved to {self.save_dir}/{save_name}.png")
    
    def create_ablation_study_heatmaps(self, baseline_results: Dict[str, Any], save_name: str = "ablation_heatmaps"):
        """
        Create detailed ablation study heatmaps like those shown in user's images.
        Shows performance across different data combinations for each model.
        """
        print("🔬 Creating ablation study heatmaps...")
        
        summary_df = self._results_to_dataframe(baseline_results)
        if summary_df.empty:
            return
        
        # Get unique models and data combinations
        models = summary_df['Model'].unique()
        data_combos = summary_df['Data_Combination'].unique()
        
        # Create three heatmaps for different metrics
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        fig.suptitle('Ablation Study Performance Heatmaps', fontsize=16, fontweight='bold')
        
        metrics = [
            ('Val_R2', 'R² Score (Returns Prediction)', 'RdYlGn', '.3f'),
            ('Directional_Accuracy', 'Directional Accuracy (%)', 'RdYlBu', '.1f'),
            ('Sharpe_Ratio_Proxy', 'Information Ratio', 'RdYlGn', '.2f')
        ]
        
        for idx, (metric_col, title, cmap, fmt) in enumerate(metrics):
            ax = axes[idx]
            
            # Create matrix for heatmap
            if metric_col == 'Directional_Accuracy':
                # Calculate directional accuracy from results
                matrix_data = self._calculate_directional_accuracy_matrix(baseline_results, models, data_combos)
                vmin, vmax = 40, 70
            elif metric_col == 'Sharpe_Ratio_Proxy':
                # Use R2 / volatility as proxy for information ratio
                matrix_data = self._calculate_information_ratio_matrix(summary_df, models, data_combos)
                vmin, vmax = -1, 2
            else:
                # Use R² values
                matrix_data = self._create_metric_matrix(summary_df, models, data_combos, 'Val_R2')
                vmin, vmax = 0, 1
            
            # Create heatmap
            im = ax.imshow(matrix_data, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
            
            # Add text annotations
            for i in range(len(models)):
                for j in range(len(data_combos)):
                    value = matrix_data[i, j]
                    if not np.isnan(value):
                        text_color = 'white' if value < (vmin + vmax) / 2 else 'black'
                        ax.text(j, i, f'{value:{fmt}}', ha="center", va="center", 
                               color=text_color, fontweight='bold', fontsize=8)
            
            # Customize axes
            ax.set_title(title, fontweight='bold')
            ax.set_xticks(range(len(data_combos)))
            ax.set_xticklabels([self._format_combo_name(combo) for combo in data_combos], 
                              rotation=45, ha='right')
            ax.set_yticks(range(len(models)))
            ax.set_yticklabels(models)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            if metric_col == 'Directional_Accuracy':
                cbar.set_label('Accuracy (%)')
            elif metric_col == 'Sharpe_Ratio_Proxy':
                cbar.set_label('Information Ratio')
            else:
                cbar.set_label('R² Score')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'{save_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Ablation study heatmaps saved to {self.save_dir}/{save_name}.png")
    
    def create_feature_importance_analysis(self, baseline_results: Dict[str, Any], save_name: str = "feature_importance"):
        """
        Create feature importance analysis showing contribution of each data source.
        """
        print("📈 Creating feature importance analysis...")
        
        summary_df = self._results_to_dataframe(baseline_results)
        if summary_df.empty:
            return
        
        # Calculate relative improvements from baseline
        baseline_perf = self._get_baseline_performance(summary_df)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Feature Group Importance Analysis', fontsize=16, fontweight='bold')
        
        # 1. R² Improvement by Adding Each Data Source
        self._plot_r2_improvement_bars(summary_df, baseline_perf, axes[0, 0])
        
        # 2. Model Performance Consistency Across Data Sources
        self._plot_performance_consistency(summary_df, axes[0, 1])
        
        # 3. Data Source Ranking by Average Improvement
        self._plot_data_source_ranking(summary_df, baseline_perf, axes[1, 0])
        
        # 4. Feature Count vs Performance Scatter
        self._plot_feature_count_scatter(summary_df, axes[1, 1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'{save_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Feature importance analysis saved to {self.save_dir}/{save_name}.png")
    
    def create_comprehensive_model_comparison(self, baseline_results: Dict[str, Any], 
                                           neural_results: Optional[Dict[str, Any]] = None,
                                           save_name: str = "comprehensive_comparison"):
        """
        Create comprehensive comparison including both baseline and neural network models.
        """
        print("🎯 Creating comprehensive model comparison...")
        
        # Combine all results
        all_results = baseline_results.copy()
        if neural_results:
            all_results.update(neural_results)
        
        summary_df = self._results_to_dataframe(all_results)
        if summary_df.empty:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        fig.suptitle('Comprehensive Model and Data Comparison', fontsize=18, fontweight='bold')
        
        # 1. Overall Performance Ranking
        self._plot_overall_ranking(summary_df, axes[0, 0])
        
        # 2. Model Type Comparison
        self._plot_model_type_comparison(summary_df, axes[0, 1])
        
        # 3. Data Combination Effectiveness
        self._plot_data_effectiveness(summary_df, axes[0, 2])
        
        # 4. Performance vs Complexity
        self._plot_performance_vs_complexity(summary_df, axes[1, 0])
        
        # 5. Top Model Performance Distribution
        self._plot_top_model_distribution(summary_df, axes[1, 1])
        
        # 6. Metric Correlation Analysis
        self._plot_metric_correlations(summary_df, axes[1, 2])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'{save_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Comprehensive comparison saved to {self.save_dir}/{save_name}.png")
    
    def save_detailed_results_tables(self, baseline_results: Dict[str, Any], 
                                   neural_results: Optional[Dict[str, Any]] = None):
        """
        Save detailed numerical results as CSV files for further analysis.
        """
        print("💾 Saving detailed results tables...")
        
        # Combine all results
        all_results = baseline_results.copy()
        if neural_results:
            all_results.update(neural_results)
        
        # Create comprehensive summary
        summary_df = self._results_to_dataframe(all_results)
        summary_df.to_csv(os.path.join(self.save_dir, 'comprehensive_results.csv'), index=False)
        
        # Create model-specific summaries
        for model in summary_df['Model'].unique():
            model_df = summary_df[summary_df['Model'] == model]
            filename = f"{model.lower().replace(' ', '_')}_results.csv"
            model_df.to_csv(os.path.join(self.save_dir, filename), index=False)
        
        # Create data combination summaries
        for combo in summary_df['Data_Combination'].unique():
            combo_df = summary_df[summary_df['Data_Combination'] == combo]
            filename = f"{combo}_results.csv"
            combo_df.to_csv(os.path.join(self.save_dir, filename), index=False)
        
        # Create best models summary
        best_models = summary_df.nlargest(20, 'Val_R2')
        best_models.to_csv(os.path.join(self.save_dir, 'best_models_summary.csv'), index=False)
        
        # Create data source impact summary
        impact_summary = self._create_data_impact_summary(summary_df)
        impact_summary.to_csv(os.path.join(self.save_dir, 'data_source_impact.csv'), index=False)
        
        print(f"✅ Detailed results saved to {self.save_dir}/")
    
    # Helper methods for plotting
    def _results_to_dataframe(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Convert results dictionary to DataFrame for easier plotting."""
        data = []
        for key, result in results.items():
            if isinstance(result, dict) and 'model' in result:
                data.append({
                    'Experiment': key,
                    'Model': result['model'],
                    'Data_Combination': result.get('combination', 'unknown'),
                    'Description': result.get('description', ''),
                    'Features_Used': result.get('features_used', 0),
                    'Train_MSE': result.get('train_mse', np.nan),
                    'Train_MAE': result.get('train_mae', np.nan),
                    'Train_R2': result.get('train_r2', np.nan),
                    'Val_MSE': result.get('val_mse', np.nan),
                    'Val_MAE': result.get('val_mae', np.nan),
                    'Val_R2': result.get('val_r2', np.nan)
                })
        
        return pd.DataFrame(data)
    
    def _plot_r2_impact_heatmap(self, summary_df: pd.DataFrame, ax):
        """Plot R² impact heatmap."""
        # Create pivot table for models vs data combinations
        pivot_data = summary_df.pivot_table(
            values='Val_R2', 
            index='Model', 
            columns='Data_Combination', 
            aggfunc='mean'
        )
        
        # Create heatmap
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                   ax=ax, cbar_kws={'label': 'R² Score'}, vmin=0, vmax=1)
        ax.set_title('R² Score by Model and Data Combination', fontweight='bold')
        ax.set_xlabel('Data Combination')
        ax.set_ylabel('Model')
    
    def _plot_feature_count_vs_performance(self, summary_df: pd.DataFrame, ax):
        """Plot feature count vs performance scatter."""
        for model in summary_df['Model'].unique():
            model_data = summary_df[summary_df['Model'] == model]
            ax.scatter(model_data['Features_Used'], model_data['Val_R2'], 
                      label=model, alpha=0.7, s=50)
        
        ax.set_xlabel('Number of Features Used')
        ax.set_ylabel('Validation R² Score')
        ax.set_title('Feature Count vs Performance', fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def _plot_data_source_contribution(self, summary_df: pd.DataFrame, ax):
        """Plot data source contribution analysis."""
        # Calculate average performance by data combination
        combo_performance = summary_df.groupby('Data_Combination')['Val_R2'].mean().sort_values(ascending=True)
        
        # Color bars by data source type
        colors = [self.data_colors.get(self._extract_data_type(combo), '#gray') 
                 for combo in combo_performance.index]
        
        bars = ax.barh(range(len(combo_performance)), combo_performance.values, color=colors)
        ax.set_yticks(range(len(combo_performance)))
        ax.set_yticklabels([self._format_combo_name(combo) for combo in combo_performance.index])
        ax.set_xlabel('Average R² Score')
        ax.set_title('Data Source Contribution to Performance', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', ha='left', va='center', fontweight='bold')
    
    def _plot_model_ranking_by_data(self, summary_df: pd.DataFrame, ax):
        """Plot model ranking by data combination."""
        # Get top 3 models for each data combination
        top_models = []
        for combo in summary_df['Data_Combination'].unique():
            combo_data = summary_df[summary_df['Data_Combination'] == combo].nlargest(3, 'Val_R2')
            for idx, (_, row) in enumerate(combo_data.iterrows()):
                top_models.append({
                    'Data_Combination': combo,
                    'Model': row['Model'],
                    'Rank': idx + 1,
                    'R2': row['Val_R2']
                })
        
        top_df = pd.DataFrame(top_models)
        
        # Create stacked bar chart
        pivot_ranks = top_df.pivot_table(values='R2', index='Data_Combination', 
                                        columns='Rank', aggfunc='first', fill_value=0)
        
        pivot_ranks.plot(kind='bar', stacked=True, ax=ax, 
                        color=['gold', 'silver', '#CD7F32'])  # Gold, Silver, Bronze
        ax.set_title('Top 3 Models by Data Combination', fontweight='bold')
        ax.set_xlabel('Data Combination')
        ax.set_ylabel('Cumulative R² Score')
        ax.legend(title='Rank', labels=['1st', '2nd', '3rd'])
        ax.tick_params(axis='x', rotation=45)
    
    def _extract_data_type(self, combo_name: str) -> str:
        """Extract the primary data type from combination name."""
        for data_type in self.data_colors.keys():
            if data_type in combo_name:
                return data_type
        return 'baseline'
    
    def _format_combo_name(self, combo_name: str) -> str:
        """Format combination name for display."""
        return combo_name.replace('_', ' ').title()
    
    def _calculate_directional_accuracy_matrix(self, results, models, data_combos):
        """Calculate directional accuracy matrix for heatmap."""
        matrix = np.full((len(models), len(data_combos)), np.nan)
        
        for i, model in enumerate(models):
            for j, combo in enumerate(data_combos):
                key = f"{model}_{combo}"
                if key in results and 'predictions' in results[key] and 'actuals' in results[key]:
                    preds = results[key]['predictions']
                    actuals = results[key]['actuals']
                    if len(preds) > 1 and len(actuals) > 1:
                        # Calculate directional accuracy
                        pred_direction = np.diff(preds) > 0
                        actual_direction = np.diff(actuals) > 0
                        accuracy = np.mean(pred_direction == actual_direction) * 100
                        matrix[i, j] = accuracy
        
        return matrix
    
    def _calculate_information_ratio_matrix(self, summary_df, models, data_combos):
        """Calculate information ratio proxy matrix."""
        matrix = np.full((len(models), len(data_combos)), np.nan)
        
        for i, model in enumerate(models):
            for j, combo in enumerate(data_combos):
                model_data = summary_df[(summary_df['Model'] == model) & 
                                      (summary_df['Data_Combination'] == combo)]
                if not model_data.empty:
                    r2 = model_data['Val_R2'].iloc[0]
                    mse = model_data['Val_MSE'].iloc[0]
                    # Simple proxy: R2 / sqrt(MSE)
                    if not np.isnan(r2) and not np.isnan(mse) and mse > 0:
                        info_ratio = r2 / np.sqrt(mse)
                        matrix[i, j] = info_ratio
        
        return matrix
    
    def _create_metric_matrix(self, summary_df, models, data_combos, metric):
        """Create matrix for any metric."""
        matrix = np.full((len(models), len(data_combos)), np.nan)
        
        for i, model in enumerate(models):
            for j, combo in enumerate(data_combos):
                model_data = summary_df[(summary_df['Model'] == model) & 
                                      (summary_df['Data_Combination'] == combo)]
                if not model_data.empty:
                    value = model_data[metric].iloc[0]
                    if not np.isnan(value):
                        matrix[i, j] = value
        
        return matrix
    
    def _get_baseline_performance(self, summary_df):
        """Get baseline performance for comparison."""
        baseline_data = summary_df[summary_df['Data_Combination'] == 'baseline']
        if baseline_data.empty:
            return summary_df.groupby('Model')['Val_R2'].mean()
        return baseline_data.groupby('Model')['Val_R2'].mean()
    
    def _plot_r2_improvement_bars(self, summary_df, baseline_perf, ax):
        """Plot R² improvement bars."""
        improvements = {}
        for combo in summary_df['Data_Combination'].unique():
            if combo != 'baseline':
                combo_data = summary_df[summary_df['Data_Combination'] == combo]
                avg_perf = combo_data.groupby('Model')['Val_R2'].mean()
                improvements[combo] = (avg_perf - baseline_perf).mean()
        
        if improvements:
            combos = list(improvements.keys())
            values = list(improvements.values())
            colors = [self.data_colors.get(self._extract_data_type(combo), 'gray') for combo in combos]
            
            bars = ax.bar(combos, values, color=colors)
            ax.set_title('R² Improvement from Baseline', fontweight='bold')
            ax.set_ylabel('Average R² Improvement')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
            
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    def _plot_performance_consistency(self, summary_df, ax):
        """Plot performance consistency across models."""
        consistency_data = []
        for combo in summary_df['Data_Combination'].unique():
            combo_data = summary_df[summary_df['Data_Combination'] == combo]
            std_dev = combo_data['Val_R2'].std()
            mean_perf = combo_data['Val_R2'].mean()
            consistency_data.append({'Combination': combo, 'Std_Dev': std_dev, 'Mean': mean_perf})
        
        cons_df = pd.DataFrame(consistency_data)
        ax.scatter(cons_df['Mean'], cons_df['Std_Dev'])
        
        for _, row in cons_df.iterrows():
            ax.annotate(self._format_combo_name(row['Combination']), 
                       (row['Mean'], row['Std_Dev']), fontsize=8)
        
        ax.set_xlabel('Mean R² Performance')
        ax.set_ylabel('R² Standard Deviation')
        ax.set_title('Performance Consistency Analysis', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_data_source_ranking(self, summary_df, baseline_perf, ax):
        """Plot data source ranking."""
        improvements = []
        for combo in summary_df['Data_Combination'].unique():
            if combo != 'baseline':
                combo_data = summary_df[summary_df['Data_Combination'] == combo]
                avg_perf = combo_data['Val_R2'].mean()
                baseline_avg = baseline_perf.mean() if not baseline_perf.empty else 0
                improvement = avg_perf - baseline_avg
                improvements.append({'Source': combo, 'Improvement': improvement})
        
        if improvements:
            imp_df = pd.DataFrame(improvements).sort_values('Improvement', ascending=True)
            colors = [self.data_colors.get(self._extract_data_type(source), 'gray') 
                     for source in imp_df['Source']]
            
            bars = ax.barh(range(len(imp_df)), imp_df['Improvement'], color=colors)
            ax.set_yticks(range(len(imp_df)))
            ax.set_yticklabels([self._format_combo_name(s) for s in imp_df['Source']])
            ax.set_xlabel('R² Improvement over Baseline')
            ax.set_title('Data Source Ranking', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
    
    def _plot_feature_count_scatter(self, summary_df, ax):
        """Plot feature count vs performance scatter."""
        ax.scatter(summary_df['Features_Used'], summary_df['Val_R2'], alpha=0.6)
        ax.set_xlabel('Number of Features')
        ax.set_ylabel('R² Score')
        ax.set_title('Features vs Performance', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_overall_ranking(self, summary_df, ax):
        """Plot overall model ranking."""
        top_models = summary_df.nlargest(15, 'Val_R2')
        ax.barh(range(len(top_models)), top_models['Val_R2'])
        ax.set_yticks(range(len(top_models)))
        ax.set_yticklabels([f"{row['Model']} ({self._format_combo_name(row['Data_Combination'])})" 
                           for _, row in top_models.iterrows()], fontsize=8)
        ax.set_xlabel('R² Score')
        ax.set_title('Top 15 Model Configurations', fontweight='bold')
    
    def _plot_model_type_comparison(self, summary_df, ax):
        """Plot model type comparison."""
        model_avg = summary_df.groupby('Model')['Val_R2'].agg(['mean', 'std']).sort_values('mean', ascending=False)
        
        ax.bar(range(len(model_avg)), model_avg['mean'], yerr=model_avg['std'], capsize=5)
        ax.set_xticks(range(len(model_avg)))
        ax.set_xticklabels(model_avg.index, rotation=45, ha='right')
        ax.set_ylabel('R² Score')
        ax.set_title('Model Type Comparison', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_data_effectiveness(self, summary_df, ax):
        """Plot data combination effectiveness."""
        combo_avg = summary_df.groupby('Data_Combination')['Val_R2'].mean().sort_values(ascending=False)
        colors = [self.data_colors.get(self._extract_data_type(combo), 'gray') for combo in combo_avg.index]
        
        ax.bar(range(len(combo_avg)), combo_avg.values, color=colors)
        ax.set_xticks(range(len(combo_avg)))
        ax.set_xticklabels([self._format_combo_name(combo) for combo in combo_avg.index], 
                          rotation=45, ha='right')
        ax.set_ylabel('Average R² Score')
        ax.set_title('Data Combination Effectiveness', fontweight='bold')
    
    def _plot_performance_vs_complexity(self, summary_df, ax):
        """Plot performance vs complexity."""
        ax.scatter(summary_df['Features_Used'], summary_df['Val_R2'], 
                  c=summary_df['Val_R2'], cmap='viridis', alpha=0.6)
        ax.set_xlabel('Model Complexity (Features)')
        ax.set_ylabel('Performance (R²)')
        ax.set_title('Performance vs Complexity', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_top_model_distribution(self, summary_df, ax):
        """Plot top model distribution."""
        top_models = summary_df.nlargest(20, 'Val_R2')
        ax.hist(top_models['Val_R2'], bins=10, alpha=0.7, edgecolor='black')
        ax.set_xlabel('R² Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Top 20 Models R² Distribution', fontweight='bold')
        ax.axvline(top_models['Val_R2'].mean(), color='red', linestyle='--', 
                  label=f'Mean: {top_models["Val_R2"].mean():.3f}')
        ax.legend()
    
    def _plot_metric_correlations(self, summary_df, ax):
        """Plot metric correlations."""
        metrics = ['Val_R2', 'Val_MSE', 'Val_MAE', 'Features_Used']
        available_metrics = [m for m in metrics if m in summary_df.columns]
        
        if len(available_metrics) > 1:
            corr_matrix = summary_df[available_metrics].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
            ax.set_title('Metric Correlations', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Insufficient data\nfor correlation analysis', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _create_data_impact_summary(self, summary_df):
        """Create data impact summary table."""
        impact_data = []
        baseline_avg = summary_df[summary_df['Data_Combination'] == 'baseline']['Val_R2'].mean()
        
        for combo in summary_df['Data_Combination'].unique():
            if combo != 'baseline':
                combo_data = summary_df[summary_df['Data_Combination'] == combo]
                avg_perf = combo_data['Val_R2'].mean()
                improvement = avg_perf - baseline_avg if not np.isnan(baseline_avg) else avg_perf
                
                impact_data.append({
                    'Data_Source': combo,
                    'Average_R2': avg_perf,
                    'Improvement_over_Baseline': improvement,
                    'Best_Model': combo_data.loc[combo_data['Val_R2'].idxmax(), 'Model'],
                    'Best_R2': combo_data['Val_R2'].max()
                })
        
        return pd.DataFrame(impact_data)
