import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import seaborn as sns
import math


class PolicyResultsVisualizer:
    """Visualizer for multi-objective policy evaluation results."""
    
    def __init__(self, output_dir: str = "visualizations"):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save generated plots
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def load_trajectory_data(self, result_folder: str) -> Dict[str, Any]:
        """
        Load trajectory data from a result folder.
        
        Args:
            result_folder: Path to folder containing evaluation_trajectories.json
            
        Returns:
            Dictionary containing loaded trajectory data
        """
        trajectory_file = os.path.join(result_folder, "evaluation_trajectories.json")
        
        if not os.path.exists(trajectory_file):
            raise FileNotFoundError(f"Trajectory file not found: {trajectory_file}")
        
        with open(trajectory_file, 'r') as f:
            trajectories = json.load(f)
        
        return trajectories
    
    def load_evaluation_results(self, result_folder: str) -> Dict[str, Any]:
        """
        Extract policy selections from evaluation trajectories.
        
        Args:
            result_folder: Path to folder containing evaluation_trajectories.json
            
        Returns:
            Dictionary containing policy selections extracted from trajectories
        """
        trajectory_file = os.path.join(result_folder, "evaluation_trajectories.json")
        if not os.path.exists(trajectory_file):
            raise FileNotFoundError(f"evaluation_trajectories.json not found in {result_folder}")
        
        with open(trajectory_file, 'r') as f:
            trajectories = json.load(f)
        
        # Extract policy selections from trajectories
        policy_selections = []
        for episode_trajectory in trajectories:
            episode_selections = []
            for step in episode_trajectory:
                if "orchestrator_info" in step and "selected_strategy" in step["orchestrator_info"]:
                    episode_selections.append(step["orchestrator_info"]["selected_strategy"])
            policy_selections.append(episode_selections)
        
        return {"policy_selections": policy_selections}
    
    def calculate_pareto_metrics(self, trajectories: List[List[Dict]], result_folder: str = None) -> Tuple[float, float]:
        """
        Calculate average performance and gap rewards across all episodes.
        
        Args:
            trajectories: List of episode trajectories
            objectives: List of objectives from config (e.g., ["performance", "gap"])
            result_folder: Path to result folder containing policy_level_profile.json
            
        Returns:
            Tuple of (avg_first_objective_reward, avg_second_objective_reward)
        """
        # Try to load from policy_level_profile.json first
        if result_folder:
            profile_data = self._load_policy_level_profile(result_folder)
            if profile_data and "objective_avg_rewards" in profile_data:
                objective_rewards = self._extract_objective_rewards_from_profile(profile_data)
                
                # first_reward = objective_rewards['performance'] if 'performance' in objective_rewards else 0
                # second_reward = objective_rewards['gap'] if 'gap' in objective_rewards else 0
                
                # return first_reward, second_reward
                # objectives = ["performance", "gap"]
        
                return self._calculate_pareto_metrics_from_trajectories(trajectories, objective_rewards.keys()), objective_rewards
    
    def _load_policy_level_profile(self, result_folder: str) -> Dict[str, Any]:
        """
        Load policy level profile from JSON file.
        
        Args:
            result_folder: Path to result folder
            
        Returns:
            Policy profile data or None if not found
        """
        try:
            # profile_file = os.path.join(result_folder, "policy_level_profile.json")
            # if os.path.exists(profile_file):
            #     with open(profile_file, 'r') as f:
            #         return json.load(f)
            if os.path.exists(os.path.join(result_folder, "policy_level_profile.json")):
                with open(os.path.join(result_folder, "policy_level_profile.json"), 'r') as f:
                    return json.load(f)
            elif os.path.exists(os.path.join(result_folder, "orchestrator_profile.json")):
                with open(os.path.join(result_folder, "orchestrator_profile.json"), 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load policy_level_profile.json from {result_folder}: {e}")
        return None
    
    def _extract_objective_rewards_from_profile(self, profile_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract objective rewards from policy level profile.
        
        Args:
            profile_data: Loaded policy profile data
            objectives: List of objectives to extract (if None, use profile's objectives)
            
        Returns:
            Dictionary mapping objective names to their reward values
        """
        objectives = profile_data["objectives"]
        
        objective_avg_rewards = profile_data["objective_avg_rewards"]
        
        # Create dictionary mapping objective names to reward values
        objective_rewards = {}
        for objective in objectives:
            if objective in objective_avg_rewards:
                objective_rewards[objective] = objective_avg_rewards[objective]
            else:
                raise ValueError(f"Objective {objective} not found in profile data")
        
        return objective_rewards
    
    def _calculate_pareto_metrics_from_trajectories(self, trajectories: List[List[Dict]], objectives: List[str] = None) -> Tuple[Tuple[float, float, float], Dict[str, float]]:
        """
        Calculate Pareto metrics from trajectory data (fallback method).
        
        Args:
            trajectories: List of episode trajectories
            objectives: List of objectives from profile data
            
        Returns:
            Tuple of ((avg_performance, avg_gap, avg_aptitude), objective_rewards_dict)
        """
        # Always compute the three canonical objectives for plotting compatibility
        canonical_objs = ["performance", "gap", "aptitude"]

        # Initialize tracking for canonical objectives
        episode_objective_rewards = {obj: [] for obj in canonical_objs}

        for episode_trajectory in trajectories:
            # Initialize episode rewards for each objective
            episode_rewards = {obj: 0.0 for obj in canonical_objs}

            for step in episode_trajectory:
                # Some trajectory formats store rewards under step['reward']['reward_dict']
                # while other formats might store a flat 'reward_dict' key. Handle both.
                reward_dict = None
                if "reward" in step and isinstance(step["reward"], dict) and "reward_dict" in step["reward"]:
                    reward_dict = step["reward"]["reward_dict"]
                elif "reward_dict" in step:
                    reward_dict = step["reward_dict"]

                if reward_dict and isinstance(reward_dict, dict):
                    # Sum rewards for each canonical objective (missing keys treated as 0)
                    for obj in canonical_objs:
                        episode_rewards[obj] += float(reward_dict.get(obj, 0.0))

            # Store episode rewards for each canonical objective
            for obj in canonical_objs:
                episode_objective_rewards[obj].append(episode_rewards[obj])

        # Calculate averages across all episodes for each canonical objective
        objective_avg_rewards = {}
        for obj in canonical_objs:
            # If there are no episodes, use NaN to indicate missing data
            if len(episode_objective_rewards[obj]) == 0:
                objective_avg_rewards[obj] = float('nan')
            else:
                objective_avg_rewards[obj] = float(np.mean(episode_objective_rewards[obj]))

        # Return (performance, gap, aptitude) tuple
        return (objective_avg_rewards["performance"], objective_avg_rewards["gap"], objective_avg_rewards["aptitude"])
    
    def _infer_objectives_from_trajectories(self, trajectories: List[List[Dict]], result_folder: str = None) -> List[str]:
        """
        Try to infer objectives from policy_level_profile.json or trajectory data.
        
        Args:
            trajectories: List of episode trajectories
            result_folder: Path to result folder containing policy_level_profile.json
            
        Returns:
            List of objectives if found, empty list otherwise
        """
        # First try to load from policy_level_profile.json
        if result_folder:
            profile_data = self._load_policy_level_profile(result_folder)
            if profile_data and "objectives" in profile_data:
                return profile_data["objectives"]
        
        # Fallback: check if objectives are in the first step's metadata (unlikely)
        if not trajectories or not trajectories[0]:
            return []
        
        first_step = trajectories[0][0]
        if "objectives" in first_step:
            return first_step["objectives"]
        
        # If still no objectives found, return empty list
        return []
    
    def calculate_mastery_progression(self, trajectories: List[List[Dict]], max_steps: int = 30) -> Tuple[List[float], float]:
        """
        Calculate average mastery progression across all episodes.
        
        Args:
            trajectories: List of episode trajectories
            max_steps: Maximum number of steps to consider
            
        Returns:
            Tuple of (avg_mastery_by_step, avg_stop_step)
        """
        # Initialize mastery tracking
        step_mastery_data = [[] for _ in range(max_steps)]
        episode_lengths = []
        
        for episode_trajectory in trajectories[:1]:
            episode_lengths.append(len(episode_trajectory))
            
            for step_idx, step in enumerate(episode_trajectory):
                if step_idx >= max_steps:
                    break
                    
                if ("state" in step and step["state"] and 
                    "mastery" in step["state"] and step["state"]["mastery"]):
                    
                    # Calculate average mastery across all skills for this step
                    mastery_values = list(step["state"]["mastery"].values())
                    if mastery_values:
                        avg_mastery = np.mean(mastery_values)
                        step_mastery_data[step_idx].append(avg_mastery)
        
        # Calculate average mastery for each step
        avg_mastery_by_step = []
        for step_data in step_mastery_data:
            if step_data:
                avg_mastery_by_step.append(np.mean(step_data))
            else:
                avg_mastery_by_step.append(np.nan)
        
        # Calculate average stop step (add 1 to convert from 0-based to 1-based indexing)
        avg_stop_step = np.mean(episode_lengths) if episode_lengths else 0.0
        
        return avg_mastery_by_step, avg_stop_step
    
    def find_pareto_front(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Find Pareto optimal points (non-dominated points).
        
        Args:
            points: List of (x, y) coordinate tuples
            
        Returns:
            List of Pareto optimal points
        """
        pareto_front = []
        
        for i, (x1, y1) in enumerate(points):
            dominated = False
            
            for j, (x2, y2) in enumerate(points):
                if i != j:
                    # Check if point i is dominated by point j
                    if x2 >= x1 and y2 >= y1 and (x2 > x1 or y2 > y1):
                        dominated = True
                        break
            
            if not dominated:
                pareto_front.append((x1, y1))
        
        return pareto_front
    

    
    def sort_pareto_front(self, pareto_front: List[Tuple[float, float]], sort_by: str = 'x') -> List[Tuple[float, float]]:
        """
        Sort Pareto front points for connecting.
        
        Args:
            pareto_front: List of Pareto optimal points
            sort_by: 'x' to sort by x-coordinate, 'y' to sort by y-coordinate
            
        Returns:
            Sorted list of Pareto front points
        """
        if sort_by == 'x':
            return sorted(pareto_front, key=lambda p: p[0])
        else:  # sort_by == 'y'
            return sorted(pareto_front, key=lambda p: p[1])
    
    def find_3d_pareto_front(self, points: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """
        Find 3D Pareto optimal points (non-dominated points).
        
        Args:
            points: List of (x, y, z) coordinate tuples
            
        Returns:
            List of 3D Pareto optimal points
        """
        pareto_front = []
        
        for i, (x1, y1, z1) in enumerate(points):
            dominated = False
            
            for j, (x2, y2, z2) in enumerate(points):
                if i != j:
                    # Check if point i is dominated by point j
                    # A point is dominated if another point is better or equal in all dimensions and strictly better in at least one
                    if x2 >= x1 and y2 >= y1 and z2 >= z1 and (x2 > x1 or y2 > y1 or z2 > z1):
                        dominated = True
                        break
            
            if not dominated:
                pareto_front.append((x1, y1, z1))
        
        return pareto_front
    
    def visualize_policies(self, policy_configs: Dict[str, str], max_steps: int = 50, config_name: str = "Default", correlation_scatter: bool = False, optimized_obj: str = "performance", flow_zone: bool = False):
        """
        Main function to visualize multiple policies.
        
        Args:
            policy_configs: Dictionary mapping policy names to result folder paths
            max_steps: Maximum number of steps for mastery progression
            config_name: Name of this configuration set for plot titles
            correlation_scatter: If True, generate correlation scatter plot
        """
        print(f"Loading policy data for configuration: {config_name}")
        policy_data = {}
        
        for policy_name, result_folder in policy_configs.items():
            print(f"Processing {policy_name}...")
            
            try:
                # Load trajectory data
                trajectories = self.load_trajectory_data(result_folder)
                
                # Calculate Pareto metrics
                pareto_metrics, objective_rewards = self.calculate_pareto_metrics(trajectories, result_folder=result_folder)
                
                avg_performance, avg_gap, avg_aptitude = pareto_metrics
                
                # Calculate mastery progression
                mastery_progression, avg_stop_step = self.calculate_mastery_progression(trajectories, max_steps)
                
                policy_data[policy_name] = {
                    'avg_performance': avg_performance,
                    'avg_gap': avg_gap,
                    'avg_aptitude': avg_aptitude,
                    'mastery_progression': mastery_progression,
                    'avg_stop_step': avg_stop_step,
                    'trajectories': trajectories  # Include trajectories for correlation plot
                }
                
                print(f"  Performance: {avg_performance:.3f}, Gap: {avg_gap:.3f}, Aptitude: {avg_aptitude:.3f}, Avg Stop: {avg_stop_step:.1f}")
                
            except Exception as e:
                print(f"Error processing {policy_name}: {e}")
                continue
        
        if not policy_data:
            print(f"No valid policy data found for {config_name}!")
            return
        
        print(f"\nGenerating visualizations for {len(policy_data)} policies in {config_name}...")
        
        # Generate plots with config-specific names
        safe_config_name = config_name.replace(" ", "_").replace("/", "_")
        
        if correlation_scatter:
            # Only generate correlation scatter plot
            correlation_save_path = os.path.join(self.output_dir, f"correlation_scatter_{safe_config_name}.png")
            self.plot_correlation_scatter(policy_data, correlation_save_path, config_name, optimized_obj)
        elif flow_zone:
            # Only generate flow zone plot
            flow_zone_save_path = os.path.join(self.output_dir, f"flow_zone_{safe_config_name}.png")
            self.plot_flow_zone(policy_data, flow_zone_save_path, config_name)
        else:
            # Generate all other plots
            pareto_save_path = os.path.join(self.output_dir, f"pareto_optimality_{safe_config_name}.png")
            mastery_save_path = os.path.join(self.output_dir, f"mastery_progression_{safe_config_name}.png")
            
            self.plot_pareto_optimality(policy_data, pareto_save_path, config_name)
            self.plot_radar_pareto_optimality(policy_data, pareto_save_path, config_name, max_steps)
            self.plot_mastery_progression(policy_data, max_steps, mastery_save_path, config_name)
        
        print(f"Visualization complete for {config_name}!")

    def plot_pareto_optimality(self, policy_data: Dict[str, Dict[str, Any]], save_path: str = None, config_name: str = "Default"):
        """
        Create Pareto optimality plot (2D or 3D based on available objectives).
        
        Args:
            policy_data: Dictionary mapping policy names to their data
            save_path: Path to save the plot (optional)
            config_name: Name of this configuration set for plot title
        """
        # Set elegant background style
        plt.style.use('default')
        plt.rcParams['figure.facecolor'] = '#f8f9fa'
        plt.rcParams['axes.facecolor'] = '#ffffff'
        plt.rcParams['savefig.facecolor'] = '#f8f9fa'
        
        self._plot_3d_pareto_optimality(policy_data, save_path, config_name)
        
    def _plot_3d_pareto_optimality(self, policy_data: Dict[str, Dict[str, Any]], save_path: str = None, config_name: str = "Default"):
        """Create 3D Pareto optimality plot (Performance vs Gap vs Aptitude)."""
        from mpl_toolkits.mplot3d import Axes3D
        
        # Create figure with elegant background
        fig = plt.figure(figsize=(14, 10), facecolor='#f8f9fa')
        ax = fig.add_subplot(111, projection='3d')
        
        # Set clean background colors
        ax.set_facecolor('white')
        # Use modern matplotlib 3D axis properties
        try:
            # Modern matplotlib 3D pane color API
            ax.xaxis.set_pane_color((0.95, 0.95, 0.95, 0.8))
            ax.yaxis.set_pane_color((0.95, 0.95, 0.95, 0.8))
            ax.zaxis.set_pane_color((0.95, 0.95, 0.95, 0.8))
        except AttributeError:
            # Legacy matplotlib 3D pane color API
            try:
                ax.w_xaxis.set_pane_color((0.95, 0.95, 0.95, 0.8))
                ax.w_yaxis.set_pane_color((0.95, 0.95, 0.95, 0.8))
                ax.w_zaxis.set_pane_color((0.95, 0.95, 0.95, 0.8))
            except AttributeError:
                pass  # Skip if neither method works
        
        pub_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        markers = ['o', 's', '^', 'D', 'p', 'h', '*', 'x']
        
        performance_values = []
        gap_values = []
        aptitude_values = []
        labels = []
        
        all_policies = list(policy_data.items())
        for i, (policy_name, data) in enumerate(all_policies):
            performance = data['avg_performance']
            gap = data['avg_gap']
            aptitude = data['avg_aptitude']
            
            performance_values.append(performance)
            gap_values.append(gap)
            aptitude_values.append(aptitude)
            labels.append(policy_name)
            
            color = pub_colors[i % len(pub_colors)]
            
            ax.scatter(performance, gap, aptitude, 
                      c=color, 
                      marker=markers[i % len(markers)],
                      s=200,
                      label=policy_name,
                      edgecolors='#2c3e50',
                      linewidth=1.5,
                      alpha=0.8)
        
        # Find and plot 3D Pareto front
        if len(performance_values) > 1:
            points = list(zip(performance_values, gap_values, aptitude_values))
            pareto_front_3d = self.find_3d_pareto_front(points)
            
            if len(pareto_front_3d) > 1:
                # Plot Pareto front points with elegant styling
                for x, y, z in pareto_front_3d:
                    ax.scatter(x, y, z, c='#e74c3c', s=80, marker='*', linewidth=2, alpha=0.9)
        
        # Elegant axis styling
        ax.set_xlabel('Average Performance Reward', fontsize=14, fontweight='bold', color='#2c3e50')
        ax.set_ylabel('Average Gap Reward', fontsize=14, fontweight='bold', color='#2c3e50')
        ax.set_zlabel('Average Aptitude Reward', fontsize=14, fontweight='bold', color='#2c3e50')
        ax.set_title(f'3D Pareto Optimality: Performance vs Gap vs Aptitude - {config_name}', 
                    fontsize=16, fontweight='bold', color='#2c3e50', pad=20)
        
        # Elegant legend styling
        ax.legend(loc='upper right', fontsize=10, frameon=True, 
                 facecolor='#ffffff', edgecolor='#bdc3c7', fancybox=True, shadow=True)
        
        # Set axes limits
        ax.set_xlim(left=-1)
        ax.set_ylim(bottom=-1)
        ax.set_zlim(bottom=-1)
        
        # Add subtle grid
        ax.grid(True, alpha=0.3, color='#bdc3c7')
        
        plt.tight_layout()
        
        if save_path:
            # Modify save path for 3D plot
            save_path_3d = save_path.replace('.png', '_3d.png')
            plt.savefig(save_path_3d, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"3D Pareto optimality plot saved to: {save_path_3d}")
        
        plt.show()
    
    def plot_radar_pareto_optimality(self, policy_data: Dict[str, Dict[str, Any]], save_path: str = None, config_name: str = "Default", max_steps: int = 30):
        """
        Create radar plot for Pareto optimality visualization.
        
        Args:
            policy_data: Dictionary mapping policy names to their data
            save_path: Path to save the plot (optional)
            config_name: Name of this configuration set for plot title
        """

        plt.rcParams.update({
            'font.family': 'DejaVu Sans',
            'font.size': 10,
            'axes.titlesize': 11,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 16,
            'figure.titlesize': 12,
            'text.usetex': False, 
        })
        
        # Define objectives for radar plot
        objectives = ['Perf', 'Gap', 'Apt']
        num_vars = len(objectives)
        
        # Compute angle for each axis
        angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
        angles += angles[:1]  # Complete the circle
        
        fig = plt.figure(figsize=(8, 5))
        ax = plt.subplot2grid((1, 5), (0, 0), colspan=5, projection='polar')
        
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        pub_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
        markers = ['o', 's', '^', 'D', 'p', 'h', '*', 'x']
        
        all_policies = list(policy_data.items())
        for i, (policy_name, data) in enumerate(all_policies):
            performance_rescaled = 0.0
            gap_rescaled = 0.0
            aptitude_rescaled = 0.0
            
            if 'trajectories' in data and data['trajectories']:
                episode_performance_scores = []
                episode_gap_scores = []
                episode_aptitude_scores = []
                
                for episode_trajectory in data['trajectories']:
                    if not episode_trajectory:
                        continue
                    
                    # Calculate total rewards for this episode
                    episode_performance_total = 0.0
                    episode_gap_total = 0.0
                    episode_aptitude_total = 0.0
                    episode_steps = len(episode_trajectory)
                    
                    for step in episode_trajectory:
                        if 'reward' in step and 'reward_dict' in step['reward']:
                            reward_dict = step['reward']['reward_dict']
                            episode_performance_total += reward_dict['performance']
                            episode_gap_total += reward_dict['gap']
                            episode_aptitude_total += reward_dict['aptitude']
                    
                    # Rescale by episode steps for performance and aptitude
                    if episode_steps > 0:
                        episode_performance_scores.append(episode_performance_total / episode_steps)
                        episode_aptitude_scores.append(episode_aptitude_total / episode_steps)
                    
                    # For gap: divide by total failed questions at the end of episode
                    last_step = episode_trajectory[-1]
                    total_failed_questions = len(last_step["state"]["all_failed_questions"])
                    if total_failed_questions > 0:
                        episode_gap_scores.append(episode_gap_total / total_failed_questions)
                    else:
                        episode_gap_scores.append(0.0)
                
                # Calculate average of rescaled scores across all episodes
                if episode_performance_scores:
                    performance_rescaled = np.mean(episode_performance_scores)
                if episode_gap_scores:
                    gap_rescaled = np.mean(episode_gap_scores)
                if episode_aptitude_scores:
                    aptitude_rescaled = np.mean(episode_aptitude_scores)
            
            print(f"Policy name: {policy_name}")
            print(f"Performance rescaled: {performance_rescaled:.4f}")
            print(f"Gap rescaled: {gap_rescaled:.4f}")
            print(f"Aptitude rescaled: {aptitude_rescaled:.4f}")
            
            values = [
                performance_rescaled,
                gap_rescaled, 
                aptitude_rescaled
            ]
            values += values[:1] 
            
            print(f"Scalarized total reward: {(performance_rescaled + gap_rescaled + aptitude_rescaled)/3:.4f}")
            line_color = pub_colors[i % len(pub_colors)]
            line_style = line_styles[i % len(line_styles)]
            marker = markers[i % len(markers)]
            
            ax.plot(angles, values, 
                   color=line_color, 
                   linestyle=line_style,
                   linewidth=4,
                   marker=marker,
                   markersize=12,
                   label=policy_name.replace("_", " "),
                   alpha=0.9,
                   markeredgecolor='white',
                   markeredgewidth=0.5)
            
            ax.fill(angles, values, alpha=0.1, color=line_color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(objectives, fontsize=24, fontweight='bold')
        
        ax.set_ylim(0, 1.1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=20, fontweight='bold')
        
        ax.grid(True, alpha=0.3, color='gray', linewidth=0.5)
        
        ax.legend(loc='best', # bbox_to_anchor=(1.2, 0.5), 
                 fontsize=16, frameon=True, 
                 facecolor='white', edgecolor='gray',
                 fancybox=False, shadow=False,
                 ncol=1, columnspacing=1.0,
                 prop={'weight': 'bold'})
        

        for angle in angles[:-1]:
            ax.axvline(x=angle, color='lightgray', alpha=0.5, linewidth=0.5)
        
        plt.tight_layout()
        

        if save_path:
            # Save as PNG for preview
            save_path_radar = save_path.replace('.png', '_radar.png')
            plt.savefig(save_path_radar, dpi=300, # bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            
            save_path_pdf = save_path.replace('.png', '_radar.pdf')
            plt.savefig(save_path_pdf, bbox_inches='tight', 
                       facecolor='white', edgecolor='none', format='pdf')
            

            print(f"Radar plot saved to:")
            print(f"  PNG: {save_path_radar}")
            print(f"  PDF: {save_path_pdf}")
        
        plt.show()
    
    def plot_mastery_progression(self, policy_data: Dict[str, Dict[str, Any]], max_steps: int = 50, save_path: str = None, config_name: str = "Default"):
        """
        Create mastery progression plot.
        
        Args:
            policy_data: Dictionary mapping policy names to their data
            max_steps: Maximum number of steps to plot
            save_path: Path to save the plot (optional)
            config_name: Name of this configuration set for plot title
        """
        max_steps=50
        fig = plt.figure(figsize=(8, 4), facecolor='white')
        ax = plt.gca()
        
        # Set elegant background colors
        ax.set_facecolor('#ffffff')
        
        pub_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        markers = ['o', 's', '^', 'D', 'p', 'h', '*', 'x']
        line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':']

        all_policies = list(policy_data.items())
        
        for i, (policy_name, data) in enumerate(all_policies):
            mastery_progression = data['mastery_progression']
            
            # Plot all steps without stopping at threshold
            plot_steps = []
            plot_mastery = []
            
            for step_idx, mastery_val in enumerate(mastery_progression):
                if not np.isnan(mastery_val):
                    plot_steps.append(step_idx + 1)  # Convert to 1-based indexing
                    plot_mastery.append(mastery_val)
            
            if plot_steps:
                line_color = pub_colors[i % len(pub_colors)]
                line_style = line_styles[i % len(line_styles)]
                
                # Plot mastery progression line with marker and line style
                plt.plot(plot_steps, plot_mastery, 
                        color=line_color, 
                        linestyle=line_style,
                        linewidth=4,
                        marker=markers[i % len(markers)],
                        markersize=20,
                        label=policy_name.replace("_", " "),
                        alpha=0.9,
                        markeredgecolor='#2c3e50',
                        markeredgewidth=1)
        
        # Add elegant mastery threshold horizontal line at 0.8
        plt.axhline(y=0.8, color='#e74c3c', linestyle='--', linewidth=4, alpha=0.8, 
                   label='Mastery Threshold')
        
        # Elegant axis styling
        plt.xlabel('Steps', fontsize=30, color='#2c3e50')
        plt.ylabel('Mastery', fontsize=30, color='#2c3e50')
        # plt.title(f'{config_name}', 
        #          fontsize=30, color='#2c3e50', pad=20)
        
        # Elegant legend styling
        plt.legend(loc='lower right', fontsize=20, frameon=True, 
                  facecolor='#ffffff', edgecolor='#bdc3c7', 
                  fancybox=True, shadow=True,
                  )

        # Elegant grid styling
        # plt.grid(True, alpha=0.4, color='#bdc3c7', linewidth=0.8)
        plt.xlim(0, max_steps + 1)  # Set x-axis limit to show steps 1-max_steps
        plt.ylim(-0.1, 1.1)
        
        # Make tick labels larger and bold with elegant colors
        plt.tick_params(axis='both', which='major', labelsize=30)
        ax = plt.gca()
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            # label.set_fontweight('bold')
            label.set_color('#2c3e50')
        
        # Elegant axis spine styling
        for spine in ax.spines.values():
            spine.set_color('#bdc3c7')
            spine.set_linewidth(1.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, # bbox_inches='tight', 
                       facecolor='white')
            print(f"Mastery progression plot saved to: {save_path}")
        
        plt.show()

    def plot_correlation_scatter(self, policy_data: Dict[str, Dict[str, Any]], save_path: str = None, config_name: str = "Default", optimized_obj: str = "performance"):
        """
        Create correlation scatter plot for two objectives (excluding optimized_obj).
        
        Args:
            policy_data: Dictionary mapping policy names to their data
            save_path: Path to save the plot (optional)
            config_name: Name of this configuration set for plot title
            optimized_obj: Objective to exclude from the plot (x and y axes will be the other two objectives)
        """
        plt.rcParams.update({
            'font.family': 'DejaVu Sans',
            'font.size': 10,
            'axes.titlesize': 11,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 12,
            'text.usetex': False,  # Set to True if LaTeX is available
        })
        
        fig = plt.figure(figsize=(10, 8), facecolor='white')
        ax = plt.gca()
        
        # Set clean background colors
        ax.set_facecolor('#ffffff')

        pub_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        markers = ['o', 's', '^', 'D', 'p', 'h', '*', 'x']
        
        # Determine which objectives to plot based on optimized_obj
        all_objectives = ["performance", "gap", "aptitude"]
        plot_objectives = [obj for obj in all_objectives if obj != optimized_obj]
        
        if len(plot_objectives) != 2:
            raise ValueError(f"Expected exactly 2 objectives to plot, but got {len(plot_objectives)}. Optimized objective: {optimized_obj}")
        
        x_obj, y_obj = plot_objectives[0], plot_objectives[1]
        
        # Calculate total steps from data for normalization
        total_steps = 0
        for policy_name, data in policy_data.items():
            trajectories = data['trajectories']
            for episode_trajectory in trajectories:
                total_steps = max(total_steps, len(episode_trajectory))
        
        # If no data found, use default
        if total_steps == 0:
            raise ValueError(f"No data found for {config_name}")
        
        all_policies = list(policy_data.items())
        
        for i, (policy_name, data) in enumerate(all_policies):
            # Extract episode-level rewards for the two objectives to plot
            episode_x_rewards = []
            episode_y_rewards = []
            
            # Get trajectories from the policy data
            trajectories = data['trajectories']
            
            for episode_trajectory in trajectories:
                episode_x = 0.0
                episode_y = 0.0
                
                for step in episode_trajectory:
                    if "reward" in step and "reward_dict" in step["reward"]:
                        reward_dict = step["reward"]["reward_dict"]
                        
                        if x_obj in reward_dict:
                            episode_x += reward_dict[x_obj]
                        if y_obj in reward_dict:
                            episode_y += reward_dict[y_obj]
                
                # Normalize based on objective type
                if x_obj == "gap" or y_obj == "gap":
                    # For gap reward, normalize by the number of failed questions at the end of episode
                    if episode_trajectory:
                        last_step = episode_trajectory[-1]
                        if "state" in last_step and "all_failed_questions" in last_step["state"]:
                            total_failed_questions = len(last_step["state"]["all_failed_questions"])
                        else:
                            raise ValueError(f"No failed questions found for {policy_name}")
                    else:
                        raise ValueError(f"No episode trajectory found for {policy_name}")
                    
                    # Normalize gap reward by failed questions count, other objectives by total steps
                    if x_obj == "gap":
                        episode_x_normalized = episode_x / total_failed_questions if total_failed_questions > 0 else episode_x
                        episode_y_normalized = episode_y / total_steps
                    else:  # y_obj == "gap"
                        episode_x_normalized = episode_x / total_steps
                        episode_y_normalized = episode_y / total_failed_questions if total_failed_questions > 0 else episode_y
                else:
                    # For non-gap objectives, normalize by total steps
                    episode_x_normalized = episode_x / total_steps
                    episode_y_normalized = episode_y / total_steps
                
                episode_x_rewards.append(episode_x_normalized)
                episode_y_rewards.append(episode_y_normalized)
            
            if episode_x_rewards and episode_y_rewards:
                color = pub_colors[i % len(pub_colors)]
                marker = markers[i % len(markers)]
                
                # Create scatter plot with professional styling
                ax.scatter(episode_x_rewards, episode_y_rewards,
                          c=color,
                          marker=marker,
                          s=80,
                          label=policy_name,
                          alpha=0.7,
                          edgecolors='#2c3e50',
                          linewidth=0.8,
                          zorder=3)
        
        # Professional axis styling with normalized labels
        x_label = f'Normalized {x_obj.title()} Reward'
        y_label = f'Normalized {y_obj.title()} Reward'
        
        # Add normalization info to labels
        # if x_obj == "gap":
        #     x_label += ' (÷failed_questions)'
        # else:
        #     x_label += f' (÷{total_steps})'
            
        # if y_obj == "gap":
        #     y_label += ' (÷failed_questions)'
        # else:
        #     y_label += f' (÷{total_steps})'
        
        ax.set_xlabel(x_label, fontsize=24, fontweight='bold', color='#2c3e50')
        ax.set_ylabel(y_label, fontsize=24, fontweight='bold', color='#2c3e50')
        # ax.set_title(f'{x_obj.title()} vs {y_obj.title()} Reward Correlation - {config_name}', 
        #              fontsize=18, fontweight='bold', color='#2c3e50', pad=20)
        
        # Professional legend styling
        ax.legend(loc='best', fontsize=16, frameon=True, 
                 facecolor='#ffffff', edgecolor='#bdc3c7', 
                 fancybox=True, shadow=True)
        
        # Professional grid styling
        ax.grid(True, alpha=0.4, color='#bdc3c7', linewidth=0.8)
        
        # Set axis limits
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        
        # Make tick labels larger and bold with elegant colors
        ax.tick_params(axis='both', which='major', labelsize=16)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
            label.set_color('#2c3e50')
        
        # Elegant axis spine styling
        for spine in ax.spines.values():
            spine.set_color('#bdc3c7')
            spine.set_linewidth(1.5)
        
        # Add correlation coefficient text
        if len(episode_x_rewards) > 1 and len(episode_y_rewards) > 1:
            correlation = np.corrcoef(episode_x_rewards, episode_y_rewards)[0, 1]
            ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                   transform=ax.transAxes, fontsize=14, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                   verticalalignment='top')
        
        plt.tight_layout()
        
        if save_path:
            save_path_correlation = save_path.replace('.png', f'_correlation_{x_obj}_vs_{y_obj}.png')
            plt.savefig(save_path_correlation, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            save_path_pdf = save_path.replace('.png', f'_correlation_{x_obj}_vs_{y_obj}.pdf')
            plt.savefig(save_path_pdf, bbox_inches='tight', 
                       facecolor='white', edgecolor='none', format='pdf')
            
            # Save as SVG for vector editing
            # save_path_svg = save_path.replace('.png', f'_correlation_{x_obj}_vs_{y_obj}.svg')
            # plt.savefig(save_path_svg, bbox_inches='tight', 
            #            facecolor='white', edgecolor='none', format='svg')
            
            print(f"Correlation scatter plot ({x_obj} vs {y_obj}) saved to:")
            print(f"  PNG: {save_path_correlation}")
            print(f"  PDF: {save_path_pdf}")
            # print(f"  SVG: {save_path_svg}")
        
        plt.show()

    def plot_flow_zone(self, policy_data: Dict[str, Dict[str, Any]], save_path: str = None, config_name: str = "Default"):
        """
        Create flow zone plot showing mastery level vs difficulty level for one episode.
        
        Args:
            policy_data: Dictionary mapping policy names to their data
            save_path: Path to save the plot (optional)
            config_name: Name of this configuration set for plot title
        """
        from matplotlib.patches import Polygon
        from scipy.spatial import ConvexHull
        
        plt.rcParams.update({
            'font.family': 'DejaVu Sans',
            'font.size': 10,
            'axes.titlesize': 11,
            'axes.labelsize': 30,
            'xtick.labelsize': 30,
            'ytick.labelsize': 30,
            'legend.fontsize': 24,  # Set legend font size to 16
            'figure.titlesize': 12,
            'text.usetex': False,  # Set to True if LaTeX is available
            'axes.grid': False,  # Explicitly disable grid
        })
        
        fig = plt.figure(figsize=(8, 4), facecolor='white')
        ax = plt.gca()
        
        # Set clean background colors
        ax.set_facecolor('#ffffff')
        
        pub_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        markers = ['o', 's', '^', 'D', 'p', 'h', '*', 'x']
        
        # Plot all policies
        all_policies = list(policy_data.items())
        
        # Collect all points for each policy to draw ellipses
        policy_points = {}
        
        for i, (policy_name, data) in enumerate(all_policies):
            trajectories = data['trajectories']
            
            if not trajectories:
                print(f"Warning: No trajectories found for {policy_name}")
                continue
            
            # Use the first episode for flow zone plot
            episode_trajectory = trajectories[-1]
            
            # Extract mastery levels and question difficulties for each step
            step_mastery_levels = []
            step_difficulties = []
            step_means = []
            
            # Collect all points for this policy
            all_x_coords = []
            all_y_coords = []
            
            for step_idx, step in enumerate(episode_trajectory):
                # Calculate average mastery level across all skills
                mastery_values = list(step["state"]["mastery"].values())
                avg_mastery = np.mean(mastery_values)
                step_mastery_levels.append(avg_mastery)
                
                # Extract question difficulties from this step
                questions_info = step["action"]["questions_info"]
                
                # Get difficulties for the 5 recommended questions
                step_diffs = []
                for q_info in questions_info:
                    if "scaled_difficulty" in q_info:
                        step_diffs.append(q_info["scaled_difficulty"])
                    else:
                        raise ValueError(f"No scaled difficulty found for {q_info}")
                
                step_difficulties.append(step_diffs)
                step_means.append(np.mean(step_diffs))
                
                # Collect all points for ellipse calculation
                for diff in step_diffs:
                    all_x_coords.append(avg_mastery)
                    all_y_coords.append(diff)
            
            # Store points for this policy
            policy_points[policy_name] = (all_x_coords, all_y_coords)
            
            # Use  color palette
            color = pub_colors[i % len(pub_colors)]
            marker = markers[i % len(markers)]
            
            # Plot individual question difficulties for each step
            for step_idx, (mastery, diffs) in enumerate(zip(step_mastery_levels, step_difficulties)):
                # Plot 5 dots for this step (same mastery level, different difficulties)
                x_coords = [mastery] * len(diffs)
                y_coords = diffs
                
                # Use same marker for all steps of this policy, different marker for different policies
                ax.scatter(x_coords, y_coords,
                          c=color,
                          marker=marker,
                          s=120,
                          alpha=0.6,
                          edgecolors='#2c3e50',
                          linewidth=2.0,
                          zorder=2,
                          label=policy_name.replace("_", " ") if step_idx == 0 else "")
        
        # Draw convex hulls for each policy
        for i, (policy_name, (x_coords, y_coords)) in enumerate(policy_points.items()):
            if len(x_coords) > 2:  # Need at least 3 points to draw convex hull
                # Prepare points for convex hull
                points = np.column_stack((x_coords, y_coords))
                
                try:
                    # Calculate convex hull
                    hull = ConvexHull(points)
                    
                    # Get hull vertices
                    hull_vertices = points[hull.vertices]
                    
                    # Create polygon for convex hull
                    hull_polygon = Polygon(hull_vertices,
                                         facecolor=pub_colors[i % len(pub_colors)], 
                                         alpha=0.2, 
                                         edgecolor=pub_colors[i % len(pub_colors)],
                                         linewidth=2,
                                         zorder=1)
                    ax.add_patch(hull_polygon)
                    
                except Exception as e:
                    print(f"Warning: Could not create convex hull for {policy_name}: {e}")
                    continue
        
        # Add flow zone window (shaded area between x=y+0.1 and x=y-0.1)
        # Create a polygon for the flow zone
        x_flow = np.linspace(0, 1, 100)
        y_upper = np.clip(x_flow + 0.1, 0, 1)  # x + 0.1, clipped to [0,1]
        y_lower = np.clip(x_flow - 0.1, 0, 1)  # x - 0.1, clipped to [0,1]
        
        # Fill the area between the two lines with gray
        ax.fill_between(x_flow, y_lower, y_upper, 
                       color='gray', alpha=0.3, 
                       label='Flow Zone',
                       zorder=0)
        
        # Add dashed gray lines for the boundaries
        ax.plot(x_flow, y_upper, '--', color='gray', alpha=0.7, linewidth=2, zorder=0)
        ax.plot(x_flow, y_lower, '--', color='gray', alpha=0.7, linewidth=2, zorder=0)
        
        # Professional axis styling
        ax.set_xlabel('Mastery', fontsize=30, color='#2c3e50')
        ax.set_ylabel('Difficulty', fontsize=30, color='#2c3e50')
        # ax.set_title(f'Flow Zone: Mastery vs Difficulty Progression - {config_name}', 
        #             fontsize=18, fontweight='bold', color='#2c3e50', pad=20)
        
        # Set axis limits
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        
        # Professional legend styling
        ax.legend(loc='lower right', fontsize=20, frameon=True, 
                 facecolor='#ffffff', edgecolor='#bdc3c7', 
                 fancybox=True, shadow=True,
                 )
        
        # Explicitly disable grid
        ax.grid(False)
        
        # Make tick labels larger and bold with elegant colors
        ax.tick_params(axis='both', which='major', labelsize=30)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            # label.set_fontweight('bold')
            label.set_color('#2c3e50')
        
        # Elegant axis spine styling
        for spine in ax.spines.values():
            spine.set_color('#bdc3c7')
            spine.set_linewidth(1.5)
        
        # Add explanation for markers
        # ax.text(0.02, 0.98, 'Markers: Different shapes represent different policies', 
        #        transform=ax.transAxes, fontsize=10, fontweight='bold',
        #        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
        #        verticalalignment='top')
        
        plt.tight_layout()
        
        if save_path:
            # Save in multiple formats
            save_path_flow = save_path.replace('.png', '_flow_zone.png')
            plt.savefig(save_path_flow, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            
            # Save as PDF for LaTeX integration
            save_path_pdf = save_path.replace('.png', '_flow_zone.pdf')
            plt.savefig(save_path_pdf, bbox_inches='tight', 
                       facecolor='white', edgecolor='none', format='pdf')
            
            # Save as SVG for vector editing
            # save_path_svg = save_path.replace('.png', '_flow_zone.svg')
            # plt.savefig(save_path_svg, bbox_inches='tight', 
            #            facecolor='white', edgecolor='none', format='svg')
            
            print(f"Flow zone plot saved to:")
            print(f"  PNG: {save_path_flow}")
            print(f"  PDF: {save_path_pdf}")
            # print(f"  SVG: {save_path_svg}")
        
        plt.show()


    def create_combined_pareto_plot(self, all_config_data: Dict[str, Dict[str, Dict[str, Any]]], save_path: str = None):
        """
        Create combined Pareto optimality plot with subplots for each configuration.
        
        Args:
            all_config_data: Dictionary mapping config names to policy data
            save_path: Path to save the combined plot
        """
        # Fixed 2x2 layout with 3D projectio
        fig = plt.figure(figsize=(16, 14), facecolor='white')
        axes = []
        
        # Create 3D subplots
        for i in range(4):
            ax = fig.add_subplot(2, 2, i+1, projection='3d')
            # Set clean background colors for each subplot
            ax.set_facecolor('white')
            # Use modern matplotlib 3D axis properties
            try:
                # Modern matplotlib 3D pane color API
                ax.xaxis.set_pane_color((0.95, 0.95, 0.95, 0.8))
                ax.yaxis.set_pane_color((0.95, 0.95, 0.95, 0.8))
                ax.zaxis.set_pane_color((0.95, 0.95, 0.95, 0.8))
            except AttributeError:
                # Legacy matplotlib 3D pane color API
                try:
                    ax.w_xaxis.set_pane_color((0.95, 0.95, 0.95, 0.8))
                    ax.w_yaxis.set_pane_color((0.95, 0.95, 0.95, 0.8))
                    ax.w_zaxis.set_pane_color((0.95, 0.95, 0.95, 0.8))
                except AttributeError:
                    pass  # Skip if neither method works
            axes.append(ax)
        
        # Define  color palette and markers
        pub_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        markers = ['o', 's', '^', 'D', 'p', 'h', '*', 'x']
        
        for idx, (config_name, policy_data) in enumerate(all_config_data.items()):
            if idx >= 4:  # Only show first 4 configurations
                break
            ax = axes[idx]
            
            # Plot all policies with styling
            all_policies = list(policy_data.items())
            performance_values = []
            gap_values = []
            aptitude_values = []
            
            for i, (policy_name, data) in enumerate(all_policies):
                performance = data['avg_performance']
                gap = data['avg_gap']
                aptitude = data['avg_aptitude']
                
                performance_values.append(performance)
                gap_values.append(gap)
                aptitude_values.append(aptitude)
                
               
                color = pub_colors[i % len(pub_colors)]
                
                # Create 3D scatter plot with elegant styling
                ax.scatter(performance, gap, aptitude, 
                          c=color, 
                          marker=markers[i % len(markers)],
                          s=200,
                          label=policy_name,
                          edgecolors='#2c3e50',
                          linewidth=1.5,
                          alpha=0.8,
                          zorder=3)
            
            # Find and plot 3D Pareto front
            if len(performance_values) > 1:
                points = list(zip(performance_values, gap_values, aptitude_values))
                pareto_front_3d = self.find_3d_pareto_front(points)
                
                if len(pareto_front_3d) > 1:
                    # Highlight Pareto front points with elegant styling
                    for x, y, z in pareto_front_3d:
                        ax.scatter(x, y, z, c='#e74c3c', s=80, marker='*', linewidth=2, alpha=0.9, zorder=5)
            
            # Elegant axis styling
            ax.set_xlabel('Performance Reward', fontsize=12, fontweight='bold', color='#2c3e50')
            ax.set_ylabel('Gap Reward', fontsize=12, fontweight='bold', color='#2c3e50')
            ax.set_zlabel('Aptitude Reward', fontsize=12, fontweight='bold', color='#2c3e50')
            ax.set_title(f'{config_name}', fontsize=14, fontweight='bold', color='#2c3e50')
            
            # Elegant grid and legend styling
            ax.grid(True, alpha=0.4, color='#bdc3c7')
            ax.legend(loc='best', fontsize=10, frameon=True, 
                     facecolor='#ffffff', edgecolor='#bdc3c7', fancybox=True, shadow=True)
            
            # Set axes limits to accommodate large markers
            ax.set_xlim(left=-1)
            ax.set_ylim(bottom=-1)
            ax.set_zlim(bottom=-1)
            
            # Make tick labels larger with elegant colors
            ax.tick_params(axis='both', which='major', labelsize=10)
            for label in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
                label.set_color('#2c3e50')
        
        # Update title to reflect 3D analysis
        plt.suptitle('Combined 3D Pareto Optimality: Performance vs Gap vs Aptitude', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Combined 3D Pareto optimality plot saved to: {save_path}")
        
        plt.show()

    def create_combined_radar_plot(self, all_config_data: Dict[str, Dict[str, Dict[str, Any]]], save_path: str = None, max_steps: int = 30):
        """
        Create combined radar plot for Pareto optimality with subplots for each configuration.
        
        Args:
            all_config_data: Dictionary mapping config names to policy data
            save_path: Path to save the combined plot
        """
        fig = plt.figure(figsize=(10, 8), facecolor='white')
        axes = []
        
        # Create polar subplots with clean styling
        for i in range(4):
            ax = fig.add_subplot(2, 2, i+1, projection='polar')
            ax.set_facecolor('white')
            axes.append(ax)
        
        # Define objectives for radar plot
        objectives = ['Performance', 'Gap', 'Aptitude']
        num_vars = len(objectives)
        
        # Compute angle for each axis
        angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
        angles += angles[:1]  # Complete the circle
        
        pub_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        markers = ['o', 's', '^', 'D', 'p', 'h', '*', 'x']
        line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
        
        for idx, (config_name, policy_data) in enumerate(all_config_data.items()):
            if idx >= 4:  # Only show first 4 configurations
                break
            ax = axes[idx]
            
            all_policies = list(policy_data.items())
            
            for i, (policy_name, data) in enumerate(all_policies):
                # Extract values for each objective and rescale them
                performance = data['avg_performance']
                gap = data['avg_gap']
                aptitude = data['avg_aptitude']
                
                # Get average number of steps for this policy
                avg_steps = data['avg_stop_step']
                print(f"Average steps: {avg_steps}")
                
                # Rescale based on objective type
                # For performance and aptitude: divide by average steps for this policy
                performance_rescaled = performance / float(avg_steps)
                aptitude_rescaled = aptitude / float(avg_steps)
                
                # For gap: divide by the number of failed questions at the end of episode
                gap_rescaled = gap
                if 'trajectories' in data and data['trajectories']:
                    # Get the first episode trajectory
                    episode_trajectory = data['trajectories'][0]
                    if episode_trajectory:
                        last_step = episode_trajectory[-1]
                        if "state" in last_step and "all_failed_questions" in last_step["state"]:
                            total_failed_questions = len(last_step["state"]["all_failed_questions"])
                            if total_failed_questions > 0:
                                gap_rescaled = gap / total_failed_questions
                
                values = [
                    performance_rescaled,
                    gap_rescaled, 
                    aptitude_rescaled
                ]
                values += values[:1]  # Complete the circle
                
                line_color = pub_colors[i % len(pub_colors)]
                line_style = line_styles[i % len(line_styles)]
                marker = markers[i % len(markers)]
                
                ax.plot(angles, values, linewidth=1.5, 
                       color=line_color, 
                       marker=marker,
                       markersize=8,
                       label=policy_name,
                       alpha=0.9,
                       linestyle=line_style,
                       markeredgecolor='white',
                       markeredgewidth=0.5)
                
                # Semi-transparent fill for overlapping areas
                ax.fill(angles, values, alpha=0.1, color=line_color)
            
            # Professional axis styling
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(objectives, fontsize=9, fontweight='normal')
            
            # Set y-axis limits and labels
            ax.set_ylim(0, 1)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
            
            # Add clear grid
            ax.grid(True, alpha=0.3, color='gray', linewidth=0.5)
            
            # Professional legend styling
            ax.legend(loc='upper right', fontsize=7, frameon=True, 
                     facecolor='white', edgecolor='gray', fancybox=False, shadow=False)
            
            # Professional title styling
            ax.set_title(f'{config_name}', fontsize=10, fontweight='normal')
            
            # Add subtle reference lines for better readability
            for angle in angles[:-1]:
                ax.axvline(x=angle, color='lightgray', alpha=0.5, linewidth=0.5)
        
        # Professional title styling
        plt.suptitle('Combined Pareto Optimality Analysis', fontsize=14, fontweight='normal', color='black')
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        if save_path:
            save_path_combined_radar = save_path.replace('.png', '_combined_radar.png')
            plt.savefig(save_path_combined_radar, dpi=300, bbox_inches='tight', facecolor='white')
            
            save_path_pdf = save_path.replace('.png', '_combined_radar.pdf')
            plt.savefig(save_path_pdf, bbox_inches='tight', facecolor='white', format='pdf')
            
            print(f"Combined radar plot saved to:")
            print(f"  PNG: {save_path_combined_radar}")
            print(f"  PDF: {save_path_pdf}")
        
        plt.show()

    

    

    
    def create_combined_mastery_plot(self, all_config_data: Dict[str, Dict[str, Dict[str, Any]]], max_steps: int = 50, save_path: str = None):
        """
        Create combined mastery progression plot with subplots for each configuration.
        
        Args:
            all_config_data: Dictionary mapping config names to policy data
            max_steps: Maximum number of steps for mastery progression
            save_path: Path to save the combined plot
        """
        # Fixed 2x2 layout with elegant styling
        fig = plt.figure(figsize=(10, 8), facecolor='#f8f9fa')
        axes = []
        
        # Create subplots with elegant styling
        for i in range(4):
            ax = fig.add_subplot(2, 2, i+1)
            ax.set_facecolor('#ffffff')
            axes.append(ax)
        
        pub_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        markers = ['o', 's', '^', 'D', 'p', 'h', '*', 'x']
        line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':']
        
        for idx, (config_name, policy_data) in enumerate(all_config_data.items()):
            if idx >= 4:  # Only show first 4 configurations
                break
            ax = axes[idx]
            

            all_policies = list(policy_data.items())
            
            for i, (policy_name, data) in enumerate(all_policies):
                mastery_progression = data['mastery_progression']
                
                # Plot all steps without stopping at threshold
                plot_steps = []
                plot_mastery = []
                
                for step_idx, mastery_val in enumerate(mastery_progression):
                    if not np.isnan(mastery_val):
                        plot_steps.append(step_idx + 1)  # Convert to 1-based indexing
                        plot_mastery.append(mastery_val)
                
                if plot_steps:
                    line_color = pub_colors[i % len(pub_colors)]
                    line_style = line_styles[i % len(line_styles)]
                    
                    # Plot mastery progression line with marker and line style
                    ax.plot(plot_steps, plot_mastery, 
                           color=line_color, 
                           linestyle=line_style,
                           linewidth=2.5,
                           marker=markers[i % len(markers)],
                           markersize=6,
                           label=policy_name,
                           alpha=0.9,
                           markeredgecolor='#2c3e50',
                           markeredgewidth=1)
            
            # Elegant axis styling
            ax.set_xlabel('Steps', fontsize=12, fontweight='bold', color='#2c3e50')
            ax.set_ylabel('Mastery Level', fontsize=12, fontweight='bold', color='#2c3e50')
            ax.set_title(f'{config_name}', fontsize=14, fontweight='bold', color='#2c3e50')
            
            # Elegant legend styling
            ax.legend(loc='best', fontsize=10, frameon=True, 
                     facecolor='#ffffff', edgecolor='#bdc3c7', fancybox=True, shadow=True)
            
            # Elegant grid styling
            ax.grid(True, alpha=0.4, color='#bdc3c7', linewidth=0.8)
            ax.set_xlim(0.5, max_steps + 0.5)  # Fixed x-axis range for max_steps steps
            ax.set_ylim(-0.05, 1)
            
            # Make tick labels larger with elegant colors
            ax.tick_params(axis='both', which='major', labelsize=10)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_color('#2c3e50')
            
            # Elegant axis spine styling
            for spine in ax.spines.values():
                spine.set_color('#bdc3c7')
                spine.set_linewidth(1.5)
        
        # Add elegant mastery threshold horizontal line at 0.8 to all subplots
        for idx in range(4):  # Fixed 2x2 layout
            axes[idx].axhline(y=0.8, color='#e74c3c', linestyle='--', linewidth=2.5, alpha=0.8, label='Mastery Threshold (0.8)')
        
        # Elegant title styling
        plt.suptitle('Combined Mastery Progression Over Time', fontsize=18, fontweight='bold', color='#2c3e50')
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Combined mastery progression plot saved to: {save_path}")
        
        plt.show()

    def visualize_multiple_configs(self, config_sets: Dict[str, Dict[str, str]], max_steps: int = 50, combined_plots: bool = False, correlation_scatter: bool = False, optimized_obj: str = "performance", flow_zone: bool = False):
        """
        Visualize multiple configuration sets separately or combined.
        
        Args:
            config_sets: Dictionary mapping configuration names to policy configurations
            max_steps: Maximum number of steps for mastery progression
            combined_plots: If True, generate combined plots; if False, generate separate plots
            correlation_scatter: If True, generate correlation scatter plot
        """
        print(f"Processing {len(config_sets)} configuration sets...")
        
        if combined_plots:
            print("Generating combined plots...")
            
            # Load all configuration data first
            all_config_data = {}
            
            for config_name, policy_configs in config_sets.items():
                print(f"Loading data for configuration: {config_name}")
                config_policy_data = {}
                
                for policy_name, result_folder in policy_configs.items():
                    try:
                        trajectories = self.load_trajectory_data(result_folder)
                        pareto_metrics, objective_rewards = self.calculate_pareto_metrics(trajectories, result_folder=result_folder)
                        avg_performance, avg_gap, avg_aptitude = pareto_metrics
                        mastery_progression, avg_stop_step = self.calculate_mastery_progression(trajectories, max_steps)
                        
                        config_policy_data[policy_name] = {
                            'avg_performance': avg_performance,
                            'avg_gap': avg_gap,
                            'avg_aptitude': avg_aptitude,
                            'mastery_progression': mastery_progression,
                            'avg_stop_step': avg_stop_step
                        }
                        
                        print(f"  {policy_name}: Performance: {avg_performance:.3f}, Gap: {avg_gap:.3f}, Aptitude: {avg_aptitude:.3f}, Avg Stop: {avg_stop_step:.1f}")
                        
                    except Exception as e:
                        print(f"Error processing {policy_name}: {e}")
                        continue
                
                if config_policy_data:
                    all_config_data[config_name] = config_policy_data
            
            if not all_config_data:
                print("No valid configuration data found!")
                return
            
            if correlation_scatter:
                # Only generate correlation scatter plots for each configuration
                for config_name, config_policy_data in all_config_data.items():
                    safe_config_name = config_name.replace(" ", "_").replace("/", "_")
                    correlation_save_path = os.path.join(self.output_dir, f"correlation_scatter_{safe_config_name}.png")
                    self.plot_correlation_scatter(config_policy_data, correlation_save_path, config_name, optimized_obj)
                
                print("Correlation scatter plots complete!")
            elif flow_zone:
                # Only generate flow zone plots for each configuration
                for config_name, config_policy_data in all_config_data.items():
                    safe_config_name = config_name.replace(" ", "_").replace("/", "_")
                    flow_zone_save_path = os.path.join(self.output_dir, f"flow_zone_{safe_config_name}.png")
                    self.plot_flow_zone(config_policy_data, flow_zone_save_path, config_name)
                
                print("Flow zone plots complete!")
            else:
                # Generate combined plots
                combined_pareto_save_path = os.path.join(self.output_dir, "combined_pareto_optimality.png")
                combined_mastery_save_path = os.path.join(self.output_dir, "combined_mastery_progression.png")
                combined_radar_save_path = os.path.join(self.output_dir, "combined_radar_pareto_optimality.png")
                
                self.create_combined_pareto_plot(all_config_data, combined_pareto_save_path)
                self.create_combined_radar_plot(all_config_data, combined_radar_save_path, max_steps)
                self.create_combined_mastery_plot(all_config_data, max_steps, combined_mastery_save_path)
                
                print("Combined visualization complete!")
            
        else:
            # Generate separate plots for each configuration
            for config_name, policy_configs in config_sets.items():
                print(f"\n{'='*60}")
                print(f"Processing Configuration: {config_name}")
                print(f"{'='*60}")
                
                try:
                    self.visualize_policies(policy_configs, max_steps, config_name, correlation_scatter, optimized_obj, flow_zone)
                except Exception as e:
                    print(f"Error processing configuration {config_name}: {e}")
                    continue
            
            print(f"\n{'='*60}")
            print("All configurations processed!")
            print(f"{'='*60}")

    def calculate_avg_tokens(self, config_data: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
        """
        Calculate average input, output, and total tokens for each policy and save to txt file.
        
        Args:
            config_data: Dictionary containing configuration data
        """
        print("\nCalculating average token usage...")
        
        # Collect token data for all policies
        all_policy_token_data = {}
        
        for config_name, policies_data in config_data.items():
            for policy_name, data in policies_data.items():
                trajectories = data['trajectories']
                
                input_tokens_list = []
                output_tokens_list = []
                total_tokens_list = []
                episode_lengths = []
                
                for episode_trajectory in trajectories:
                    # Record episode length
                    episode_lengths.append(len(episode_trajectory))
                    
                    for step in episode_trajectory:
                        if "token" in step:
                            token_data = step["token"]
                            if "input_tokens" in token_data:
                                input_tokens_list.append(token_data["input_tokens"])
                            if "output_tokens" in token_data:
                                output_tokens_list.append(token_data["output_tokens"])
                            if "total_tokens" in token_data:
                                total_tokens_list.append(token_data["total_tokens"])
                
                # Calculate averages
                avg_input_tokens = np.mean(input_tokens_list) if input_tokens_list else 0.0
                avg_output_tokens = np.mean(output_tokens_list) if output_tokens_list else 0.0
                avg_total_tokens = np.mean(total_tokens_list) if total_tokens_list else 0.0
                avg_trajectory_length = np.mean(episode_lengths) if episode_lengths else 0.0
                
                all_policy_token_data[policy_name] = {
                    'avg_input_tokens': avg_input_tokens,
                    'avg_output_tokens': avg_output_tokens,
                    'avg_total_tokens': avg_total_tokens,
                    'avg_trajectory_length': avg_trajectory_length,
                    'total_steps': len(input_tokens_list)
                }
                
                print(f"  {policy_name}: Avg Input: {avg_input_tokens:.2f}, Avg Output: {avg_output_tokens:.2f}, Avg Total: {avg_total_tokens:.2f}, Avg Length: {avg_trajectory_length:.2f}, Steps: {len(input_tokens_list)}")
        
        # Save to txt file
        token_file_path = os.path.join(self.output_dir, "token_usage_analysis.txt")
        
        with open(token_file_path, 'w') as f:
            f.write("Policy Name\tAvg Input Tokens\tAvg Output Tokens\tAvg Total Tokens\tAvg Trajectory Length\tTotal Steps\n")
            f.write("-" * 120 + "\n")
            
            for policy_name, token_data in all_policy_token_data.items():
                f.write(f"{policy_name}\t{token_data['avg_input_tokens']:.2f}\t{token_data['avg_output_tokens']:.2f}\t{token_data['avg_total_tokens']:.2f}\t{token_data['avg_trajectory_length']:.2f}\t{token_data['total_steps']}\n")
        
        print(f"Token usage analysis saved to: {token_file_path}")

    def calculate_cost_only(self, config_sets: Dict[str, Dict[str, str]]):
        """
        Calculate and save token usage analysis only (no plots).
        
        Args:
            config_sets: Dictionary mapping configuration names to policy configurations
        """
        print(f"Processing {len(config_sets)} configurations for cost calculation...")
        
        # Load data for all configurations
        config_data = {}
        
        for config_name, policy_configs in config_sets.items():
            print(f"Loading data for configuration: {config_name}")
            config_data[config_name] = {}
            
            for policy_name, result_folder in policy_configs.items():
                try:
                    # Load trajectory data only (no evaluation results needed for cost calculation)
                    trajectories = self.load_trajectory_data(result_folder)
                    
                    config_data[config_name][policy_name] = {
                        'trajectories': trajectories
                    }
                    
                    print(f"  {policy_name}: Data loaded successfully")
                    
                except Exception as e:
                    print(f"Error loading data for {policy_name}: {e}")
                    continue
        
        if not config_data:
            print("No valid configuration data found!")
            return
        
        # Calculate token usage only
        self.calculate_avg_tokens(config_data)
        
        print("Cost calculation complete!")

    def analyze_orchestrator_performance(self, config_sets: Dict[str, Dict[str, str]]):
        """
        Analyze orchestrator performance with three specific plots.
        
        Args:
            config_sets: Dictionary mapping configuration names to policy configurations
        """
        print(f"Processing {len(config_sets)} configurations for orchestrator analysis...")
        
        # Load data for all configurations
        config_data = {}
        
        for config_name, policy_configs in config_sets.items():
            print(f"Loading data for configuration: {config_name}")
            config_data[config_name] = {}
            
            for policy_name, result_folder in policy_configs.items():
                try:
                    # Load both trajectory and evaluation results
                    trajectories = self.load_trajectory_data(result_folder)
                    evaluation_results = self.load_evaluation_results(result_folder)
                    
                    config_data[config_name][policy_name] = {
                        'trajectories': trajectories,
                        'evaluation_results': evaluation_results
                    }
                    
                    print(f"  {policy_name}: Data loaded successfully")
                    
                except Exception as e:
                    print(f"Error loading data for {policy_name}: {e}")
                    continue
        
        if not config_data:
            print("No valid configuration data found!")
            return
        
        # Calculate token usage
        self.calculate_avg_tokens(config_data)
        
        # Generate the three analysis plots
        print("\nGenerating orchestrator analysis plots...")
        
        # 1. Step latency plot
        self.plot_step_latency(config_data)
        
        # 2. Policy usage ratio plot
        self.plot_policy_usage_ratio(config_data)
        
        # 3. Switching ratio plot
        self.plot_switching_ratio(config_data)
        
        print("Orchestrator analysis complete!")

    def plot_step_latency(self, config_data: Dict[str, Dict[str, Dict[str, Any]]]):
        """
        Plot step latency for different configurations using box plots.
        
        Args:
            config_data: Dictionary containing configuration data
        """
        plt.figure(figsize=(12, 8))
        
        config_names = []
        all_step_latencies_per_config = []
        
        for config_name, policies_data in config_data.items():
            all_step_latencies = []
            
            for policy_name, data in policies_data.items():
                trajectories = data['trajectories']
                
                for episode_trajectory in trajectories:
                    for step in episode_trajectory:
                        if "latency" in step:
                            all_step_latencies.append(step["latency"])
            
            if all_step_latencies:
                config_names.append(config_name)
                all_step_latencies_per_config.append(all_step_latencies)
                print(f"  {config_name}: {len(all_step_latencies)} steps, Avg latency: {np.mean(all_step_latencies):.3f}s")
            else:
                print(f"  Warning: No steps with latency found for {config_name}")
        
        if not all_step_latencies_per_config:
            print("No valid latency data found!")
            return
        
        # Create box plot
        box_plot = plt.boxplot(all_step_latencies_per_config, 
                              labels=config_names,
                              patch_artist=True,
                              boxprops=dict(facecolor='skyblue', alpha=0.7),
                              medianprops=dict(color='navy', linewidth=2),
                              flierprops=dict(marker='o', markerfacecolor='red', markersize=6),
                              whiskerprops=dict(color='navy', linewidth=1.5),
                              capprops=dict(color='navy', linewidth=1.5))
        
        # Customize the plot
        plt.ylabel('Step Latency (seconds)', fontsize=16, fontweight='bold')
        plt.title('Step Latency Distribution', fontsize=18, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Set y-axis limit to show outliers properly
        all_values = [val for sublist in all_step_latencies_per_config for val in sublist]
        if all_values:
            plt.ylim(0, max(all_values) * 1.1)
        
        plt.tight_layout()
        
        # Save the plot
        save_path = os.path.join(self.output_dir, "step_latency.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Step latency box plot saved to: {save_path}")
        
        plt.show()

    def plot_policy_usage_ratio(self, config_data: Dict[str, Dict[str, Dict[str, Any]]]):
        """
        Plot policy usage ratio for different configurations.
        
        Args:
            config_data: Dictionary containing configuration data
        """
        plt.figure(figsize=(14, 8))
        
        config_names = []
        policy_usage_data = {}
        
        # First pass: collect all unique policy names from policy_selections data
        all_policy_names = set()
        for config_name, policies_data in config_data.items():
            for policy_name, data in policies_data.items():
                evaluation_results = data['evaluation_results']
                
                if 'policy_selections' in evaluation_results:
                    policy_selections = evaluation_results['policy_selections']
                    
                    for episode_selections in policy_selections:
                        for step_selection in episode_selections:
                            all_policy_names.add(step_selection)
        
        # Sort policy names for consistent ordering
        all_policy_names = sorted(list(all_policy_names))
        print(f"Found policy names from data: {all_policy_names}")
        
        for config_name, policies_data in config_data.items():
            config_names.append(config_name)
            
            # Store usage ratios for each episode
            episode_usage_ratios = []
            
            # For each configuration, we need to look at the policy_selections from evaluation_results
            for policy_name, data in policies_data.items():
                evaluation_results = data['evaluation_results']
                
                if 'policy_selections' in evaluation_results:
                    policy_selections = evaluation_results['policy_selections']
                    
                    for episode_selections in policy_selections:
                        # Calculate usage for this episode
                        episode_usage = {policy: 0 for policy in all_policy_names}
                        episode_steps = len(episode_selections)
                        
                        for step_selection in episode_selections:
                            if step_selection in all_policy_names:
                                episode_usage[step_selection] += 1
                        
                        # Calculate ratios for this episode
                        if episode_steps > 0:
                            episode_ratios = {policy: episode_usage[policy] / episode_steps for policy in all_policy_names}
                            episode_usage_ratios.append(episode_ratios)
                else:
                    print(f"  Warning: No 'policy_selections' found in evaluation_results for {config_name}")
            
            # Average usage ratios over all episodes
            if episode_usage_ratios:
                avg_usage = {}
                for policy in all_policy_names:
                    policy_ratios = [episode_ratios[policy] for episode_ratios in episode_usage_ratios]
                    avg_usage[policy] = np.mean(policy_ratios)
                
                policy_usage_data[config_name] = avg_usage
                print(f"  {config_name}: {len(episode_usage_ratios)} episodes, Avg usage: {avg_usage}")
            else:
                # No episodes found, set all to 0
                policy_usage_data[config_name] = {policy: 0.0 for policy in all_policy_names}
                print(f"  Warning: No episodes found for {config_name}")
        
        # Create stacked bar chart
        x_pos = np.arange(len(config_names))
        bottom = np.zeros(len(config_names))
        
        # Define colors for different policies
        colors = plt.cm.Set3(np.linspace(0, 1, len(all_policy_names)))
        
        for i, policy_name in enumerate(all_policy_names):
            values = [policy_usage_data[config][policy_name] for config in config_names]
            plt.bar(x_pos, values, bottom=bottom, label=policy_name, 
                   color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)
            bottom += values
        
        # Customize the plot
        # plt.xlabel('Configuration', fontsize=16, fontweight='bold')
        plt.ylabel('Policy Usage Ratio (%)', fontsize=16, fontweight='bold')
        plt.title('Policy Usage Ratio', fontsize=18, fontweight='bold')
        plt.xticks(x_pos, config_names, rotation=45, ha='right')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(True, alpha=0.3, axis='y')
        plt.ylim(0, 1.0)
        
        # Convert y-axis to percentage
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0f}%'.format(y * 100)))
        
        plt.tight_layout()
        
        # Save the plot
        save_path = os.path.join(self.output_dir, "policy_usage_ratio.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Policy usage ratio plot saved to: {save_path}")
        
        plt.show()

    def plot_switching_ratio(self, config_data: Dict[str, Dict[str, Dict[str, Any]]]):
        """
        Plot policy switching ratio for different configurations using box plots.
        
        Args:
            config_data: Dictionary containing configuration data
        """
        plt.figure(figsize=(12, 8))
        
        config_names = []
        all_episode_ratios = []
        
        for config_name, policies_data in config_data.items():
            episode_ratios = []
            
            for policy_name, data in policies_data.items():
                evaluation_results = data['evaluation_results']
                
                if 'policy_selections' in evaluation_results:
                    policy_selections = evaluation_results['policy_selections']
                    
                    for episode_selections in policy_selections:
                        if len(episode_selections) > 1:
                            episode_switches = 0
                            for i in range(1, len(episode_selections)):
                                if episode_selections[i] != episode_selections[i-1]:
                                    episode_switches += 1
                            
                            # Calculate ratio for this episode
                            episode_ratio = episode_switches / len(episode_selections)
                            episode_ratios.append(episode_ratio)
                        else:
                            # Single step episode has no switches
                            episode_ratios.append(0.0)
                else:
                    print(f"  Warning: No 'policy_selections' found in evaluation_results for {config_name}")
            
            if episode_ratios:
                config_names.append(config_name)
                all_episode_ratios.append(episode_ratios)
                print(f"  {config_name}: {len(episode_ratios)} episodes, Avg switching ratio: {np.mean(episode_ratios):.3f}")
            else:
                print(f"  Warning: No episodes found for {config_name}")
        
        if not all_episode_ratios:
            print("No valid switching ratio data found!")
            return
        
        # Create box plot
        box_plot = plt.boxplot(all_episode_ratios, 
                              labels=config_names,
                              patch_artist=True,
                              boxprops=dict(facecolor='lightcoral', alpha=0.7),
                              medianprops=dict(color='darkred', linewidth=2),
                              flierprops=dict(marker='o', markerfacecolor='red', markersize=6),
                              whiskerprops=dict(color='darkred', linewidth=1.5),
                              capprops=dict(color='darkred', linewidth=1.5))
        
        # Customize the plot
        plt.ylabel('Policy Switching Ratio', fontsize=16, fontweight='bold')
        plt.title('Policy Switching Ratio Distribution', fontsize=18, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.ylim(0, 1.0)  # Switching ratio is always between 0 and 1
        
        # Convert y-axis to percentage
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0f}%'.format(y * 100)))
        
        plt.tight_layout()
        
        # Save the plot
        save_path = os.path.join(self.output_dir, "policy_switching_ratio.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Policy switching ratio box plot saved to: {save_path}")
        
        plt.show()




def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Visualize multi-objective policy results")
    
    parser.add_argument("--config_file", type=str, default="policy_comparison_RQ1_a2c.json", help="JSON file containing policy configurations")
    parser.add_argument("--output_dir", type=str, default="visualizations", help="Output directory for plots")
    parser.add_argument("--max_steps", type=int, default=50, help="Maximum steps for mastery progression")
    parser.add_argument("--combined_plots", action="store_true", default = False, help="Generate combined plots instead of separate plots")
    parser.add_argument("--all_orchestrator_analysis", action="store_true", default = False, help="Generate orchestrator-specific analysis plots (latency, policy usage, switching)")
    parser.add_argument("--calculate_cost", action="store_true", default = False, help="Calculate and save token usage analysis only (no plots)")
    parser.add_argument("--correlation_scatter_plot", action="store_true", default = False, help="Generate correlation scatter plot (aptitude vs gap rewards per episode)")
    parser.add_argument("--optimized_obj", type=str, default="performance", choices=["performance", "gap", "aptitude"], help="Objective to exclude from correlation plot (x and y axes will be the other two objectives)")
    parser.add_argument("--flow_zone_plot", action="store_true", default = False, help="Generate flow zone plot (mastery level vs difficulty level for one episode)")
    
    # Allow individual policy specification
    parser.add_argument("--policy_name", type=str, action="append", help="Policy name")
    parser.add_argument("--result_folder", type=str, action="append", help="Result folder path")
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = PolicyResultsVisualizer(args.output_dir)
    
    # Load policy configurations
    policy_configs = {}
    
    if args.config_file and os.path.exists(args.config_file):
        # Load from JSON config file
        with open(args.config_file, 'r') as f:
            loaded_configs = json.load(f)
            
        # Check if it's a multi-config format or single-config format
        if isinstance(loaded_configs, dict):
            # Check if the first value is also a dict (multi-config format)
            first_value = next(iter(loaded_configs.values()))
            if isinstance(first_value, dict):
                # Multi-config format: {"Config1": {"Policy1": "path1", ...}, "Config2": {...}}
                print("Detected multi-configuration format. Processing each configuration separately...")
                if args.calculate_cost:
                    visualizer.calculate_cost_only(loaded_configs)
                elif args.all_orchestrator_analysis:
                    visualizer.analyze_orchestrator_performance(loaded_configs)
                else:
                    visualizer.visualize_multiple_configs(loaded_configs, args.max_steps, args.combined_plots, args.correlation_scatter_plot, args.optimized_obj, args.flow_zone_plot)
            else:
                # Single-config format: {"Policy1": "path1", "Policy2": "path2"}
                print("Detected single configuration format. Processing as single configuration...")
                if args.calculate_cost:
                    # Convert to multi-config format for cost calculation
                    multi_config = {"Single_Config": loaded_configs}
                    visualizer.calculate_cost_only(multi_config)
                elif args.all_orchestrator_analysis:
                    # Convert to multi-config format for orchestrator analysis
                    multi_config = {"Single_Config": loaded_configs}
                    visualizer.analyze_orchestrator_performance(multi_config)
                else:
                    visualizer.visualize_policies(loaded_configs, args.max_steps, "Single_Config", args.correlation_scatter_plot, args.optimized_obj, args.flow_zone_plot)
        else:
            print("Error: Invalid configuration file format")
            return
            
    elif args.policy_name and args.result_folder:
        # Load from command line arguments
        if len(args.policy_name) != len(args.result_folder):
            print("Error: Number of policy names must match number of result folders")
            return
        
        policy_configs = dict(zip(args.policy_name, args.result_folder))
        if args.calculate_cost:
            # Convert to multi-config format for cost calculation
            multi_config = {"Command_Line_Config": policy_configs}
            visualizer.calculate_cost_only(multi_config)
        elif args.all_orchestrator_analysis:
            # Convert to multi-config format for orchestrator analysis
            multi_config = {"Command_Line_Config": policy_configs}
            visualizer.analyze_orchestrator_performance(multi_config)
        else:
            visualizer.visualize_policies(policy_configs, args.max_steps, "Command_Line_Config", args.correlation_scatter_plot, args.optimized_obj, args.flow_zone_plot)
    else:
        # Example configuration for testing
        print("No configuration provided. Using example configuration...")
        print("\nExample JSON format for multiple configurations:")
        print('''
{
  "Experiment_1": {
    "Policy_A": "results/policy_a",
    "Policy_B": "results/policy_b"
  },
  "Experiment_2": {
    "Policy_C": "results/policy_c",
    "Policy_D": "results/policy_d"
  }
}
        ''')
        
        # Create example config for testing
        policy_configs = {
            "Example_Config": {
                "Example_Policy_1": "results/example_policy_1",
                "Example_Policy_2": "results/example_policy_2",
            }
        }
        print("To use your own data, provide --config_file or --policy_name and --result_folder arguments")
        if args.calculate_cost:
            visualizer.calculate_cost_only(policy_configs)
        elif args.all_orchestrator_analysis:
            visualizer.analyze_orchestrator_performance(policy_configs)
        else:
            visualizer.visualize_multiple_configs(policy_configs, args.max_steps, args.combined_plots, args.correlation_scatter_plot, args.optimized_obj, args.flow_zone_plot)


if __name__ == "__main__":
    main() 