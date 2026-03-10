#!/usr/bin/env python3
"""
Plot mastery heatmaps for all LLM policy folders.

This script reads evaluate_trajectories.json from each folder in base_path,
extracts mastery data, and creates heatmaps where:
- X-axis: Total 30 steps
- Y-axis: Mastery level (0-1)
- Each entry: Mastery value
- Red boxes: Indicate stop actions
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any, Tuple
import re


def extract_mastery_data(trajectory_data: List[List[Dict]]) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Extract mastery data and stop positions from trajectory data.
    
    Args:
        trajectory_data: List of episodes, each episode is a list of steps
        
    Returns:
        Tuple of (mastery_matrix, stop_positions, valid_stops, invalid_stops)
        - mastery_matrix: 2D array of mastery values (episodes x steps)
        - stop_positions: List of (episode, step) tuples where stop=True
        - valid_stops: List of (episode, step) tuples for valid stops (mastery >= 0.8)
        - invalid_stops: List of (episode, step) tuples for invalid stops (mastery < 0.8)
    """
    max_steps = 30
    max_episodes = len(trajectory_data)
    
    # Initialize mastery matrix with NaN (to handle variable episode lengths)
    mastery_matrix = np.full((max_episodes, max_steps), np.nan)
    stop_positions = []
    valid_stops = []
    invalid_stops = []
    
    for episode_idx, episode in enumerate(trajectory_data):
        episode_length = len(episode)
        last_step_idx = min(episode_length - 1, max_steps - 1)
        
        for step_data in episode:
            step = step_data['step']
            if step >= max_steps:
                continue
                
            # Extract mastery value (take the first skill's mastery if multiple)
            mastery_dict = step_data['state']['mastery']
            if mastery_dict:
                # Get the first skill's mastery value
                first_skill = list(mastery_dict.keys())[0]
                mastery_value = mastery_dict[first_skill]
                mastery_matrix[episode_idx, step] = mastery_value
            
            # Check if this is a stop action (last step of episode)
            if step == last_step_idx:
                stop_positions.append((episode_idx, step))
                
                # Determine if it's a valid stop based on mastery
                if mastery_dict:
                    first_skill = list(mastery_dict.keys())[0]
                    mastery_value = mastery_dict[first_skill]
                    
                    if mastery_value >= 0.8:
                        valid_stops.append((episode_idx, step))
                    else:
                        invalid_stops.append((episode_idx, step))
                else:
                    # If no mastery data, consider it invalid
                    invalid_stops.append((episode_idx, step))
    
    return mastery_matrix, stop_positions, valid_stops, invalid_stops


def plot_mastery_heatmap(mastery_matrix: np.ndarray, 
                         stop_positions: List[Tuple[int, int]], 
                         valid_stops: List[Tuple[int, int]],
                         invalid_stops: List[Tuple[int, int]],
                         folder_name: str,
                         output_path: str = None):
    """
    Plot mastery heatmap with stop actions marked as colored boxes.
    
    Args:
        mastery_matrix: 2D array of mastery values
        stop_positions: List of (episode, step) tuples where stop=True
        valid_stops: List of (episode, step) tuples for valid stops (mastery >= 0.8)
        invalid_stops: List of (episode, step) tuples for invalid stops (mastery < 0.8)
        folder_name: Name of the folder for the plot title
        output_path: Path to save the plot (optional)
    """
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create heatmap using seaborn
    # Replace NaN with a special value for visualization
    plot_matrix = mastery_matrix.copy()
    plot_matrix = np.where(np.isnan(plot_matrix), -0.1, plot_matrix)
    
    # Create custom colormap
    colors = ['#f0f0f0', '#e6f3ff', '#b3d9ff', '#80bfff', '#4da6ff', '#1a8cff', '#0066cc', '#004499']
    cmap = sns.color_palette(colors, as_cmap=True)
    
    # Plot heatmap with step labels starting from 1
    sns.heatmap(plot_matrix, 
                cmap=cmap,
                cbar_kws={'label': 'Mastery Level'},
                xticklabels=range(1, 31),  # Start from 1 instead of 0
                yticklabels=range(1, len(mastery_matrix) + 1),
                vmin=0, vmax=1,
                mask=plot_matrix == -0.1)  # Mask NaN values
    
    # Mark valid stops with red boxes
    for episode_idx, step_idx in valid_stops:
        if episode_idx < len(mastery_matrix) and step_idx < 30:
            # Draw red rectangle around the valid stop position
            rect = plt.Rectangle((step_idx, episode_idx), 1, 1, 
                               linewidth=2, edgecolor='red', facecolor='none')
            plt.gca().add_patch(rect)
    
    # Mark invalid stops with grey boxes
    for episode_idx, step_idx in invalid_stops:
        if episode_idx < len(mastery_matrix) and step_idx < 30:
            # Draw grey rectangle around the invalid stop position
            rect = plt.Rectangle((step_idx, episode_idx), 1, 1, 
                               linewidth=2, edgecolor='grey', facecolor='none')
            plt.gca().add_patch(rect)
    
    # Customize plot
    plt.title(f'Mastery Progression Heatmap\n{folder_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Episode', fontsize=12)
    
    # Add legend for stop actions with statistics
    from matplotlib.patches import Rectangle
    total_episodes = len(mastery_matrix)
    valid_count = len(valid_stops)
    
    legend_elements = [
        Rectangle((0, 0), 1, 1, linewidth=2, edgecolor='red', facecolor='none', 
                 label=f'Valid Stop ({valid_count}/{total_episodes})'),
        Rectangle((0, 0), 1, 1, linewidth=2, edgecolor='grey', facecolor='none', 
                 label='Invalid Stop')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    # Save plot if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    plt.show()


def process_llm_policy_folders(base_path: str = "Policy_Set_Results_IRT") -> None:
    """
    Process all folders in base_path and create mastery heatmaps.
    
    Args:
        base_path: Path to the base directory
    """
    base_path = Path(base_path)
    
    if not base_path.exists():
        print(f"Error: {base_path} does not exist!")
        return
    
    # Get all subdirectories
    policy_folders = [d for d in base_path.iterdir() if d.is_dir()]
    
    if not policy_folders:
        print(f"No policy folders found in {base_path}")
        return
    
    print(f"Found {len(policy_folders)} policy folders")
    
    # Create output directory for plots
    output_dir = Path("Policy_Set_Results_IRT")
    output_dir.mkdir(exist_ok=True)
    
    for folder in policy_folders:
        folder_name = folder.name
        trajectories_file = folder / "evaluation_trajectories.json"
        
        if not trajectories_file.exists():
            print(f"Warning: {trajectories_file} not found in {folder_name}")
            continue
        
        print(f"\nProcessing: {folder_name}")
        
        try:
            # Read trajectory data
            with open(trajectories_file, 'r') as f:
                trajectory_data = json.load(f)
            
            # Extract mastery data
            mastery_matrix, stop_positions, valid_stops, invalid_stops = extract_mastery_data(trajectory_data)
            
            print(f"  - Episodes: {len(mastery_matrix)}")
            print(f"  - Total stops: {len(stop_positions)}")
            print(f"  - Valid stops (mastery >= 0.8): {len(valid_stops)}")
            print(f"  - Invalid stops (mastery < 0.8): {len(invalid_stops)}")
            
            # Create output filename
            safe_name = re.sub(r'[^\w\-_]', '_', folder_name)
            output_file = output_dir / f"{safe_name}_mastery_heatmap.png"
            
            # Plot heatmap
            plot_mastery_heatmap(mastery_matrix, stop_positions, valid_stops, invalid_stops, folder_name, str(output_file))
            
        except Exception as e:
            print(f"  Error processing {folder_name}: {e}")
            continue
    
    print(f"\nAll plots saved to: {output_dir}")


def main():
    """Main function to run the script."""
    print("LLM Policy Mastery Heatmap Generator")
    print("=" * 40)
    
    # Process all LLM policy folders
    process_llm_policy_folders()
    
    print("\nScript completed!")


if __name__ == "__main__":
    main() 