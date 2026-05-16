import numpy as np
import matplotlib.pyplot as plt

import os

def plot_sac_results(filename='sac_pendulumn_automated_temp_tuning_eval_results.npy'):
    # 1. Load the dictionary from the .npy file
    try:
        # .item() extracts the dictionary from the 0-d numpy array
        data = np.load(filename, allow_pickle=True).item()
    except FileNotFoundError:
        print(f"Error: Could not find '{filename}'. Check the exact filename.")
        return

    # Set up the plot
    plt.figure(figsize=(10, 6))
    
    # Define a color map to easily distinguish 8 different targets
    colors = plt.cm.tab10.colors 

    # 2. Iterate through each target angle in the data
    for idx, (angle, seeds_data) in enumerate(sorted(data.items())):
        # Extract the timesteps (x-axis) from the first seed
        first_seed = list(seeds_data.keys())[0]
        steps = [metric[0] for metric in seeds_data[first_seed]]
        
        # Accumulate the returns for all 15 seeds into a matrix
        # Shape will be: (num_seeds, num_steps)
        returns_matrix = []
        for seed, metrics in seeds_data.items():
            returns = [metric[1] for metric in metrics]
            returns_matrix.append(returns)
            
        returns_matrix = np.array(returns_matrix)
        n_seeds = returns_matrix.shape[0]
        
        # 3. Calculate Mean and 95% Confidence Interval
        mean_returns = np.mean(returns_matrix, axis=0)
        std_returns = np.std(returns_matrix, axis=0)
        
        # 95% CI formula: 1.96 * (std / sqrt(n))
        ci = 1.96 * (std_returns / np.sqrt(n_seeds))
        
        # 4. Plot Mean and shade the Confidence Interval
        color = colors[idx % len(colors)]
        plt.plot(steps, mean_returns, label=f"Target: {angle}°", color=color, linewidth=2)
        plt.fill_between(steps, 
                         mean_returns - ci, 
                         mean_returns + ci, 
                         color=color, 
                         alpha=0.15) # Alpha controls the transparency of the shading

    # 5. Formatting the graph
    plt.title('SAC Offline Evaluation Returns (15 Seeds, 95% CI)', fontsize=16, pad=15)
    plt.xlabel('Environment Steps', fontsize=14)
    plt.ylabel('Average Return', fontsize=14)
    
    # Place legend outside the plot if it overlaps with lines, or inside if space permits
    plt.legend(title='Target Angles', fontsize=11, title_fontsize=12, loc='lower right')
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # Save and display
    os.makedirs('plots/pendulum', exist_ok=True)
    plt.savefig('plots/pendulum/automated_temp_tuning_eval.png', dpi=300)
    print("Plot saved successfully as 'plots/pendulum/automated_temp_tuning_eval.png'")
    plt.show()

def plot_temp_tuning_comparison_2x2(manual_filename, auto_filename):
    # 1. Load the nested dictionaries from the .npy files
    try:
        data_manual = np.load(manual_filename, allow_pickle=True).item()
    except FileNotFoundError:
        print(f"Error: Could not find '{manual_filename}'.")
        return

    try:
        data_auto = np.load(auto_filename, allow_pickle=True).item()
    except FileNotFoundError:
        print(f"Error: Could not find '{auto_filename}'.")
        return

    # Extract sorted angles to ensure consistent plotting order
    angles = sorted(data_manual.keys())
    angles = angles[:4]  # Take the first 4 for the 2x2 grid
    
    if len(angles) == 0:
        print("No data found.")
        return
        
    alphas = sorted(data_manual[angles[0]].keys())
    
    # 2. Setup the Subplot Grid (HD 2x2)
    rows, cols = 2, 2
    fig, axes = plt.subplots(rows, cols, figsize=(16, 12), sharex=True, dpi=150)
    axes = axes.flatten() 
    
    # Use a colormap for manual alphas
    colors = plt.cm.tab10.colors

    # 3. Iterate through each subplot (target angle)
    for idx, angle in enumerate(angles):
        ax = axes[idx]
        ax.set_title(f"Target Angle: {angle}°", fontsize=18, fontweight='bold', pad=12)
        
        # --- PLOT MANUAL TEMP TUNING (Solid Lines) ---
        for c_idx, alpha in enumerate(alphas):
            seeds_data = data_manual[angle][alpha]
            
            first_seed = list(seeds_data.keys())[0]
            steps = [metric[0] for metric in seeds_data[first_seed]]
            
            returns_matrix = []
            for seed, metrics in seeds_data.items():
                returns = [metric[1] for metric in metrics]
                returns_matrix.append(returns)
                
            returns_matrix = np.array(returns_matrix)
            n_seeds = returns_matrix.shape[0]
            
            mean_returns = np.mean(returns_matrix, axis=0)
            std_returns = np.std(returns_matrix, axis=0)
            ci = 1.96 * (std_returns / np.sqrt(n_seeds))
            
            color = colors[c_idx % len(colors)]
            ax.plot(steps, mean_returns, label=f"Manual α = {alpha}", color=color, linewidth=2.0, alpha=0.8)
            ax.fill_between(steps, mean_returns - ci, mean_returns + ci, color=color, alpha=0.1)

        # --- PLOT AUTOMATED TEMP TUNING (Dotted Line) ---
        if angle in data_auto:
            # Handle standard nested structures depending on how auto data was saved
            auto_angle_data = data_auto[angle]
            if isinstance(auto_angle_data, dict):
                # If there's an intermediate key (e.g., data['auto'])
                first_key = list(auto_angle_data.keys())[0]
                if isinstance(auto_angle_data[first_key], dict):
                    seeds_data_auto = auto_angle_data[first_key]
                else:
                    # If it's mapped directly to seeds
                    seeds_data_auto = auto_angle_data
            
            first_seed_auto = list(seeds_data_auto.keys())[0]
            steps_auto = [metric[0] for metric in seeds_data_auto[first_seed_auto]]
            
            returns_matrix_auto = []
            for seed, metrics in seeds_data_auto.items():
                returns = [metric[1] for metric in metrics]
                returns_matrix_auto.append(returns)
                
            returns_matrix_auto = np.array(returns_matrix_auto)
            n_seeds_auto = returns_matrix_auto.shape[0]
            
            mean_returns_auto = np.mean(returns_matrix_auto, axis=0)
            std_returns_auto = np.std(returns_matrix_auto, axis=0)
            ci_auto = 1.96 * (std_returns_auto / np.sqrt(n_seeds_auto))
            
            # Plot Automated tuning with a distinct thick dotted line
            ax.plot(steps_auto, mean_returns_auto, label="Automated Tuning", color='black', 
                    linewidth=3.5, linestyle=':')
            ax.fill_between(steps_auto, mean_returns_auto - ci_auto, mean_returns_auto + ci_auto, 
                            color='black', alpha=0.15)

        # 5. Formatting each subplot
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        # Add X labels to the bottom row
        if idx >= 2:
            ax.set_xlabel('Environment Steps', fontsize=16)
            
        # Add Y labels to the leftmost column
        if idx % 2 == 0:
            ax.set_ylabel('Average Return', fontsize=16)
            
    # 6. Global Legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, 
               loc='lower center', 
               ncol=len(alphas) + 1, # +1 to fit the Automated label neatly in the row
               fontsize=14, 
               title='SAC Temperature Strategy', 
               title_fontsize=16, 
               bbox_to_anchor=(0.5, -0.06))
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save with Ultra-High Resolution (600 DPI)
    os.makedirs('plots/pendulum', exist_ok=True)
    out_path = 'plots/pendulum/temp_tuning_comparison_2x2.png'
    plt.savefig(out_path, dpi=600, bbox_inches='tight')
    print(f"Plot saved successfully as '{out_path}' at 600 DPI")
    plt.show()

def plot_scaled_rewards_comparison(filename='sac_pendulum_scaled_reward_results.npy'):
    """
    Assumed data structure in the .npy file:
    data = {
        10.0: {
            'manual': {seed_0: [(step, return), ...], seed_1: [...]},
            'auto':   {seed_0: [(step, return), ...], seed_1: [...]}
        },
        0.1: {
            'manual': {...},
            'auto':   {...}
        }
    }
    """
    try:
        data = np.load(filename, allow_pickle=True).item()
    except FileNotFoundError:
        print(f"Error: Could not find '{filename}'.")
        return

    # Filter for the specific scales requested
    scales = [10.0, 0.1]
    
    # Check if scales exist in data (handling potential string keys)
    available_scales = []
    for scale in scales:
        if scale in data:
            available_scales.append(scale)
        elif str(scale) in data:
            available_scales.append(str(scale))
            
    if not available_scales:
        print("Error: Could not find data for scales 10.0 and 0.1 in the .npy file.")
        print(f"Available keys: {list(data.keys())}")
        return

    # Setup the Subplot Grid (1x2 for the two scales)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=150)
    
    strategies = ['constant', 'auto']
    colors = {'constant': '#e74c3c', 'auto': '#2c3e50'} # Red for constant, Dark Blue/Black for auto
    linestyles = {'constant': '-', 'auto': ':'}

    for idx, scale in enumerate(available_scales):
        ax = axes[idx]
        ax.set_title(f"Reward Scale: {scale}×", fontsize=18, fontweight='bold', pad=12)
        
        scale_data = data[scale]
        
        for strategy in strategies:
            if strategy not in scale_data:
                continue
                
            seeds_data = scale_data[strategy]
            
            # Extract steps from the first seed
            first_seed = list(seeds_data.keys())[0]
            steps = [metric[0] for metric in seeds_data[first_seed]]
            
            # Aggregate returns (Remember to un-scale the plotted returns to compare against true performance)
            # If your saved returns are already un-scaled, remove the division by float(scale)
            returns_matrix = []
            for seed, metrics in seeds_data.items():
                returns = [metric[1] / float(scale) for metric in metrics] 
                returns_matrix.append(returns)
                
            returns_matrix = np.array(returns_matrix)
            n_seeds = returns_matrix.shape[0]
            
            mean_returns = np.mean(returns_matrix, axis=0)
            std_returns = np.std(returns_matrix, axis=0)
            ci = 1.96 * (std_returns / np.sqrt(n_seeds))
            
            label_name = "Automated Tuning" if strategy == 'auto' else r"Manual Tuning ($\alpha_{mnl}$)"
            linewidth = 3.5 if strategy == 'auto' else 2.5
            
            ax.plot(steps, mean_returns, label=label_name, 
                    color=colors[strategy], linestyle=linestyles[strategy], linewidth=linewidth)
            ax.fill_between(steps, mean_returns - ci, mean_returns + ci, 
                            color=colors[strategy], alpha=0.15)

        # Formatting
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlabel('Environment Steps', fontsize=16)
        
        if idx == 0:
            ax.set_ylabel('True Average Return', fontsize=16)

    # Global Legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=15, 
               title='SAC Temperature Strategy', title_fontsize=16, bbox_to_anchor=(0.5, -0.1))
    
    plt.tight_layout()
    
    # Save High-Res
    os.makedirs('plots/pendulum', exist_ok=True)
    out_path = 'plots/pendulum/sac_scaled_rewards_comparison.png'
    plt.savefig(out_path, dpi=600, bbox_inches='tight')
    print(f"Plot saved successfully as '{out_path}' at 600 DPI")
    plt.show()

if __name__ == "__main__":

    plot_sac_results('sac_pendulum_automated_temp_tuning_eval_results.npy')

    MANUAL_FILE = 'sac_pendulum_manual_temp_tuning_eval_results.npy'
    AUTO_FILE = 'sac_pendulum_automated_temp_tuning_eval_results.npy' 
    plot_temp_tuning_comparison_2x2(MANUAL_FILE, AUTO_FILE)
    
    plot_scaled_rewards_comparison()
