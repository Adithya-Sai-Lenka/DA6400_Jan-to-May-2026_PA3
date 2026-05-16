import numpy as np
import matplotlib.pyplot as plt
import os

def plot_sac_vs_pebble(sac_file='sac_pendulum_automated_temp_tuning_eval_results.npy', 
                       pebble_file='pebble_results_combined.npy'):
    
    print("Loading datasets...")
    try:
        data_sac = np.load(sac_file, allow_pickle=True).item()
        data_pebble = np.load(pebble_file, allow_pickle=True).item()
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        print("Please ensure both 'sac_pendulumn_automated_temp_tuning_eval_results.npy' and 'pebble_results_combined_latest.npy' are in the current directory.")
        return

    # Extract target angles present in the PEBBLE dataset
    angles = sorted(data_pebble.keys())
    
    if len(angles) == 0:
        print("No PEBBLE data found.")
        return
        
    budgets = sorted(data_pebble[angles[0]].keys())
    
    # Setup the Subplot Grid (2 rows x 3 columns for 5 angles)
    rows, cols = 2, 3
    fig, axes = plt.subplots(rows, cols, figsize=(18, 10), sharex=True, sharey=False)
    axes = axes.flatten() 
    
    # Hide the 6th empty subplot
    axes[5].axis('off')
    
    # Colors for different PEBBLE budgets
    colors = plt.cm.tab10.colors

    for idx, angle in enumerate(angles):
        ax = axes[idx]
        ax.set_title(f"Target Angle: {angle}°", fontsize=15, fontweight='bold', pad=10)
        
        # ==========================================
        # 1. Plot SAC Ground-Truth Baseline
        # ==========================================
        if angle in data_sac:
            seeds_data_sac = data_sac[angle]
            first_seed_sac = list(seeds_data_sac.keys())[0]
            steps_sac = [metric[0] for metric in seeds_data_sac[first_seed_sac]]
            
            # Aggregate returns
            returns_matrix_sac = np.array([[m[1] for m in metrics] for metrics in seeds_data_sac.values()])
            n_seeds_sac = returns_matrix_sac.shape[0]
            
            mean_returns_sac = np.mean(returns_matrix_sac, axis=0)
            ci_sac = 1.96 * (np.std(returns_matrix_sac, axis=0) / np.sqrt(n_seeds_sac))
            
            # Thick, dotted black line for Ground Truth SAC
            ax.plot(steps_sac, mean_returns_sac, label="SAC (Ground Truth)", color='black', linewidth=3.0, linestyle=':')
            ax.fill_between(steps_sac, mean_returns_sac - ci_sac, mean_returns_sac + ci_sac, color='black', alpha=0.1)

        # ==========================================
        # 2. Plot PEBBLE (Different Budgets)
        # ==========================================
        for c_idx, budget in enumerate(budgets):
            seeds_data_peb = data_pebble[angle][budget]
            
            if not seeds_data_peb: continue
            
            first_seed_peb = list(seeds_data_peb.keys())[0]
            steps_peb = [metric[0] for metric in seeds_data_peb[first_seed_peb]]
            
            returns_matrix_peb = np.array([[m[1] for m in metrics] for metrics in seeds_data_peb.values()])
            n_seeds_peb = returns_matrix_peb.shape[0]
            
            mean_returns_peb = np.mean(returns_matrix_peb, axis=0)
            ci_peb = 1.96 * (np.std(returns_matrix_peb, axis=0) / np.sqrt(n_seeds_peb))
            
            color = colors[c_idx % len(colors)]
            ax.plot(steps_peb, mean_returns_peb, label=f"PEBBLE (Budget: {budget})", color=color, linewidth=2.0)
            ax.fill_between(steps_peb, mean_returns_peb - ci_peb, mean_returns_peb + ci_peb, color=color, alpha=0.15)

        # ==========================================
        # 3. Formatting
        # ==========================================
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(axis='both', which='major', labelsize=11)
                
        # Add labels to edge plots
        if idx >= 2:
            ax.set_xlabel('Environment Steps', fontsize=13)
        if idx % cols == 0:
            ax.set_ylabel('True Unscaled Return', fontsize=13)
            
    # ==========================================
    # 4. Global Legend
    # ==========================================
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, 
               loc='lower center', 
               ncol=len(budgets) + 1, 
               fontsize=13, 
               title='Algorithm & Feedback Budget', 
               title_fontsize=14, 
               bbox_to_anchor=(0.5, -0.05)) 
    
    os.makedirs('plots/bonus', exist_ok=True)
    plt.tight_layout()
    plt.savefig('plots/bonus/sac_vs_pebble_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Plot saved successfully as 'plots/bonus/sac_vs_pebble_comparison.png'")
    plt.show()

if __name__ == "__main__":
    plot_sac_vs_pebble()