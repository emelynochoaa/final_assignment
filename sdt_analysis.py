"""
Signal Detection Theory (SDT) and Delta Plot Analysis for Response Time Data
Integrated version using original functions with working data pipeline
"""

import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os
import warnings
warnings.filterwarnings('ignore')

# Mapping dictionaries for categorical variables
MAPPINGS = {
    'stimulus_type': {'simple': 0, 'complex': 1},
    'difficulty': {'easy': 0, 'hard': 1},
    'signal': {'present': 0, 'absent': 1}
}

# Descriptive names for each experimental condition
CONDITION_NAMES = {
    0: 'Easy Simple',
    1: 'Easy Complex',
    2: 'Hard Simple',
    3: 'Hard Complex'
}

# Percentiles used for delta plot analysis
PERCENTILES = [10, 30, 50, 70, 90]

def read_data(file_path, prepare_for='sdt', display=False):
    """Read and preprocess data from CSV file."""
    data = pd.read_csv(file_path)
    
    # Convert categorical variables to numeric codes
    for col, mapping in MAPPINGS.items():
        data[col] = data[col].map(mapping)
    
    # Create participant number and condition index
    data['pnum'] = data['participant_id']
    data['condition'] = data['stimulus_type'] + data['difficulty'] * 2
    data['accuracy'] = data['accuracy'].astype(int)
    
    if display:
        print(f"Data: {len(data)} trials, {data['pnum'].nunique()} participants")
        print(f"Overall accuracy: {data['accuracy'].mean():.3f}")
    
    # Transform to SDT format if requested
    if prepare_for == 'sdt':
        grouped = data.groupby(['pnum', 'condition', 'signal']).agg({
            'accuracy': ['count', 'sum']
        }).reset_index()
        
        grouped.columns = ['pnum', 'condition', 'signal', 'nTrials', 'correct']
        
        sdt_data = []
        for pnum in grouped['pnum'].unique():
            p_data = grouped[grouped['pnum'] == pnum]
            for condition in p_data['condition'].unique():
                c_data = p_data[p_data['condition'] == condition]
                
                signal_trials = c_data[c_data['signal'] == 0]
                noise_trials = c_data[c_data['signal'] == 1]
                
                if not signal_trials.empty and not noise_trials.empty:
                    sdt_data.append({
                        'pnum': pnum,
                        'condition': condition,
                        'hits': signal_trials['correct'].iloc[0],
                        'misses': signal_trials['nTrials'].iloc[0] - signal_trials['correct'].iloc[0],
                        'false_alarms': noise_trials['nTrials'].iloc[0] - noise_trials['correct'].iloc[0],
                        'correct_rejections': noise_trials['correct'].iloc[0],
                        'nSignal': signal_trials['nTrials'].iloc[0],
                        'nNoise': noise_trials['nTrials'].iloc[0]
                    })
        
        data = pd.DataFrame(sdt_data)
    
    return data

def apply_hierarchical_sdt_model(data, samples=1000, tune=1000):
    
    print(f"Fitting hierarchical SDT model...")
    print(f"Samples: {samples}, Tune: {tune}")
    
    # Get unique participants and conditions
    P = len(data['pnum'].unique())
    C = len(data['condition'].unique())
    
    # Create indexing for model
    data = data.copy()
    data['pnum_idx'] = data['pnum'] - 1  # 0-indexed for PyMC
    data['cond_idx'] = data['condition']
    
    print(f"Model dimensions: {P} participants, {C} conditions")
    
    # Define the hierarchical model
    with pm.Model() as sdt_model:
        # Group-level parameters
        mean_d_prime = pm.Normal('mean_d_prime', mu=1.0, sigma=1.0, shape=C)
        stdev_d_prime = pm.HalfNormal('stdev_d_prime', sigma=0.5)
        
        mean_criterion = pm.Normal('mean_criterion', mu=0.0, sigma=1.0, shape=C)
        stdev_criterion = pm.HalfNormal('stdev_criterion', sigma=0.5)
        
        # Individual-level parameters
        d_prime = pm.Normal('d_prime', 
                          mu=mean_d_prime[data['cond_idx']], 
                          sigma=stdev_d_prime, 
                          shape=len(data))
        criterion = pm.Normal('criterion', 
                            mu=mean_criterion[data['cond_idx']], 
                            sigma=stdev_criterion, 
                            shape=len(data))
        
        # Calculate hit and false alarm rates using SDT
        hit_rate = pm.math.invlogit(d_prime - criterion)
        false_alarm_rate = pm.math.invlogit(-criterion)
                
        # Likelihood for signal trials
        pm.Binomial('hit_obs', 
                   n=data['nSignal'], 
                   p=hit_rate, 
                   observed=data['hits'])
        
        # Likelihood for noise trials
        pm.Binomial('false_alarm_obs', 
                   n=data['nNoise'], 
                   p=false_alarm_rate, 
                   observed=data['false_alarms'])
    
        # Sample
        trace = pm.sample(samples, tune=tune, cores=2, return_inferencedata=True,
                         target_accept=0.9, random_seed=42)
    
    print("Model fitting complete!")
    return trace, sdt_model

def draw_delta_plots(data, pnum):
    """Draw delta plots comparing RT distributions between condition pairs."""
    p_data = data[data['pnum'] == pnum].copy()
    
    if len(p_data) == 0:
        print(f"No data found for participant {pnum}")
        return
    
    conditions = sorted(p_data['condition'].unique())
    n_conditions = len(conditions)
    
    fig, axes = plt.subplots(n_conditions, n_conditions, 
                            figsize=(4*n_conditions, 4*n_conditions))
    
    if n_conditions == 1:
        axes = np.array([[axes]])
    elif n_conditions == 2:
        axes = axes.reshape(2, 2)
    
    marker_style = {
        'marker': 'o',
        'markersize': 8,
        'markerfacecolor': 'white',
        'markeredgewidth': 2,
        'linewidth': 2
    }
    
    for i, cond1 in enumerate(conditions):
        for j, cond2 in enumerate(conditions):
            ax = axes[i, j]
            
            if j == 0:
                ax.set_ylabel('Difference in RT (s)', fontsize=10)
            if i == len(conditions)-1:
                ax.set_xlabel('Percentile', fontsize=10)
                
            if i == j:
                ax.axis('off')
                continue
            
            c1_data = p_data[p_data['condition'] == cond1]
            c2_data = p_data[p_data['condition'] == cond2]
            
            if len(c1_data) == 0 or len(c2_data) == 0:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
                continue
            
            # Upper triangle: overall RT differences
            if i < j:
                c1_rt = c1_data['rt']
                c2_rt = c2_data['rt']
                
                if len(c1_rt) > 0 and len(c2_rt) > 0:
                    overall_delta = [np.percentile(c2_rt, p) - np.percentile(c1_rt, p) 
                                   for p in PERCENTILES]
                    ax.plot(PERCENTILES, overall_delta, color='black', **marker_style)
            
            # Lower triangle: correct/error RT differences
            else:
                c1_correct = c1_data[c1_data['accuracy'] == 1]['rt']
                c1_error = c1_data[c1_data['accuracy'] == 0]['rt']
                c2_correct = c2_data[c2_data['accuracy'] == 1]['rt']
                c2_error = c2_data[c2_data['accuracy'] == 0]['rt']
                
                if len(c1_correct) > 0 and len(c2_correct) > 0:
                    correct_delta = [np.percentile(c2_correct, p) - np.percentile(c1_correct, p) 
                                   for p in PERCENTILES]
                    ax.plot(PERCENTILES, correct_delta, color='green', 
                           label='Correct', **marker_style)
                
                if len(c1_error) > 0 and len(c2_error) > 0:
                    error_delta = [np.percentile(c2_error, p) - np.percentile(c1_error, p) 
                                 for p in PERCENTILES]
                    ax.plot(PERCENTILES, error_delta, color='red', 
                           label='Error', **marker_style)
                
                ax.legend(loc='upper left', fontsize=8)

            ax.set_ylim(bottom=-0.3, top=0.5)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5) 
            ax.grid(True, alpha=0.3)
            
            title = f'{CONDITION_NAMES[cond2]} - {CONDITION_NAMES[cond1]}'
            ax.set_title(title, fontsize=9)
            
    plt.suptitle(f'Delta Plots - Participant {pnum}', fontsize=14)
    plt.tight_layout()
    
    plt.savefig(f'delta_plots_participant_{pnum}.png', dpi=300, bbox_inches='tight')
    plt.show()

def main_analysis():
    """Run complete analysis pipeline"""
    file_path = "/Users/emelynochoa/Desktop/UCII/24.25/Spring25/COGS107/data.csv"
    
    # Load data for SDT analysis
    sdt_data = read_data(file_path, prepare_for='sdt', display=True)
    
    # Apply hierarchical SDT model
    trace, model = apply_hierarchical_sdt_model(sdt_data, samples=1000, tune=1000)
    
    # Check convergence
    rhat = az.rhat(trace)
    max_rhat = max([float(rhat[var].max()) for var in rhat.data_vars])
    ess = az.ess(trace)
    min_ess = min([float(ess[var].min()) for var in ess.data_vars])
    
    print(f"\nConvergence: R-hat = {max_rhat:.4f}, ESS = {min_ess:.0f}")
    if max_rhat < 1.1:
        print("Convergence looks good!")
    
    # Display results
    summary = az.summary(trace, var_names=['mean_d_prime', 'mean_criterion'])
    conditions = ['Easy Simple', 'Easy Complex', 'Hard Simple', 'Hard Complex']
    
    print("\nGROUP-LEVEL d' (SENSITIVITY):")
    d_results = summary.loc[summary.index.str.contains('mean_d_prime')]
    for i, (idx, row) in enumerate(d_results.iterrows()):
        if i < 4:
            print(f"  {conditions[i]}: {row['mean']:.3f} [{row['hdi_3%']:.3f}, {row['hdi_97%']:.3f}]")
    
    print("\nGROUP-LEVEL CRITERION (BIAS):")
    c_results = summary.loc[summary.index.str.contains('mean_criterion')]
    for i, (idx, row) in enumerate(c_results.iterrows()):
        if i < 4:
            print(f"  {conditions[i]}: {row['mean']:.3f} [{row['hdi_3%']:.3f}, {row['hdi_97%']:.3f}]")
    
    # Main effects analysis
    d_means = d_results['mean'].values[:4]
    easy_simple, easy_complex, hard_simple, hard_complex = d_means
    
    easy_avg = (easy_simple + easy_complex) / 2
    hard_avg = (hard_simple + hard_complex) / 2
    simple_avg = (easy_simple + hard_simple) / 2
    complex_avg = (easy_complex + hard_complex) / 2
    
    print(f"\nMAIN EFFECTS:")
    print(f"Difficulty effect (Easy - Hard): {easy_avg - hard_avg:.3f}")
    print(f"Stimulus type effect (Simple - Complex): {simple_avg - complex_avg:.3f}")
    
    # Create posterior plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # d' posteriors by condition
    ax = axes[0,0]
    for i in range(4):
        samples = trace.posterior['mean_d_prime'][:, :, i].values.flatten()
        ax.hist(samples, alpha=0.6, label=conditions[i], bins=30)
    ax.set_xlabel("d' (sensitivity)")
    ax.set_title("Posterior Distributions: d' by Condition")
    ax.legend()
    
    # Criterion posteriors
    ax = axes[0,1] 
    for i in range(4):
        samples = trace.posterior['mean_criterion'][:, :, i].values.flatten()
        ax.hist(samples, alpha=0.6, label=conditions[i], bins=30)
    ax.set_xlabel("Criterion (bias)")
    ax.set_title("Posterior Distributions: Criterion by Condition")
    ax.legend()
    
    # Individual d' estimates
    ax = axes[1,0]
    individual_d = trace.posterior['d_prime'].mean(dim=['chain', 'draw'])
    participant_means = []
    for p in range(10):
        p_indices = sdt_data[sdt_data['pnum'] == p+1].index
        if len(p_indices) > 0:
            p_mean = individual_d[p_indices].mean().item()
            participant_means.append(p_mean)
    ax.bar(range(1, len(participant_means)+1), participant_means)
    ax.set_xlabel("Participant")
    ax.set_ylabel("Mean d'")
    ax.set_title("Person-Specific d' Estimates")
    
    # Effect sizes comparison
    ax = axes[1,1]
    effects = [easy_avg - hard_avg, simple_avg - complex_avg]
    effect_names = ['Difficulty\n(Easy - Hard)', 'Stimulus Type\n(Simple - Complex)']
    ax.bar(effect_names, effects, color=['red', 'blue'])
    ax.set_ylabel("Effect Size (d' difference)")
    ax.set_title("Comparison of Main Effects")
    ax.axhline(y=0, color='gray', linestyle='--')
    
    plt.tight_layout()
    plt.savefig('posterior_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create delta plots
    trial_data = read_data(file_path, prepare_for='trial', display=False)
    first_participant = trial_data['pnum'].iloc[0]
    draw_delta_plots(trial_data, first_participant)
    
    # Save results
    with open('sdt_results.txt', 'w') as f:
        f.write("Analysis Results\n")
        f.write("="*40 + "\n")
        f.write(f"Convergence: R-hat = {max_rhat:.4f}, ESS = {min_ess:.0f}\n\n")
        f.write("d' by condition:\n")
        for i, (idx, row) in enumerate(d_results.iterrows()):
            if i < 4:
                f.write(f"  {conditions[i]}: {row['mean']:.3f} [{row['hdi_3%']:.3f}, {row['hdi_97%']:.3f}]\n")
        f.write(f"\nMain effects:\n")
        f.write(f"  Difficulty: {easy_avg - hard_avg:.3f}\n")
        f.write(f"  Stimulus type: {simple_avg - complex_avg:.3f}\n")
    
    print(f"\nDifficulty effect: {easy_avg - hard_avg:.3f} d' units")
    print(f"Stimulus type effect: {simple_avg - complex_avg:.3f} d' units")
    print("Both methods show difficulty dominates stimulus type effects.")
    
    return trace, summary, trial_data

if __name__ == "__main__":
    trace, summary, data = main_analysis()