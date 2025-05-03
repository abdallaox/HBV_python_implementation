class uncertainty:
    
    def evaluate_uncertainty(self, n_runs=10000, objective='NSE', save_best=10, 
                        plot_results=True, verbose=True, seed=None, narrow_percent=None,
                        exclude_warmup_in_plots=True):
        """
        Perform uncertainty analysis on an HBV model by sampling from parameter ranges.
        
        Parameters:
        -----------
        self : HBVmodel
            The HBV model instance to analyze (typically after calibration)
        n_runs : int, default 10000
            Number of model runs with different parameter sets
        objective : str, default 'NSE'
            Objective function to evaluate model performance. Options are:
            - 'NSE': Nash-Sutcliffe Efficiency (higher is better)
            - 'KGE': Kling-Gupta Efficiency (higher is better)
            - 'RMSE': Root Mean Square Error (lower is better)
            - 'MAE': Mean Absolute Error (lower is better)
        save_best : int, default 10
            Number of best parameter sets to save
        plot_results : bool, default True
            Whether to plot the results after analysis
        verbose : bool, default True
            Whether to print progress information
        seed : int, optional
            Random seed for reproducibility
        narrow_percent : float, optional
            If provided, narrows the parameter search space to this percentage around the default values
        exclude_warmup_in_plots : bool, default True
            Whether to exclude the warmup period from plots
            
        Returns:
        --------
        dict
            Dictionary containing the best parameter sets, their performance metrics,
            and uncertainty statistics
        """
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import time
        import copy
        from tqdm.auto import tqdm
        
        if self.data is None:
            raise ValueError("No data loaded. Use load_data() method first.")
            
        # Check if observed discharge data is available
        if (self.column_names['obs_q'] is None or 
            self.column_names['obs_q'] not in self.data.columns):
            raise ValueError("Observed discharge data is required for uncertainty analysis.")
        
        # Set random seed for reproducibility if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Store the initial states and parameters to restore later
        initial_states = copy.deepcopy(self.states)
        original_params = copy.deepcopy(self.params)
        
        # Extract observed discharge data
        obs_q = self.data[self.column_names['obs_q']].values
        
        # Get valid indices (where obs_q is not NaN)
        valid_idx = ~np.isnan(obs_q)
        if np.sum(valid_idx) == 0:
            raise ValueError("No valid observed discharge values found.")
        
        # Create flat parameter list for sampling
        param_info = []
        for group_name, group in self.params.items():
            for param_name, param_data in group.items():
                param_info.append({
                    'group': group_name,
                    'name': param_name,
                    'min': param_data['min'],
                    'max': param_data['max'],
                    'default': param_data['default']
                })
        # Narrow ranges around the defaults (best fit) if narrow_percent is specified
        if narrow_percent is not None:
            for p in param_info:
                best_val = p['default']
                full_range = p['max'] - p['min']
                delta = full_range * narrow_percent

                new_min = max(p['min'], best_val - delta)
                new_max = min(p['max'], best_val + delta)

                p['min'] = new_min
                p['max'] = new_max
                if verbose:
                    print(f"Narrowed range {p['group']}_{p['name']}: {new_min:.4f} to {new_max:.4f}")        
        
        # Helper function to create parameter dictionary from sampled values
        def create_param_dict(flat_params):
            param_dict = {group_name: {} for group_name in set(p['group'] for p in param_info)}
            
            for i, p in enumerate(param_info):
                if p['name'] not in param_dict[p['group']]:
                    param_dict[p['group']][p['name']] = {}
                
                param_dict[p['group']][p['name']]['min'] = p['min']
                param_dict[p['group']][p['name']]['max'] = p['max']
                param_dict[p['group']][p['name']]['default'] = flat_params[i]
            
            return param_dict
        
        # Evaluate model with a given parameter set
        def evaluate_model(params):
            # Create parameter dictionary
            param_dict = create_param_dict(params)
            
            # Update model parameters
            self.params = param_dict
            
            # Reset states
            self.states = copy.deepcopy(initial_states)
            
            # Run the model
            self.run(verbose=False)  # Setting verbose=False to reduce output clutter
            
            # Calculate performance metrics
            self.calculate_performance_metrics(verbose=False)  # Setting verbose=False to reduce output clutter
            
            # Return the specified objective
            if objective == 'NSE':
                return self.performance_metrics['NSE']
            elif objective == 'KGE':
                return self.performance_metrics['KGE']
            elif objective == 'RMSE':
                return -self.performance_metrics['RMSE']  # Negative for minimization
            elif objective == 'MAE':
                return -self.performance_metrics['MAE']  # Negative for minimization
            else:
                raise ValueError(f"Unknown objective function: {objective}")
        
        # Prepare to store results
        n_params = len(param_info)
        results = {
            'parameters': np.zeros((n_runs, n_params)),
            'performance': np.zeros(n_runs)
        }
        
        # Start timing
        start_time = time.time()
        
        if verbose:
            print(f"Starting uncertainty analysis with {n_runs} runs...")
            print(f"Sampling {n_params} parameters uniformly across their ranges")
            print(f"Evaluating with {objective} as the objective function")
        
        # Run Monte Carlo simulations
        for i in tqdm(range(n_runs), disable=not verbose):
            # Sample parameters uniformly from their ranges
            sampled_params = np.array([np.random.uniform(p['min'], p['max']) for p in param_info])
            
            # Store parameters
            results['parameters'][i, :] = sampled_params
            
            # Evaluate model and store performance
            results['performance'][i] = evaluate_model(sampled_params)
        
        # Sort results by performance (descending order)
        sort_indices = np.argsort(-results['performance'])
        sorted_performance = results['performance'][sort_indices]
        sorted_parameters = results['parameters'][sort_indices]
        
        # Get the best parameter sets
        best_indices = sort_indices[:save_best]
        best_performance = results['performance'][best_indices]
        best_parameters = results['parameters'][best_indices]
        
        # Save best parameter sets
        best_param_sets = []
        for i in range(save_best):
            param_set = create_param_dict(best_parameters[i])
            best_param_sets.append({
                'parameters': param_set,
                'performance': best_performance[i] if objective in ['NSE', 'KGE'] else -best_performance[i]
            })
        
        # Store performance with original (calibrated) parameters
        self.params = original_params
        self.states = copy.deepcopy(initial_states)
        self.run(verbose=False)
        self.calculate_performance_metrics(verbose=False)
        original_performance = self.performance_metrics[objective]
        if objective in ['RMSE', 'MAE']:
            original_performance = -original_performance  # Adjust sign for consistent comparison
        
        # Compare best run with original (calibrated) run
        best_vs_original = best_performance[0] - original_performance
        
        # Generate prediction intervals
        time_index = self.data.index
        dates = self.results['dates']  # Get dates from model results
        obs_q_valid = obs_q[valid_idx]
        
        # Create a dataframe to store all the best model runs
        best_runs_df = pd.DataFrame(index=time_index)
        best_runs_df['dates'] = dates  # Add dates column
        best_runs_df['observed'] = self.data[self.column_names['obs_q']]
        
        # Run model with best parameter sets and store results
        for i in range(save_best):
            self.params = create_param_dict(best_parameters[i])
            self.states = copy.deepcopy(initial_states)
            self.run(verbose=False)
            best_runs_df[f'best_{i+1}'] = self.results['discharge']
        
        # Run model with original parameters to get baseline
        self.params = original_params
        self.states = copy.deepcopy(initial_states)
        self.run(verbose=False)
        best_runs_df['original'] = self.results['discharge']
        
        # Calculate 95% prediction interval from best runs
        best_runs_df['q5'] = best_runs_df.filter(like='best_').quantile(0.025, axis=1)
        best_runs_df['q95'] = best_runs_df.filter(like='best_').quantile(0.975, axis=1)
        
        # Determine warmup period index for plotting
        warmup_idx = 0
        if exclude_warmup_in_plots and hasattr(self, 'warmup_end') and self.warmup_end is not None:
            if 'dates' in best_runs_df.columns:
                # Find the index of the first date after warmup_end
                warmup_idx = np.sum(best_runs_df['dates'] <= self.warmup_end)
                if verbose:
                    print(f"Excluding data up to {self.warmup_end} ({warmup_idx} timesteps) from plots as warmup period.")
            else:
                # Default to 10% if no dates available
                warmup_idx = int(len(obs_q) * 0.1)
                if verbose:
                    print(f"No dates found. Defaulting to exclude first {warmup_idx} timesteps (10% of data) from plots.")
        elif exclude_warmup_in_plots:
            # Default: exclude first 10% of the data
            warmup_idx = int(len(obs_q) * 0.1)
            if verbose:
                print(f"No warmup_end specified. Excluding first {warmup_idx} timesteps (10% of data) from plots.")
        
        # Plot results if requested
        if plot_results:
            plt.figure(figsize=(12, 6))
            
            # Apply warmup exclusion for plotting
            plot_df = best_runs_df.iloc[warmup_idx:].copy()
            plot_dates = dates[warmup_idx:] if isinstance(dates, np.ndarray) else plot_df.index
            
            # Plot uncertainty band
            plt.fill_between(plot_dates, plot_df['q5'], plot_df['q95'], 
                            color='lightgray', alpha=0.7, label='95% Prediction Interval')
            
            # Plot best run
            plt.plot(plot_dates, plot_df['best_1'], 'b-', linewidth=1, label='Best Run')
            
            # Plot original (calibrated) run
            plt.plot(plot_dates, plot_df['original'], 'r--', linewidth=1.5, label='Calibrated Run')
            
            # Plot observed data
            valid_obs = ~np.isnan(plot_df['observed'])
            plt.plot(plot_dates[valid_obs], plot_df['observed'][valid_obs], 'k.', 
                     markersize=3, label='Observed')
            
            plt.title(f'Uncertainty Analysis Results (n={n_runs}, objective={objective})')
            plt.xlabel('Time')
            plt.ylabel('Discharge (mm/day)')
            
            # Set sensible y-axis limits
            valid_data = np.concatenate([
                plot_df['observed'][valid_obs],
                plot_df['original'],
                plot_df['best_1'],
                plot_df['q95']
            ])
            max_val = np.nanmax(valid_data)
            leeway = 0.1 * max_val  # 10% extra space
            plt.ylim(0, max_val + leeway)
            
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add annotation about performance
            if objective in ['NSE', 'KGE']:
                better_text = "better" if best_vs_original > 0 else "worse"
                diff_text = f"Best run is {abs(best_vs_original):.4f} {better_text} than calibrated run"
            else:
                better_text = "better" if best_vs_original < 0 else "worse"
                diff_text = f"Best run is {abs(best_vs_original):.4f} {better_text} than calibrated run"
            
            plt.figtext(0.5, 0.01, diff_text, ha='center', fontsize=12)
            
            plt.tight_layout()
            plt.show()
            
            # Plot parameter distributions for the top runs
            n_params = len(param_info)
            n_cols = min(3, n_params)
            n_rows = (n_params + n_cols - 1) // n_cols
            
            plt.figure(figsize=(15, n_rows * 3))
            
            for i, p in enumerate(param_info):
                plt.subplot(n_rows, n_cols, i + 1)
                param_values = best_parameters[:save_best, i]
                
                plt.hist(param_values, bins=min(10, save_best), alpha=0.7)
                plt.axvline(p['default'], color='r', linestyle='--', 
                            linewidth=2, label='Calibrated')
                plt.axvline(best_parameters[0, i], color='b', linestyle='-',
                            linewidth=2, label='Best Run')
                
                plt.title(f"{p['group']}_{p['name']}")
                plt.xlabel('Parameter Value')
                plt.ylabel('Frequency')
                
                # Only show legend on the first subplot
                if i == 0:
                    plt.legend()
                    
            plt.tight_layout()
            plt.show()
        
        # Compute elapsed time
        elapsed_time = time.time() - start_time
        
        if verbose:
            print(f"\nUncertainty analysis completed in {elapsed_time:.2f} seconds")
            print(f"Analyzed {n_runs} parameter sets")
            
            print("\nTop Performance Values:")
            for i in range(min(5, save_best)):
                perf_value = best_performance[i] if objective in ['NSE', 'KGE'] else -best_performance[i]
                print(f"  Run {i+1}: {objective} = {perf_value:.4f}")
            
            print(f"\nOriginal (Calibrated) Performance: {objective} = {original_performance:.4f}")
            
            if objective in ['NSE', 'KGE']:
                better_text = "better" if best_vs_original > 0 else "worse"
            else:
                better_text = "better" if best_vs_original < 0 else "worse"
                
            print(f"Best run is {abs(best_vs_original):.4f} {better_text} than calibrated run")
        
        # Restore original parameters and states
        self.params = original_params
        self.states = copy.deepcopy(initial_states)
        
        # Return results
        return {
            'best_parameter_sets': best_param_sets,
            'uncertainty_bounds': {
                'lower': best_runs_df['q5'].tolist(),
                'upper': best_runs_df['q95'].tolist()
            },
            'best_runs': best_runs_df,
            'objective': objective,
            'n_runs': n_runs,
            'save_best': save_best,
            'original_performance': original_performance,
            'best_performance': best_performance[0] if objective in ['NSE', 'KGE'] else -best_performance[0],
            #'warmup_idx': warmup_idx
        }