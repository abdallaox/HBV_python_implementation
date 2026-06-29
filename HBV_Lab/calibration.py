class calibration:

    def calibrate(self, method='Nelder-Mead', objective='NSE', iterations=100,
                            verbose=True, plot_results=True, progress_callback=None):
        """
        Calibrate an HBV model's parameters to optimize the objective function.

        Parameters:
        -----------
        self : HBVmodel
            The HBV model instance to calibrate
        method : str, default 'Nelder-Mead'
            Optimization method to use (see scipy.optimize.minimize).
            A gradient-free method is used by default because the HBV objective
            surface is piecewise-constant (many min/max thresholds), which makes
            finite-difference gradients ~0 and causes gradient-based methods such
            as 'SLSQP' or 'L-BFGS-B' to terminate immediately without improving
            on the starting values. Gradient-free options that respect the
            parameter bounds include 'Nelder-Mead' and 'Powell'.
        objective : str, default 'NSE'
            Objective function to maximize. Options are:
            - 'NSE': Nash-Sutcliffe Efficiency (higher is better)
            - 'KGE': Kling-Gupta Efficiency (higher is better)
            - 'RMSE': Root Mean Square Error (lower is better)
            - 'MAE': Mean Absolute Error (lower is better)
        iterations : int, default 100
            Maximum number of iterations for the optimizer
        verbose : bool, default True
            Whether to print progress information
        plot_results : bool, default True
            Whether to plot the final results after calibration
        progress_callback : callable, optional
            Called once per optimizer iteration as
            ``progress_callback(iteration, total_iterations, current_value, best_value)``,
            where the values are in human-facing objective terms (e.g. NSE). Useful for
            driving an external progress UI (e.g. the MCP server). Exceptions raised by
            the callback are swallowed so they cannot interrupt calibration.

        Returns:
        --------
        dict
            Dictionary with keys ``parameters`` (optimized parameter dict),
            ``performance`` (final metrics), ``optimization_result`` (the SciPy result),
            and ``trajectory`` (best objective value per iteration).
        """
        import scipy.optimize as opt
        import numpy as np
        import time
        import copy
        
        if self.data is None:
            raise ValueError("No data loaded. Use load_data() method first.")
            
        # Check if observed discharge data is available
        if (self.column_names['obs_q'] is None or 
            self.column_names['obs_q'] not in self.data.columns):
            raise ValueError("Observed discharge data is required for calibration.")
        # Store the initial states to use it later for reseting the model 
        initial_states = self.states
        # Extract observed discharge data
        obs_q = self.data[self.column_names['obs_q']].values
        
        # Get valid indices (where obs_q is not NaN)
        valid_idx = ~np.isnan(obs_q)
        if np.sum(valid_idx) == 0:
            raise ValueError("No valid observed discharge values found.")
        
        # Initial parameter values (using default values)
        initial_params = []
        param_names = []
        param_bounds = []
        param_groups = []
        
        # Create flat parameter list for optimization
        for group_name, group in self.params.items():
            for param_name, param_info in group.items():
                initial_params.append(param_info['default'])
                param_names.append(f"{group_name}_{param_name}")
                param_bounds.append((param_info['min'], param_info['max']))
                param_groups.append(group_name)
        
        # Helper function to convert flat parameter array to nested dictionary
        def create_param_dict(flat_params):
            # Create a deep copy of current parameters structure
            param_dict = {group: {} for group in set(param_groups)}
            
            # Fill in the parameter values
            for i, (name, group) in enumerate(zip(param_names, param_groups)):
                param_name = name.split('_', 1)[1]  # Extract parameter name
                
                # Initialize if not exists
                if param_name not in param_dict[group]:
                    param_dict[group][param_name] = {}
                
                # Copy min/max from original params
                orig_group = self.params[group]
                param_dict[group][param_name]['min'] = orig_group[param_name]['min']
                param_dict[group][param_name]['max'] = orig_group[param_name]['max']
                
                # Set the default to the optimized value
                param_dict[group][param_name]['default'] = flat_params[i]
            
            return param_dict
        
        
        
        # Define the objective function to minimize
        def objective_function(params):
            # Update parameters structure with flat parameter array
            param_dict = create_param_dict(params)
            
            # Store original parameters to restore later
            original_params = copy.deepcopy(self.params)
            
            # Update model parameters
            self.params = param_dict
            
            
            # Run the model (run() computes performance metrics internally when
            # observed discharge is available, honouring the warmup period)
            self.run(verbose=False)

            # Ensure performance metrics reflect the current parameter set
            self.calculate_performance_metrics(verbose=False)

            # Calculate objective function value
            if objective == 'NSE':
                # Nash-Sutcliffe Efficiency (to be maximized)
                # mean_obs = np.mean(obs_q_valid)
                # nse_numerator = np.sum((obs_q_valid - sim_q) ** 2)
                # nse_denominator = np.sum((obs_q_valid - mean_obs) ** 2)
                # value = 1 - (nse_numerator / nse_denominator)
                # # For minimization, return negative NSE
                
                return - self.performance_metrics['NSE']
                
            elif objective == 'KGE':
                # # Kling-Gupta Efficiency (to be maximized)
                # mean_sim = np.mean(sim_q)
                # mean_obs = np.mean(obs_q_valid)
                # std_sim = np.std(sim_q)
                # std_obs = np.std(obs_q_valid)
                
                # r = np.corrcoef(obs_q_valid, sim_q)[0, 1]  # Correlation
                # alpha = (std_sim/mean_sim) / (std_obs/mean_sim)  # Relative variability
                # beta = mean_sim / mean_obs  # Bias
                
                # kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
                # For minimization, return negative KGE
                return - self.performance_metrics['KGE']
                
            elif objective == 'RMSE':
                # Root Mean Square Error (to be minimized)
                # rmse = np.sqrt(np.mean((obs_q_valid - sim_q) ** 2))
                return self.performance_metrics['RMSE']
                
            elif objective == 'MAE':
                # Mean Absolute Error (to be minimized)
                # mae = np.mean(np.abs(obs_q_valid - sim_q))
                return self.performance_metrics['MAE']
            
            else:
                raise ValueError(f"Unknown objective function: {objective}")
        
        # Callback function to track progress.
        # objective_function always returns a value to be MINIMIZED (for NSE/KGE it
        # returns the negative metric), so the best-so-far starts at +inf for every
        # objective and improves downward.
        num_iter = [0]
        best_value = [float('inf')]
        start_time = time.time()
        # Per-iteration best-so-far objective values, in human-facing terms
        # (e.g. NSE/KGE as-is, RMSE/MAE as-is). Returned to the caller.
        trajectory = []

        def callback(params):
            num_iter[0] += 1
            current_value = objective_function(params)

            # For NSE and KGE, we're minimizing the negative value
            if objective in ['NSE', 'KGE']:
                display_value = -current_value
                is_better = current_value < best_value[0]
            else:
                display_value = current_value
                is_better = current_value < best_value[0]

            if is_better:
                best_value[0] = current_value

            # Best objective so far, in human-facing terms
            best_display = -best_value[0] if objective in ['NSE', 'KGE'] else best_value[0]
            trajectory.append(best_display)

            # Optional external progress hook (e.g. for an MCP/agent UI).
            # Kept defensive: a misbehaving callback must not break calibration.
            if progress_callback is not None:
                try:
                    progress_callback(num_iter[0], iterations, display_value, best_display)
                except Exception:
                    pass

            if verbose and num_iter[0] % max(1, iterations // 10) == 0:
                elapsed = time.time() - start_time
                print(f"Iteration {num_iter[0]}/{iterations}, "
                    f"{objective}: {display_value:.4f}, "
                    f"Time: {elapsed:.1f}s")
        
        if verbose:
            print(f"Starting calibration using {method} method...")
            print(f"Optimizing {objective} with {len(param_names)} parameters and {iterations} iterations")
        
        # Store original parameters to restore if needed
        original_params = copy.deepcopy(self.params)
        
        try:
            # Run optimization
            result = opt.minimize(
                objective_function,
                initial_params,
                method=method,
                bounds=param_bounds,
                callback=callback,
                options={'maxiter': iterations}
            )
            
            # Get optimized parameters
            opt_params = result.x
            
            # Create parameter dictionary from optimized values
            optimized_params = create_param_dict(opt_params)
            
            # Update model with optimized parameters
            self.params = optimized_params
            
            # Run the model with optimized parameters
            
            self.run(verbose)
            
            # Calculate final performance metrics
            self.calculate_performance_metrics(verbose)
            
            # Display results
            if verbose:
                print("\nCalibration completed!")
                print(f"Final {objective}: {-result.fun if objective in ['NSE', 'KGE'] else result.fun:.4f}")
                print(f"Success: {result.success}, Message: {result.message}")
                print("\nOptimized Parameters:")
                
                for group_name, group in optimized_params.items():
                    print(f"\n{group_name.upper()} parameters:")
                    for param_name, param_info in group.items():
                        print(f"  {param_name}: {param_info['default']:.4f} (range: {param_info['min']}-{param_info['max']})")
                
                print("\nPerformance Metrics:")
                for metric, value in self.performance_metrics.items():
                    print(f"  {metric}: {value:.4f}")
            
            # Plot results if requested
            if plot_results:
                self.plot_results(show_plots=True)
            
            # Return optimized parameters and performance
            return {
                'parameters': optimized_params,
                'performance': self.performance_metrics,
                'optimization_result': result,
                'trajectory': trajectory
            }
            
        except Exception as e:
            # Restore original parameters on error
            self.params = original_params
            print(f"Calibration failed with error: {str(e)}")
            raise