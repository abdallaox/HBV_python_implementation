"""
HBV Hydrological Model

This module integrates the snow, soil, and response routines into a complete
HBV-like hydrological model. It handles parameter management, data reading, model
execution, and output visualization.

Usage:
    from hbv_model import HBVModel
    model = HBVModel()
    model.load_data("path/to/data.csv")
    model.set_parameters(params)
    model.run()
    model.plot_results()
    model.save_results()
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import os
import datetime
from types import MethodType
from .uncertainty import uncertainty
from .calibration import calibration
## from calibration import calibrate_hbv_model
##  from uncertainty import evaluate_uncertainty
from .hbv_step import hbv_step
from .routing import route_with_maxbas

class HBVModel(uncertainty, calibration):
    """
    HBV hydrological model class that integrates snow, soil, and response routines.
    """
    
    def __init__(self):
        """Initialize the HBV model with default values."""
        self.data = None
        self.results = None
        self.params= {
            'snow': {
                'TT': {'min': -2.0, 'max': 2.0, 'default': 0.0},
                'CFMAX': {'min': 1.0, 'max': 6.0, 'default': 3.5},
               # 'PCF': {'min': 0.5, 'max': 1.5, 'default': 1.0},
                'SFCF': {'min': 0.5, 'max': 1.5, 'default': 1.0},
                'CFR': {'min': 0.0, 'max': 0.2, 'default': 0.05},
                'CWH': {'min': 0.0, 'max': 0.2, 'default': 0.1}
            },
            'soil': {
                'FC': {'min': 50.0, 'max': 300.0, 'default': 150.0},
                'LP': {'min': 0.3, 'max': 1.0, 'default': 0.7},
                'BETA': {'min': 1.0, 'max': 5.0, 'default': 2.0}
            },
            'response': {
                'K0': {'min': 0.1, 'max': 0.9, 'default': 0.5},
                'K1': {'min': 0.05, 'max': 0.5, 'default': 0.2},
                'K2': {'min': 0.01, 'max': 0.1, 'default': 0.05},
                'UZL': {'min': 5.0, 'max': 50.0, 'default': 20.0},
                'PERC': {'min': 0.5, 'max': 3.0, 'default': 1.5},
                'MAXBAS' :{'min': 1, 'max': 10, 'default': 3}  
            }
        }
        
        # Initial states 
        self.states = {
            'snowpack': 0.0,         # Snow pack (mm)
            'liquid_water': 0.0,      # Liquid water in snow (mm)
            'soil_moisture': 30.0,    # Soil moisture (mm)
            'upper_storage': 10.0,    # Upper zone storage (mm)
            'lower_storage': 20.0     # Lower zone storage (mm)
        }
        
        # Initialize time tracking
        self.start_date = None
        self.end_date = None
        self.time_step = 'D'  # Default: daily
        ##### link externaly defined functions to the model
        # self.calibrate = MethodType(calibrate_hbv_model, self)
        # self.evaluate_uncertainty = MethodType(evaluate_uncertainty, self)
    
    def load_data(self, file_path=None, data=None, date_column='Date',
              precip_column='Precipitation', temp_column='Temperature',
              pet_column='PotentialET', obs_q_column=None, date_format='%Y%m%d',
              start_date=None, warmup_end=None, end_date=None  ):
        """
        Load data from file or DataFrame, handling PET interpolation and flexible date parsing.

        Parameters:
        -----------
        file_path : str, optional
            Path to CSV file.
        data : pandas.DataFrame, optional
            Pre-loaded DataFrame.
        date_column : str, default 'Date'
            Name of column containing date.
        precip_column : str
        temp_column : str
        pet_column : str
        obs_q_column : str, optional
        start_date : str or datetime, optional
        end_date : str or datetime, optional
        date_format : str, optional
            Format string for parsing dates (e.g. '%Y%m%d' for '19510601').
        warmup_end : str or datetime, optional
            The end date of the warmup period. Data before or at this date will be included
            in the simulation but excluded from performance evaluation.
        """
        import pandas as pd

        if file_path is not None:
            data = pd.read_csv(file_path)

        if data is None:
            raise ValueError("Either file_path or data must be provided.")

        has_date = date_column in data.columns and data[date_column].notna().all()

        if has_date:
            try:
               
                data[date_column] = pd.to_datetime(data[date_column], format=date_format)
               
            except Exception as e:
                print(f"Warning: Failed to convert {date_column} to datetime. {e}")
                has_date = False

        # Expand PET if monthly means
        if pet_column in data.columns:
            pet_data = data[pet_column].dropna()
            if len(pet_data) == 12:
                print("Detected 12 PET values (monthly means), expanding to daily values...")
                monthly_pet = pd.DataFrame({'month': range(1, 13), 'pet': pet_data.values})

                if has_date:
                    min_date = data[date_column].min()
                    max_date = data[date_column].max()
                    full_date_range = pd.date_range(start=min_date, end=max_date, freq='D')
                    daily_df = pd.DataFrame({date_column: full_date_range, 'month': full_date_range.month})
                else:
                    daily_df = pd.DataFrame({'index': data.index, 'month': ((data.index // 30) % 12 + 1)})

                daily_df = daily_df.merge(monthly_pet, on='month', how='left')
                daily_df['pet'] = daily_df['pet'].where(daily_df['pet'] != daily_df['pet'].shift(), np.nan)
                daily_df['pet'] = daily_df['pet'].interpolate(method='linear')

                # smoothed_pet = pd.Series(monthly_pet['pet'].tolist() * 3).interpolate().iloc[12:24].values
                # daily_df['pet'] = daily_df['month'].map(lambda m: smoothed_pet[m - 1])

                data = data.drop(columns=[pet_column])
                merge_col = date_column if has_date else 'index'
                data = data.reset_index().merge(daily_df[[merge_col, 'pet']], on=merge_col, how='left').set_index('index')
                data = data.rename(columns={'pet': pet_column})
        
        if has_date:
            # Convert all dates to datetime objects first
            start_dt = pd.to_datetime(start_date, format=date_format) if start_date else None
            end_dt = pd.to_datetime(end_date, format=date_format) if end_date else None
            warmup_dt = pd.to_datetime(warmup_end, format=date_format) if warmup_end else None
            
            # Validate date order
            if start_dt is not None and end_dt is not None:
                if start_dt >= end_dt:
                    raise ValueError("start_date must be earlier than end_date")
            
            if warmup_dt is not None:
                if start_dt is not None and warmup_dt <= start_dt:
                    raise ValueError("warmup_end must be after start_date")
                if end_dt is not None and warmup_dt >= end_dt:
                    raise ValueError("warmup_end must be before end_date")
            # apply filtering    
            if start_date is not None:
                data = data[data[date_column] >= pd.to_datetime(start_date,format=date_format)]
            if end_date is not None:
                data = data[data[date_column] <= pd.to_datetime(end_date,format=date_format)]

        self.data = data.reset_index(drop=True)
        self.column_names = {
            'date': date_column if has_date else None,
            'precip': precip_column,
            'temp': temp_column,
            'pet': pet_column,
            'obs_q': obs_q_column
        }

        if has_date and len(data) > 0:
            self.start_date = data[date_column].min()
            self.end_date = data[date_column].max()
            diff = data[date_column].diff().dropna()
            if not diff.empty:
                modal_diff = diff.mode().iloc[0]
                self.time_step = 'D' if modal_diff == pd.Timedelta(days=1) else (
                    'H' if modal_diff == pd.Timedelta(hours=1) else str(modal_diff))
                print(f"Time step detected: {self.time_step}")
        else:
            self.start_date = None
            self.end_date = None
            warmup_end = None
            self.time_step = 'Index-based'
            print("No date column found; using index as time step.")
             

        # Store warmup_end
        self.warmup_end = None
        if warmup_end is not None:
            try:
                self.warmup_end = pd.to_datetime(warmup_end, format=date_format)
                print(f"Warmup period ends at: {self.warmup_end}")
            except:
                print(f"Warning: Could not parse warmup_end date '{warmup_end}'. 10% warmup period will be used.")
        else:   print(f"No warmup_end specified. Excluding first 10% of data when evaluating——as warmup period.")
        
        print(f"Loaded data with {len(self.data)} time steps, from {self.start_date} to {self.end_date}")

    
    def set_parameters(self, custom_ranges=None):
        """Set parameters and thier ranges to overwrite the default.
        
        Args:
            custom_ranges (dict, optional): A dictionary with the same structure as `self.params`, 
                                containing custom min/max/default values.
        """
        
        # If custom ranges are provided, update the defaults
        if custom_ranges is not None:
            for group in custom_ranges:
                if group in self.params:
                    for param in custom_ranges[group]:
                        if param in self.params[group]:
                            self.params[group][param].update(custom_ranges[group][param])
                        else:
                            raise ValueError(f"Unknown parameter '{param}' in group '{group}'")
                else:
                    raise ValueError(f"Unknown group '{group}'")
        else: raise ValueError(f"No parameters provided—–make sure to provide the input in the correct format")
  

    def set_initial_conditions(self, snowpack=None, liquid_water=None, 
                              soil_moisture=None, upper_storage=None, 
                              lower_storage=None):
        """
        Set initial conditions for model states.
        
        Parameters:
        -----------
        snowpack : float, optional
            Initial snow pack (mm)
        liquid_water : float, optional
            Initial liquid water in snow (mm)
        soil_moisture : float, optional
            Initial soil moisture (mm)
        upper_storage : float, optional
            Initial upper zone storage (mm)
        lower_storage : float, optional
            Initial lower zone storage (mm)
        """
        if snowpack is not None:
            self.states['snowpack'] = snowpack
        if liquid_water is not None:
            self.states['liquid_water'] = liquid_water
        if soil_moisture is not None:
            self.states['soil_moisture'] = soil_moisture
        if upper_storage is not None:
            self.states['upper_storage'] = upper_storage
        if lower_storage is not None:
            self.states['lower_storage'] = lower_storage
            
        print("Initial conditions updated.")
    
    
    def run(self, verbose= True):
        """
        Run the HBV model for the entire simulation period.
        """
        # Store the initial states to reset them at the end
        initial_states = self.states

        if self.data is None:
            raise ValueError("No data loaded. Use load_data() method first.")
            
        # Extract data arrays
        precip = self.data[self.column_names['precip']].values
        temp = self.data[self.column_names['temp']].values
        pet = self.data[self.column_names['pet']].values
        
        # Get dates if available
        if self.column_names['date'] in self.data.columns:
            dates = self.data[self.column_names['date']].values
        else:
            dates = np.arange(len(precip))
            
        # Get observed discharge if available
        if self.column_names['obs_q'] is not None and self.column_names['obs_q'] in self.data.columns:
            obs_q = self.data[self.column_names['obs_q']].values
        else:
            obs_q = None
            
        # Initialize storage arrays
        n_steps = len(precip)
        
        # Initialize results dictionary
        results = {
            'dates': dates,
            'snowpack': np.zeros(n_steps),
            'liquid_water': np.zeros(n_steps),
            'runoff_from_snow': np.zeros(n_steps),
            'soil_moisture': np.zeros(n_steps),
            'recharge': np.zeros(n_steps),
            'runoff_soil': np.zeros(n_steps),
            'actual_et': np.zeros(n_steps),
            'upper_storage': np.zeros(n_steps),
            'lower_storage': np.zeros(n_steps),
            'quick_flow': np.zeros(n_steps),
            'intermediate_flow': np.zeros(n_steps),
            'baseflow': np.zeros(n_steps),
            'discharge': np.zeros(n_steps),
            'precipitation': precip,
            'temperature': temp,
            'potential_et': pet
        }
        
        if obs_q is not None:
            results['observed_q'] = obs_q
        
        
        if verbose: print(f"Starting model run for {n_steps} time steps...")
        
        # Main simulation loop
        for t in range(n_steps):
          
            self.states, fluxes = hbv_step(precip[t],  temp[t], pet[t], self.params, self.states)
            
            # Store results
            results['snowpack'][t] = self.states['snowpack']
            results['liquid_water'][t] = self.states['liquid_water']
            results['runoff_from_snow'][t] = fluxes['runoff_from_snow']
            results['soil_moisture'][t] = self.states['soil_moisture']
            results['recharge'][t] = fluxes['recharge']
            results['runoff_soil'][t] = fluxes['runoff_soil']
            results['actual_et'][t] = fluxes['actual_et']
            results['upper_storage'][t] = self.states['upper_storage']
            results['lower_storage'][t] = self.states['lower_storage']
            results['quick_flow'][t] = fluxes['quick_flow']
            results['intermediate_flow'][t] = fluxes['intermediate_flow']
            results['baseflow'][t] = fluxes['baseflow']
            results['discharge'][t] = fluxes['discharge']
            
        # # Store final storage values
        # self.storages['snowpack'] = snowpack
        # self.storages['liquid_water'] = liquid_water
        # self.storages['soil_moisture'] = soil_moisture
        # self.storages['upper_storage'] = upper_storage
        # self.storages['lower_storage'] = lower_storage
        
        # Store results
        # Add routing if MAXBAS > 1
        maxbas = int(self.params['response']['MAXBAS']['default'])
        if maxbas > 1:
            if verbose: 
                print(f"Applying MAXBAS routing with n={maxbas} time steps")
            routed_discharge = route_with_maxbas(results['discharge'], maxbas)
            results['discharge'] = routed_discharge
            
            # Also route components if needed
            results['quick_flow'] = route_with_maxbas(results['quick_flow'], maxbas)
            results['intermediate_flow'] = route_with_maxbas(results['intermediate_flow'], maxbas)
            results['baseflow'] = route_with_maxbas(results['baseflow'], maxbas)
        self.results = results
        
        if verbose: print("Model run completed successfully!")
        self.states= initial_states  # states restored to the initial
        
        # Calculate performance metrics if observed discharge is available
        if obs_q is not None:
            self.calculate_performance_metrics(verbose)
        
        return results
    
    def calculate_performance_metrics(self, verbose = True):
        """
        Calculate performance metrics if observed discharge is available.
        Uses the warmup_end parameter that was set during data loading to exclude
        warmup period from performance evaluation.
        """
        if 'observed_q' not in self.results:
            print("No observed discharge data available for performance evaluation.")
            return
            
        # Get simulated and observed discharge
        sim_q = self.results['discharge']
        obs_q = self.results['observed_q']
        
        # Define the evaluation period (exclude warmup period)
        warmup_idx = 0
        
        if hasattr(self, 'warmup_end') and self.warmup_end is not None:
            # If warmup_end is stored in the model and dates are available
            if 'dates' in self.results:
                dates = self.results['dates']
                # Find the index of the first date after warmup_end
                warmup_idx = np.sum(dates <= self.warmup_end)
                if verbose: print(f"Excluding data up to {self.warmup_end} ({warmup_idx} timesteps) as warmup period.")
            else:
                if verbose: print("Warning: No dates found in results. Unable to apply date-based warmup period.")
                # Default to 10% if no dates available
                warmup_idx = int(len(obs_q) * 0.1)
                if verbose: print(f"Defaulting to exclude first {warmup_idx} timesteps (10% of data) as warmup period.")
        else:
            # Default: exclude first 10% of the data
            warmup_idx = int(len(obs_q) * 0.1)
            if verbose: print(f"No warmup_end specified. Excluding first {warmup_idx} timesteps (10% of data) as warmup period.")
        
        # Apply the warmup period
        if warmup_idx > 0:
            sim_q = sim_q[warmup_idx:]
            obs_q = obs_q[warmup_idx:]
        
        # Remove NaN values
        valid_idx = ~np.isnan(obs_q)
        if np.sum(valid_idx) == 0:
            print("No valid observed discharge values found after applying warmup period.")
            return
            
        sim_q_valid = sim_q[valid_idx]
        obs_q_valid = obs_q[valid_idx]
        
        # Calculate Nash-Sutcliffe Efficiency (NSE)
        mean_obs = np.mean(obs_q_valid)
        nse_numerator = np.sum((obs_q_valid - sim_q_valid) ** 2)
        nse_denominator = np.sum((obs_q_valid - mean_obs) ** 2)
        nse = 1 - (nse_numerator / nse_denominator)
        
        # Calculate Kling-Gupta Efficiency (KGE)
        mean_sim = np.mean(sim_q_valid)
        std_obs = np.std(obs_q_valid)
        std_sim = np.std(sim_q_valid)
        
        r = np.corrcoef(obs_q_valid, sim_q_valid)[0, 1]  # Correlation coefficient
        alpha = (std_sim/mean_sim) / (std_obs/mean_obs)  # Relative variability
        beta = mean_sim / mean_obs  # Bias
        
        kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
        
        # Calculate percent bias
        pbias = 100 * (np.sum(sim_q_valid - obs_q_valid) / np.sum(obs_q_valid))
        
        # Calculate RMSE and MAE
        rmse = np.sqrt(np.mean((sim_q_valid - obs_q_valid) ** 2))
        mae = np.mean(np.abs(sim_q_valid - obs_q_valid))
        
        # Store metrics
        self.performance_metrics = {
            'NSE': nse,
            'KGE': kge,
            'PBIAS': pbias,
            'RMSE': rmse,
            'MAE': mae,
            'r': r,
           # 'warmup_end': self.warmup_end if hasattr(self, 'warmup_end') else None,
           # 'warmup_timesteps': warmup_idx
        }
        if verbose:
            print(f"Performance metrics calculated:")
            print(f"NSE: {nse:.3f}")
            print(f"KGE: {kge:.3f}")
            print(f"PBIAS: {pbias:.1f}%")
            print(f"Correlation: {r:.3f}")
    
    def plot_results(self, output_file=None, show_plots=True, exclude_warmup=True):
        """
        Plot model results with customized layout and additional figures.
        
        Parameters:
        -----------
        output_file : str, optional
            If provided, save the plot to this file
        show_plots : bool, default True
            Whether to display the plots
        exclude_warmup : bool, default True
            Whether to exclude the warmup period from the plots
        """
        # --- Create directory if it doesn't exist ---
        if output_file is not None:
            output_dir = os.path.dirname(output_file)
            if output_dir:  # only attempt if there's a directory part
                os.makedirs(output_dir, exist_ok=True)
        if self.results is None:
            raise ValueError("No results to plot. Run the model first.")
            
        # Determine the warmup period index
        warmup_idx = 0
        
        if exclude_warmup:
            if hasattr(self, 'warmup_end') and self.warmup_end is not None:
                # If warmup_end is stored in the model and dates are available
                if 'dates' in self.results:
                    dates = self.results['dates']
                    # Find the index of the first date after warmup_end
                    if isinstance(dates[0], (datetime.datetime, np.datetime64)):
                        warmup_idx = np.sum(dates <= self.warmup_end)
                        print(f"Excluding data up to {self.warmup_end} ({warmup_idx} timesteps) as warmup period.")
                    else:
                        # Default to 10% if dates are not datetime objects
                        warmup_idx = int(len(dates) * 0.1)
                        print(f"Dates are not datetime objects. Defaulting to exclude first {warmup_idx} timesteps (10% of data) as warmup period.")
                else:
                    # Default to 10% if no dates available
                    warmup_idx = int(len(self.results['precipitation']) * 0.1)
                    print(f"No dates found in results. Defaulting to exclude first {warmup_idx} timesteps (10% of data) as warmup period.")
            else:
                # Default: exclude first 10% of the data
                warmup_idx = int(len(self.results['precipitation']) * 0.1)
                print(f"No warmup_end specified. Excluding first {warmup_idx} timesteps (10% of data) as warmup period.")
        else:
            print("Including entire simulation period (including warmup period)")
        
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(12, 28))  # Increased height for more subplots
        
        # Define subplots layout (now 9 subplots)
        axs = []
        axs.append(fig.add_subplot(11, 1, 1))   # Precipitation
        axs.append(fig.add_subplot(11, 1, 2))   # Temperature
        axs.append(fig.add_subplot(11, 1, 3))   # Snow pack and liquid water
        axs.append(fig.add_subplot(11, 1, 4))   # Runoff from snow
        axs.append(fig.add_subplot(11, 1, 5))   # Potential and actual ET
        axs.append(fig.add_subplot(11, 1, 6))   # Soil moisture
        axs.append(fig.add_subplot(11, 1, 7))   # Recharge (output from soil to response)
        axs.append(fig.add_subplot(11, 1, 8))   # Groundwater storages
        axs.append(fig.add_subplot(11, 1, 9))   # Discharge components and total
        axs.append(fig.add_subplot(11, 1, 10))  # Discharge components and total
        axs.append(fig.add_subplot(11, 1, 11))  # Discharge components and total

        # Get dates for x-axis
        dates = self.results['dates'][warmup_idx:] if warmup_idx > 0 else self.results['dates']
        if isinstance(dates[0], (datetime.datetime, np.datetime64)):
            date_formatter = DateFormatter('%Y-%m-%d')
            is_datetime = True
        else:
            is_datetime = False
        
        # Function to get data with warmup period excluded
        def get_data(key):
            return self.results[key][warmup_idx:] if warmup_idx > 0 else self.results[key]
        
        # 1. Precipitation
        ax1 = axs[0]
        ax1.bar(dates, get_data('precipitation'), color='skyblue', label='Precipitation')
        ax1.set_ylabel('Precipitation (mm)')
        ax1.set_title('Precipitation')
        ax1.legend(loc='upper right')
        
        # 2. Temperature with TT threshold
        ax2 = axs[1]
        ax2.plot(dates, get_data('temperature'), color='red', label='Temperature')
        ax2.axhline(y=self.params['snow']['TT']['default'], color='gray', linestyle='--', 
                label=f"TT Threshold ({self.params['snow']['TT']['default']}°C)")
        ax2.set_ylabel('Temperature (°C)')
        ax2.set_title('Temperature')
        ax2.legend(loc='upper right')
        
        # 3. Snow pack and liquid water
        ax3 = axs[2]
        ax3.plot(dates, get_data('snowpack'), color='blue', label='Snow Pack')
        ax3.plot(dates, get_data('liquid_water'), color='lightblue', label='Liquid Water')
        ax3.set_ylabel('Water (mm)')
        ax3.set_title('Snow Pack and Liquid Water Content')
        ax3.legend(loc='upper right')
        
        # 4. Runoff from snow
        ax4 = axs[3]
        ax4.plot(dates, get_data('runoff_from_snow'), color='skyblue', label='Runoff from Snow')
        ax4.set_ylabel('Runoff (mm/day)')
        ax4.set_title('Runoff from the Snow Routine')
        ax4.legend(loc='upper right')
        
        # 5. Potential and actual ET
        ax5 = axs[4]
        ax5.plot(dates, get_data('potential_et'), color='orange', label='Potential ET')
        ax5.plot(dates, get_data('actual_et'), color='green', label='Actual ET')
        ax5.set_ylabel('ET (mm/day)')
        ax5.set_title('Potential and Actual Evapotranspiration')
        ax5.legend(loc='upper right')
        
        # 6. Soil moisture
        ax6 = axs[5]
        ax6.plot(dates, get_data('soil_moisture'), color='brown', label='Soil Moisture')
        ax6.axhline(y=self.params['soil']['FC']['default'], color='gray', linestyle='--', 
                label=f"Field Capacity ({self.params['soil']['FC']['default']:.2f} mm)")
        ax6.set_ylabel('Soil Moisture (mm)')
        ax6.set_title('Soil Moisture')
        ax6.legend(loc='upper right')
        
        # 7. Recharge (output from soil to response routine)
        ax7 = axs[6]
        ax7.plot(dates, get_data('recharge'), color='purple',linewidth=0.5, label='Recharge')
        ax7.plot(dates, get_data('runoff_soil'), color='red',linewidth=0.5, label='Runoff (overflow from soil)')
        ax7.set_ylabel('Water (mm/day)')
        ax7.set_title('Soil Output to the Response Routine')
        ax7.legend(loc='upper right')
        
        # 8. upper storages
        ax8 = axs[7]
        ax8.plot(dates, get_data('upper_storage'), color='lightcoral', label='Upper Storage')
        ax8.axhline(y=self.params['response']['UZL']['default'], color='gray', linestyle='--', label='Fast Response Threshold (ULZ)')
        ax8.set_ylabel('Storage (mm)')
        ax8.set_title('Upper Tank Storages')
        ax8.legend(loc='upper right')
        
        # 9. Groundwater storages
        ax9 = axs[8]
        ax9.plot(dates, get_data('lower_storage'), color='darkblue', label='Lower Storage')
        ax9.set_ylabel('Storage (mm)')
        ax9.set_title('Lower Tank Storages')
        ax9.legend(loc='upper right')
        
        # 10. Discharge Components (Stacked)
        ax10 = axs[9]
        
        # Stackplot with components only
        ax10.stackplot(dates,
                    get_data('baseflow'),
                    get_data('intermediate_flow'),
                    get_data('quick_flow'),
                    labels=['Baseflow', 'Intermediate Flow', 'Quick Flow'],
                    colors=['royalblue', 'darkorange', 'tomato'])
        
        # Add total discharge line
        ax10.plot(dates, get_data('discharge'), 
                color='black', linestyle=':', linewidth=0.5,
                label='Total Discharge (sum)')
        
        ax10.set_ylabel('Flow (mm/day)')
        ax10.set_title('Runoff Components (Stacked)')
        ax10.legend(loc='upper right')
        
        # 11. Discharge Comparison (Total vs Observed)
        ax11 = axs[10]
        
        # Plot simulated total
        ax11.plot(dates, get_data('discharge'), 
                color='darkgreen', linewidth=2,
                label='Simulated Discharge')
        
        # Plot observed if available
        if 'observed_q' in self.results:
            ax11.plot(dates, get_data('observed_q'), 
                    color='black', linestyle='--', linewidth=1.5,
                    label='Observed Discharge')
            
            # Add performance metrics to title
            if hasattr(self, 'performance_metrics'):
                metrics = self.performance_metrics
                ax11.set_title(
                    f"Discharge Comparison (NSE: {metrics['NSE']:.2f}, KGE: {metrics['KGE']:.2f}, PBIAS: {metrics['PBIAS']:.2f}) "
                    
                )
            else: 
                ax11.set_title('Discharge Comparison')
        
        ax11.set_ylabel('Discharge (mm/day)')
        ax11.set_xlabel('Date')
        
        ax11.legend(loc='upper right')
        
        # Format x-axis dates if available
        if is_datetime:
            for ax in axs:
                ax.xaxis.set_major_formatter(date_formatter)
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {output_file}")
            
        if show_plots:
            plt.show()
        return None  
        
    def save_results(self, output_file):
        """
        Save model results to a CSV file.
        
        Parameters:
        -----------
        output_file : str
            Path to output file
        """
        # --- Create directory if it doesn't exist ---
        if output_file is not None:
            output_dir = os.path.dirname(output_file)
            if output_dir:  # only attempt if there's a directory part
                os.makedirs(output_dir, exist_ok=True)
            if self.results is None:
                raise ValueError("No results to save. Run the model first.")
            
        # Create a DataFrame from results
        results_df = pd.DataFrame()
        
        # Add date column if dates are available
        if isinstance(self.results['dates'][0], (datetime.datetime, np.datetime64)):
            results_df['Date'] = self.results['dates']
        else:
            results_df['TimeStep'] = self.results['dates']
            
        # Add input data
        results_df['Precipitation'] = self.results['precipitation']
        results_df['Temperature'] = self.results['temperature']
        results_df['PotentialET'] = self.results['potential_et']
        
        # Add observed discharge if available
        if 'observed_q' in self.results:
            results_df['ObservedQ'] = self.results['observed_q']
            
        # Add model results
        results_df['SnowPack'] = self.results['snowpack']
        results_df['LiquidWater'] = self.results['liquid_water']
        results_df['RunoffFromSnow'] = self.results['runoff_from_snow']
        results_df['SoilMoisture'] = self.results['soil_moisture']
        results_df['Recharge'] = self.results['recharge']
        results_df['RunoffSoil'] = self.results['runoff_soil']
        results_df['ActualET'] = self.results['actual_et']
        results_df['UpperStorage'] = self.results['upper_storage']
        results_df['LowerStorage'] = self.results['lower_storage']
        results_df['QuickFlow'] = self.results['quick_flow']
        results_df['IntermediateFlow'] = self.results['intermediate_flow']
        results_df['Baseflow'] = self.results['baseflow']
        results_df['Discharge'] = self.results['discharge']
        
        # Save to file
        results_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")


    def save_model(self, output_path):
        """
        Save the entire model instance to a file using pickle.
        
        This saves all model components including:
        - Parameters
        - Current states
        - Input data
        - Results (if model has been run)
        - Performance metrics (if calculated)
        
        Parameters:
        -----------
        output_path : str
            Path to save the model file. Will create directories if they don't exist.
        """
        import pickle
        import os
        
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Save the model using pickle
        with open(output_path, 'wb') as f:
            pickle.dump(self, f)
        
        print(f"Model saved to {output_path}")
        
        return None

    def load_model(model_path):

        """
        Load a saved HBV model from a file.
        
        Parameters:
        -----------
        model_path : str
            Path to the saved model file.
            
        Returns:
        --------
        HBVModel
            The loaded model instance.
        """
        import pickle
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        print(f"Model loaded from {model_path}")
        
        return model
    