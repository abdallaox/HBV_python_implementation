"""
HBV Hydrological Model

This module integrates the snow, soil, and response routines into a complete
HBV hydrological model. It handles parameter management, data reading, model
execution, and output visualization.

Usage:
    from hbv_model import HBVModel
    model = HBVModel()
    model.load_data("path/to/data.csv")
    model.set_parameters(snow_params, soil_params, response_params)
    model.run()
    model.plot_results()
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import os
import datetime

from hbv_step import hbv_step

class HBVModel:
    """
    HBV hydrological model class that integrates snow, soil, and response routines.
    """
    
    def __init__(self):
        """Initialize the HBV model with default values."""
        self.data = None
        self.results = None
        self.params = {
            'snow': {
                'TT': 0.0,      # Temperature threshold for snow/rain (°C)
                'CFMAX': 3.5,   # Degree-day factor (mm/°C/day)
                'PCF': 1.0,     # Precipitation  correction factor (-)
                'SFCF': 1.0,    # Snowfall correction factor (-)
                'CFR': 0.05,    # Refreezing coefficient (-)
                'CWH': 0.1      # Water holding capacity of snow (-)
            },
            'soil': {
                'FC': 150.0,    # Field capacity (mm)
                'LP': 0.7,      # Limit for potential evaporation (-)
                'BETA': 2.0     # Shape coefficient (-)
            },
            'response': {
                'K0': 0.5,      # Quick flow recession coefficient (1/day)
                'K1': 0.2,      # Intermediate flow recession coefficient (1/day)
                'K2': 0.05,     # Baseflow recession coefficient (1/day)
                'UZL': 20.0,    # Upper zone threshold (mm)
                'PERC': 1.5     # Percolation rate (mm/day)
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
    
    def load_data(self, file_path=None, data=None, date_column='Date', 
                precip_column='Precipitation', temp_column='Temperature', 
                pet_column='PotentialET', obs_q_column=None,
                start_date=None, end_date=None):
        """
        Load data from file or pandas DataFrame, optionally between specified dates.

        Parameters:
        -----------
        file_path : str, optional
            Path to data file (CSV format).
        data : pandas.DataFrame, optional
            Data already in a DataFrame.
        date_column : str, default 'Date'
            Name of column containing dates.
        precip_column : str, default 'Precipitation'
            Name of column containing precipitation data.
        temp_column : str, default 'Temperature'
            Name of column containing temperature data.
        pet_column : str, default 'PotentialET'
            Name of column containing potential evapotranspiration data.
        obs_q_column : str, optional
            Name of column containing observed discharge data.
        start_date : str or datetime, optional
            Start date for filtering the data.
        end_date : str or datetime, optional
            End date for filtering the data.
        """
        if file_path is not None:
            data = pd.read_csv(file_path)

        # Try to convert the date column to datetime
        try:
            data[date_column] = pd.to_datetime(data[date_column])
        except Exception as e:
            print(f"Warning: Couldn't convert {date_column} to datetime. {e}")

        # --- Filter data if start_date or end_date provided ---
        if start_date is not None:
            start_date = pd.to_datetime(start_date)
            data = data[data[date_column] >= start_date]

        if end_date is not None:
            end_date = pd.to_datetime(end_date)
            data = data[data[date_column] <= end_date]

        # Store the data
        self.data = data.reset_index(drop=True)

        # Set column names
        self.column_names = {
            'date': date_column,
            'precip': precip_column,
            'temp': temp_column,
            'pet': pet_column,
            'obs_q': obs_q_column
        }

        # Save the start and end dates (IMPORTANT: based on actual data loaded)
        if date_column in data.columns and len(data) > 0:
            self.start_date = data[date_column].min()
            self.end_date = data[date_column].max()

            # Try to determine time step
            if len(data) > 1:
                diff = data[date_column].diff().dropna()
                modal_diff = diff.mode().iloc[0]
                if modal_diff == pd.Timedelta(days=1):
                    self.time_step = 'D'
                elif modal_diff == pd.Timedelta(hours=1):
                    self.time_step = 'H'
                else:
                    self.time_step = str(modal_diff)
                print(f"Time step detected: {self.time_step}")
        else:
            self.start_date = None
            self.end_date = None
            print("Warning: No dates found in data!")

        print(f"Loaded data with {len(self.data)} time steps, from {self.start_date} to {self.end_date}")
    
    def set_parameters(self, snow_params=None, soil_params=None, response_params=None):
        """
        Set model parameters.
        
        Parameters:
        -----------
        snow_params : dict, optional
            Parameters for snow routine
        soil_params : dict, optional
            Parameters for soil routine
        response_params : dict, optional
            Parameters for response routine
        """
        if snow_params:
            self.params['snow'].update(snow_params)
        if soil_params:
            self.params['soil'].update(soil_params)
        if response_params:
            self.params['response'].update(response_params)
            
        print("Parameters updated.")
    
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
    
    
    def run(self):
        """
        Run the HBV model for the entire simulation period.
        """
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
        
        # # Get initial storage values
        # snowpack = self.states['snowpack']
        # liquid_water = self.states['liquid_water']
        # soil_moisture = self.states['soil_moisture']
        # upper_storage = self.states['upper_storage']
        # lower_storage = self.states['lower_storage']
        
        print(f"Starting model run for {n_steps} time steps...")
        
        # Main simulation loop
        for t in range(n_steps):
            # # Snow routine
            # snowpack, liquid_water, runoff_from_snow = self.snow_routine(
            #     precip[t],
            #     temp[t],
            #     snowpack,
            #     liquid_water,
            #     self.params['snow']
            # )
            
            # # Soil routine
            # soil_moisture, out_to_response, recharge, runoff_soil, actual_et = self.soil_routine(
            #     runoff_from_snow,
            #     temp[t],
            #     pet[t],
            #     soil_moisture,
            #     self.params['soil']
            # )
            
            # # Response routine
            # upper_storage, lower_storage, discharge, quick_flow, intermediate_flow, baseflow = self.response_routine_two_tanks(
            #     out_to_response,
            #     upper_storage,
            #     lower_storage,
            #     self.params['response']
            # )

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
        self.results = results
        
        print("Model run completed successfully!")
        
        # Calculate performance metrics if observed discharge is available
        if obs_q is not None:
            self.calculate_performance_metrics()
        
        return results
    
    def calculate_performance_metrics(self):
        """
        Calculate performance metrics if observed discharge is available.
        """
        if 'observed_q' not in self.results:
            print("No observed discharge data available for performance evaluation.")
            return
            
        # Get simulated and observed discharge
        sim_q = self.results['discharge']
        obs_q = self.results['observed_q']
        
        # Remove NaN values
        valid_idx = ~np.isnan(obs_q)
        if np.sum(valid_idx) == 0:
            print("No valid observed discharge values found.")
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
        alpha = std_sim / std_obs  # Relative variability
        beta = mean_sim / mean_obs  # Bias
        
        kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
        
        # Calculate percent bias
        pbias = 100 * (np.sum(sim_q_valid - obs_q_valid) / np.sum(obs_q_valid))
        
        # Store metrics
        self.performance_metrics = {
            'NSE': nse,
            'KGE': kge,
            'PBIAS': pbias,
            'r': r
        }
        
        print(f"Performance metrics calculated:")
        print(f"NSE: {nse:.3f}")
        print(f"KGE: {kge:.3f}")
        print(f"PBIAS: {pbias:.1f}%")
        print(f"Correlation: {r:.3f}")
    
    def plot_results(self, output_file=None, show_plots=True):
        """
        Plot model results with customized layout and additional figures.
        
        Parameters:
        -----------
        output_file : str, optional
            If provided, save the plot to this file
        show_plots : bool, default True
            Whether to display the plots
        """
        if self.results is None:
            raise ValueError("No results to plot. Run the model first.")
            
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(12, 28))  # Increased height for more subplots
        
        # Define subplots layout (now 9 subplots)
        axs = []
        axs.append(fig.add_subplot(10, 1, 1))  # Precipitation
        axs.append(fig.add_subplot(10, 1, 2))  # Temperature
        axs.append(fig.add_subplot(10, 1, 3))  # Snow pack and liquid water
        axs.append(fig.add_subplot(10, 1, 4))  # Runoff from snow
        axs.append(fig.add_subplot(10, 1, 5))  # Potential and actual ET
        axs.append(fig.add_subplot(10, 1, 6))  # Soil moisture
        axs.append(fig.add_subplot(10, 1, 7))  # Recharge (output from soil to response)
        axs.append(fig.add_subplot(10, 1, 8))  # Groundwater storages
        axs.append(fig.add_subplot(10, 1, 9))  # Discharge components and total
        axs.append(fig.add_subplot(10, 1, 10))  # Discharge components and total

        # Get dates for x-axis
        dates = self.results['dates']
        if isinstance(dates[0], (datetime.datetime, np.datetime64)):
            date_formatter = DateFormatter('%Y-%m-%d')
            is_datetime = True
        else:
            is_datetime = False
        
        # 1. Precipitation
        ax1 = axs[0]
        ax1.bar(dates, self.results['precipitation'], color='skyblue', label='Precipitation')
        ax1.set_ylabel('Precipitation (mm)')
        ax1.set_title('Precipitation')
        ax1.legend(loc='upper right')
        
        # 2. Temperature with TT threshold
        ax2 = axs[1]
        ax2.plot(dates, self.results['temperature'], color='red', label='Temperature')
        ax2.axhline(y=self.params['snow']['TT'], color='gray', linestyle='--', 
                label=f"TT Threshold ({self.params['snow']['TT']}°C)")
        ax2.set_ylabel('Temperature (°C)')
        ax2.set_title('Temperature with Snow Threshold')
        ax2.legend(loc='upper right')
        
        # 3. Snow pack and liquid water
        ax3 = axs[2]
        ax3.plot(dates, self.results['snowpack'], color='blue', label='Snow Pack')
        ax3.plot(dates, self.results['liquid_water'], color='lightblue', label='Liquid Water')
        ax3.set_ylabel('Water (mm)')
        ax3.set_title('Snow Pack and Liquid Water Content')
        ax3.legend(loc='upper right')
        
        # 4. Runoff from snow
        ax4 = axs[3]
        ax4.plot(dates, self.results['runoff_from_snow'], color='skyblue', label='Runoff from Snow')
        ax4.set_ylabel('Runoff (mm/day)')
        ax4.set_title('Runoff from Snow')
        ax4.legend(loc='upper right')
        
        # 5. Potential and actual ET
        ax5 = axs[4]
        ax5.plot(dates, self.results['potential_et'], color='orange', label='Potential ET')
        ax5.plot(dates, self.results['actual_et'], color='green', label='Actual ET')
        ax5.set_ylabel('ET (mm/day)')
        ax5.set_title('Potential and Actual Evapotranspiration')
        ax5.legend(loc='upper right')
        
        # 6. Soil moisture
        ax6 = axs[5]
        ax6.plot(dates, self.results['soil_moisture'], color='brown', label='Soil Moisture')
        ax6.axhline(y=self.params['soil']['FC'], color='gray', linestyle='--', 
                label=f"Field Capacity ({self.params['soil']['FC']} mm)")
        ax6.set_ylabel('Soil Moisture (mm)')
        ax6.set_title('Soil Moisture')
        ax6.legend(loc='upper right')
        
        # 7. Recharge (output from soil to response routine)
        ax7 = axs[6]
        ax7.plot(dates, self.results['recharge'], color='purple',linewidth=0.5, label='Recharge')
        ax7.plot(dates, self.results['runoff_soil'], color='red',linewidth=0.5, label='Runoff (overflow from soil))')
        # ax7.plot(dates, self.results['recharge'] + self.results['runoff_soil'], 
        #         color='darkviolet', linestyle='--', linewidth=1, label='Total to Response')
        ax7.set_ylabel('Water (mm/day)')
        ax7.set_title('Soil Output to Response Routine')
        ax7.legend(loc='upper right')
        
        # 8. Groundwater storages
        ax8 = axs[7]
        ax8.plot(dates, self.results['upper_storage'], color='lightcoral', label='Upper Storage')
        ax8.plot(dates, self.results['lower_storage'], color='darkblue', label='Lower Storage')
        ax8.axhline(y=self.params['response']['UZL'], color='gray', linestyle='--', label='Upper Zone Threshold')
        ax8.set_ylabel('Storage (mm)')
        ax8.set_title('Groundwater Storages')
        ax8.legend(loc='upper right')
        
        # 9. Discharge Components (Stacked)
        ax9 = axs[8]
        
        # Stackplot with components only
        ax9.stackplot(dates,
                    self.results['baseflow'],
                    self.results['intermediate_flow'],
                    self.results['quick_flow'],
                    labels=['Baseflow', 'Intermediate Flow', 'Quick Flow'],
                    colors=['royalblue', 'darkorange', 'tomato'])
        
        # Add total discharge line
        ax9.plot(dates, self.results['discharge'], 
                color='black', linestyle=':', linewidth=0.5,
                label='Total Discharge (sum)')
        
        ax9.set_ylabel('Flow (mm/day)')
        ax9.set_title('Runoff Components (Stacked)')
        ax9.legend(loc='upper right')
        
        # 10. Discharge Comparison (Total vs Observed)
        ax10 = axs[9]
        
        # Plot simulated total
        ax10.plot(dates, self.results['discharge'], 
                color='darkgreen', linewidth=2,
                label='Simulated Discharge')
        
        # Plot observed if available
        if 'observed_q' in self.results:
            ax10.plot(dates, self.results['observed_q'], 
                    color='black', linestyle='--', linewidth=1.5,
                    label='Observed Discharge')
            
            # Add performance metrics to title
            if hasattr(self, 'performance_metrics'):
                metrics = self.performance_metrics
                ax10.set_title(
                    f"Discharge Comparison (NSE: {metrics['NSE']:.2f}, KGE: {metrics['KGE']:.2f})"
                )
            else: ax10.set_title('Discharge Comparison')
        
        ax10.set_ylabel('Discharge (mm/day)')
        ax10.set_xlabel('Date')
        
        ax10.legend(loc='upper right')
        
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