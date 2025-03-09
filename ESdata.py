import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import time
import multiprocessing as mp
from functools import partial

# Try to import GPU acceleration libraries
try:
    import cupy as cp
    GPU_AVAILABLE = True
    CUDF_AVAILABLE = False
    print("GPU acceleration enabled using CuPy")
    
    # Check if cuDF is also available, but don't require it
    try:
        import cudf
        CUDF_AVAILABLE = True
        print("RAPIDS cuDF library also available")
    except ImportError:
        CUDF_AVAILABLE = False
        print("CuPy available, but RAPIDS cuDF library not available. Using pandas for dataframes.")
except ImportError:
    GPU_AVAILABLE = False
    CUDF_AVAILABLE = False
    print("GPU acceleration libraries (CuPy) not available, using CPU")

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.ticker import FuncFormatter
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    
try:
    import itertools
    from tqdm import tqdm
except ImportError:
    pass

# Default parameters
default_params = {
    "cluster_levels": [0.2, 0.1, 0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9],
    "analysis_mode": "long",
    "selected_days": None,
    "use_wdr_filter": False,
    "use_m7b_filter": True,
    "long_confirm_start": "10:30",
    "long_confirm_end": "10:55",
    "short_confirm_start": "10:30",
    "short_confirm_end": "10:50",
    "tp_multiple": 1.0,
    "sl_distance": 0.1,
    "sl_extra_ticks": 2,
    "min_trades": 5,
    "top_configs": 10,
    "target_win_rate": 50.0,
    "target_expectancy": 1.5,
    "show_raw_data": True,
    "show_overall_stats": True,
    "show_day_performance": True,
    "show_top_configs": True,
    "show_target_configs": True,
    "show_filter_stats": True,
    "calculate_profit_factor": False,
    "analyze_streaks": False,
    "perform_param_sweep": False,
    "perform_day_by_day_sweep": True,  # New parameter for day-by-day parameter sweep
    "perform_walk_forward": False,
    "generate_equity_curve": False,
    "use_gpu": True,      # New parameter for enabling GPU acceleration
    "use_multiprocessing": True,  # New parameter for enabling multiprocessing
    "data_file": "es_data.txt",
    "output_dir": "results",
    "tick_size": 0.25
}

# =========================== INTERACTIVE INTERFACE FUNCTIONS ===========================

def get_user_input():
    """Interactive parameter input from the user"""
    print("\n" + "=" * 60)
    print(" " * 15 + "ES TRADING STRATEGY CONFIGURATION")
    print("=" * 60)
    
    params = default_params.copy()
    
    # Main configuration categories
    categories = [
        "Core Strategy Parameters",
        "Filtering Options",
        "Time Windows",
        "Trade Parameters",
        "Analysis Parameters",
        "Output Controls",
        "Advanced Analysis Options",
        "Performance Options",  # New category
        "File Options",
        "Run with current settings"
    ]
    
    while True:
        print("\nConfiguration Categories:")
        for i, category in enumerate(categories, 1):
            print(f"{i}. {category}")
        
        try:
            choice = int(input("\nSelect a category (1-10): "))
            if not 1 <= choice <= len(categories):
                print("Invalid selection. Please try again.")
                continue
            
            # If user selects "Run with current settings"
            if choice == len(categories):
                break
                
            # Process each category
            if choice == 1:  # Core Strategy Parameters
                print("\n--- Core Strategy Parameters ---")
                
                # Analysis mode
                print(f"\nCurrent analysis mode: {params['analysis_mode']}")
                mode_choice = input("Select analysis mode (long/short/both) [leave blank to keep current]: ")
                if mode_choice in ["long", "short", "both"]:
                    params["analysis_mode"] = mode_choice
                
                # Cluster levels
                print(f"\nCurrent cluster levels: {params['cluster_levels']}")
                cluster_input = input("Enter cluster levels separated by spaces [leave blank to keep current]: ")
                if cluster_input.strip():
                    try:
                        params["cluster_levels"] = [float(x) for x in cluster_input.split()]
                        print(f"New cluster levels: {params['cluster_levels']}")
                    except ValueError:
                        print("Invalid input. Keeping current cluster levels.")
                
                # Selected days
                weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
                current_days = params["selected_days"] if params["selected_days"] else "All days"
                print(f"\nCurrent selected days: {current_days}")
                print("Select days of week to analyze:")
                print("1. All days")
                print("2. Specific days")
                days_choice = input("Enter choice (1-2) [leave blank to keep current]: ")
                
                if days_choice == "1":
                    params["selected_days"] = None
                elif days_choice == "2":
                    selected = []
                    for day in weekdays:
                        include = input(f"Include {day}? (y/n): ").lower()
                        if include == "y":
                            selected.append(day)
                    if selected:
                        params["selected_days"] = selected
                        print(f"Selected days: {selected}")
                    else:
                        print("No days selected. Using all days.")
                        params["selected_days"] = None
            
            elif choice == 2:  # Filtering Options
                print("\n--- Filtering Options ---")
                
                # WDR filter
                print(f"Current WDR filter status: {'Enabled' if params['use_wdr_filter'] else 'Disabled'}")
                wdr_choice = input("Enable WDR filter? (y/n) [leave blank to keep current]: ").lower()
                if wdr_choice in ["y", "n"]:
                    params["use_wdr_filter"] = (wdr_choice == "y")
                
                # M7B filter
                print(f"Current M7B filter status: {'Enabled' if params['use_m7b_filter'] else 'Disabled'}")
                m7b_choice = input("Enable M7B filter? (y/n) [leave blank to keep current]: ").lower()
                if m7b_choice in ["y", "n"]:
                    params["use_m7b_filter"] = (m7b_choice == "y")
            
            elif choice == 3:  # Time Windows
                print("\n--- Time Windows ---")
                
                # Long confirmation window
                print(f"Current long confirmation window: {params['long_confirm_start']} to {params['long_confirm_end']}")
                long_start = input("Long confirmation start time (HH:MM) [leave blank to keep current]: ")
                if long_start.strip():
                    params["long_confirm_start"] = long_start
                
                long_end = input("Long confirmation end time (HH:MM) [leave blank to keep current]: ")
                if long_end.strip():
                    params["long_confirm_end"] = long_end
                
                # Short confirmation window
                print(f"Current short confirmation window: {params['short_confirm_start']} to {params['short_confirm_end']}")
                short_start = input("Short confirmation start time (HH:MM) [leave blank to keep current]: ")
                if short_start.strip():
                    params["short_confirm_start"] = short_start
                
                short_end = input("Short confirmation end time (HH:MM) [leave blank to keep current]: ")
                if short_end.strip():
                    params["short_confirm_end"] = short_end
            
            elif choice == 4:  # Trade Parameters
                print("\n--- Trade Parameters ---")
                
                # TP multiple
                print(f"Current TP multiple: {params['tp_multiple']}")
                tp_input = input("TP multiple of SD [leave blank to keep current]: ")
                if tp_input.strip():
                    try:
                        params["tp_multiple"] = float(tp_input)
                    except ValueError:
                        print("Invalid input. Keeping current value.")
                
                # SL distance
                print(f"Current SL distance: {params['sl_distance']}")
                sl_input = input("SL distance as multiple of SD [leave blank to keep current]: ")
                if sl_input.strip():
                    try:
                        params["sl_distance"] = float(sl_input)
                    except ValueError:
                        print("Invalid input. Keeping current value.")
                
                # SL extra ticks
                print(f"Current SL extra ticks: {params['sl_extra_ticks']}")
                ticks_input = input("Additional ticks to add to stop loss [leave blank to keep current]: ")
                if ticks_input.strip():
                    try:
                        params["sl_extra_ticks"] = int(ticks_input)
                    except ValueError:
                        print("Invalid input. Keeping current value.")
            
            elif choice == 5:  # Analysis Parameters
                print("\n--- Analysis Parameters ---")
                
                # Minimum trades
                print(f"Current minimum trades for config analysis: {params['min_trades']}")
                min_trades_input = input("Minimum trades for a configuration to be considered [leave blank to keep current]: ")
                if min_trades_input.strip():
                    try:
                        params["min_trades"] = int(min_trades_input)
                    except ValueError:
                        print("Invalid input. Keeping current value.")
                
                # Top configs
                print(f"Current number of top configs to show: {params['top_configs']}")
                top_configs_input = input("Number of top configurations to display [leave blank to keep current]: ")
                if top_configs_input.strip():
                    try:
                        params["top_configs"] = int(top_configs_input)
                    except ValueError:
                        print("Invalid input. Keeping current value.")
                
                # Target win rate
                print(f"Current target win rate: {params['target_win_rate']}%")
                win_rate_input = input("Target win rate for filtering (%) [leave blank to keep current]: ")
                if win_rate_input.strip():
                    try:
                        params["target_win_rate"] = float(win_rate_input)
                    except ValueError:
                        print("Invalid input. Keeping current value.")
                
                # Target expectancy
                print(f"Current target expectancy: {params['target_expectancy']}R")
                expectancy_input = input("Target expectancy for filtering (R) [leave blank to keep current]: ")
                if expectancy_input.strip():
                    try:
                        params["target_expectancy"] = float(expectancy_input)
                    except ValueError:
                        print("Invalid input. Keeping current value.")
            
            elif choice == 6:  # Output Controls
                print("\n--- Output Controls ---")
                output_options = [
                    ("show_raw_data", "Show raw configuration data"),
                    ("show_overall_stats", "Show overall statistics"),
                    ("show_day_performance", "Show day of week performance"),
                    ("show_top_configs", "Show top configurations"),
                    ("show_target_configs", "Show configurations meeting target criteria"),
                    ("show_filter_stats", "Show filter statistics")
                ]
                
                print("Toggle output sections:")
                for i, (param_name, description) in enumerate(output_options, 1):
                    status = "Enabled" if params[param_name] else "Disabled"
                    print(f"{i}. {description}: {status}")
                
                option_choice = input("\nSelect option to toggle (1-6) or 'a' for all, 'n' for none [leave blank to keep current]: ")
                if option_choice.strip():
                    if option_choice.lower() == "a":
                        for param_name, _ in output_options:
                            params[param_name] = True
                        print("All output sections enabled.")
                    elif option_choice.lower() == "n":
                        for param_name, _ in output_options:
                            params[param_name] = False
                        print("All output sections disabled.")
                    elif option_choice.isdigit() and 1 <= int(option_choice) <= len(output_options):
                        idx = int(option_choice) - 1
                        param_name = output_options[idx][0]
                        params[param_name] = not params[param_name]
                        status = "Enabled" if params[param_name] else "Disabled"
                        print(f"{output_options[idx][1]}: {status}")
            
            elif choice == 7:  # Advanced Analysis Options
                if not MATPLOTLIB_AVAILABLE and (params['generate_equity_curve'] or params['perform_param_sweep'] or params['perform_walk_forward']):
                    print("\nWARNING: Matplotlib not available. Visualization features will be disabled.")
                
                print("\n--- Advanced Analysis Options ---")
                advanced_options = [
                    ("calculate_profit_factor", "Calculate profit factor (gross profits/gross losses)"),
                    ("analyze_streaks", "Analyze winning and losing streaks"),
                    ("perform_param_sweep", "Perform parameter sweep to find optimal settings"),
                    ("perform_day_by_day_sweep", "Analyze parameter sweep separately for each day"),
                    ("perform_walk_forward", "Perform walk-forward analysis to test strategy robustness"),
                    ("generate_equity_curve", "Generate equity curve charts")
                ]
                
                print("Toggle advanced analysis options:")
                for i, (param_name, description) in enumerate(advanced_options, 1):
                    status = "Enabled" if params[param_name] else "Disabled"
                    print(f"{i}. {description}: {status}")
                
                option_choice = input(f"\nSelect option to toggle (1-{len(advanced_options)}) or 'a' for all, 'n' for none [leave blank to keep current]: ")
                if option_choice.strip():
                    if option_choice.lower() == "a":
                        for param_name, _ in advanced_options:
                            params[param_name] = True
                        print("All advanced analysis options enabled.")
                    elif option_choice.lower() == "n":
                        for param_name, _ in advanced_options:
                            params[param_name] = False
                        print("All advanced analysis options disabled.")
                    elif option_choice.isdigit() and 1 <= int(option_choice) <= len(advanced_options):
                        idx = int(option_choice) - 1
                        param_name = advanced_options[idx][0]
                        params[param_name] = not params[param_name]
                        status = "Enabled" if params[param_name] else "Disabled"
                        print(f"{advanced_options[idx][1]}: {status}")
            
            elif choice == 8:  # Performance Options
                print("\n--- Performance Options ---")
                
                # GPU acceleration
                if GPU_AVAILABLE:
                    print(f"Current GPU acceleration status: {'Enabled' if params['use_gpu'] else 'Disabled'}")
                    gpu_choice = input("Enable GPU acceleration? (y/n) [leave blank to keep current]: ").lower()
                    if gpu_choice in ["y", "n"]:
                        params["use_gpu"] = (gpu_choice == "y")
                else:
                    print("GPU acceleration not available on this system (RAPIDS/CuPy not installed)")
                    params["use_gpu"] = False
                
                # Multiprocessing
                print(f"Current multiprocessing status: {'Enabled' if params['use_multiprocessing'] else 'Disabled'}")
                mp_choice = input("Enable multiprocessing for faster calculations? (y/n) [leave blank to keep current]: ").lower()
                if mp_choice in ["y", "n"]:
                    params["use_multiprocessing"] = (mp_choice == "y")
            
            elif choice == 9:  # File Options
                print("\n--- File Options ---")
                
                # Data file
                print(f"Current data file: {params['data_file']}")
                data_file = input("Input data file path [leave blank to keep current]: ")
                if data_file.strip():
                    params["data_file"] = data_file
                
                # Output directory
                print(f"Current output directory: {params['output_dir']}")
                output_dir = input("Directory to save output files [leave blank to keep current]: ")
                if output_dir.strip():
                    params["output_dir"] = output_dir
        
        except (ValueError, IndexError):
            print("Invalid input. Please try again.")
    
    # Final confirmation
    print("\n" + "=" * 60)
    print("CURRENT CONFIGURATION SUMMARY:")
    print("=" * 60)
    print(f"Analysis Mode: {params['analysis_mode']}")
    print(f"Selected Days: {params['selected_days'] if params['selected_days'] else 'All days'}")
    print(f"Cluster Levels: {params['cluster_levels']}")
    print(f"TP Multiple: {params['tp_multiple']}")
    print(f"SL Distance: {params['sl_distance']}")
    print(f"SL Extra Ticks: {params['sl_extra_ticks']}")
    print(f"WDR Filter: {'Enabled' if params['use_wdr_filter'] else 'Disabled'}")
    print(f"M7B Filter: {'Enabled' if params['use_m7b_filter'] else 'Disabled'}")
    print(f"GPU Acceleration: {'Enabled' if params['use_gpu'] and GPU_AVAILABLE else 'Disabled'}")
    print(f"Multiprocessing: {'Enabled' if params['use_multiprocessing'] else 'Disabled'}")
    print(f"Long Confirmation Window: {params['long_confirm_start']} to {params['long_confirm_end']}")
    print(f"Short Confirmation Window: {params['short_confirm_start']} to {params['short_confirm_end']}")
    
    confirm = input("\nRun analysis with these settings? (y/n): ").lower()
    if confirm != "y":
        print("Aborted by user.")
        sys.exit()
    
    return params

# =========================== UTILITY FUNCTIONS ===========================

def calculate_expectancy(valid_trades):
    """
    Calculate expectancy consistently based on profitable vs unprofitable trades.
    
    Parameters:
    valid_trades (DataFrame): DataFrame containing trade results with pnl_R column
    
    Returns:
    tuple: (win_rate, avg_winner, avg_loser, expectancy)
    """
    if valid_trades.empty:
        return 0, 0, 0, 0
    
    # Define winners and losers based on profit/loss
    winners = valid_trades[valid_trades["pnl_R"] > 0]
    losers = valid_trades[valid_trades["pnl_R"] <= 0]
    
    # Calculate win rate based on profitability (not TP hits)
    win_rate = (len(winners) / len(valid_trades)) * 100 if len(valid_trades) > 0 else 0
    
    # Calculate average winner and loser
    avg_winner = winners["pnl_R"].mean() if not winners.empty else 0
    avg_loser = losers["pnl_R"].mean() if not losers.empty else 0
    
    # Proper expectancy calculation
    expectancy = (win_rate/100) * avg_winner + (1 - win_rate/100) * avg_loser
    
    return win_rate, avg_winner, avg_loser, expectancy

def calculate_expectancy_gpu(valid_trades):
    """
    GPU-accelerated version of calculate_expectancy using CuPy
    """
    if valid_trades.empty:
        return 0, 0, 0, 0
    
    # Convert pnl_R to CuPy array for faster computation
    pnl_array = cp.array(valid_trades["pnl_R"].values)
    
    # Define winners and losers
    winners_mask = pnl_array > 0
    losers_mask = ~winners_mask
    
    # Calculate win rate
    win_rate = (cp.sum(winners_mask) / len(pnl_array)) * 100
    
    # Calculate average winner and loser
    winners = pnl_array[winners_mask]
    losers = pnl_array[losers_mask]
    
    avg_winner = cp.mean(winners).item() if len(winners) > 0 else 0
    avg_loser = cp.mean(losers).item() if len(losers) > 0 else 0
    
    # Expectancy calculation
    expectancy = (win_rate/100) * avg_winner + (1 - win_rate/100) * avg_loser
    
    return win_rate.item(), avg_winner, avg_loser, expectancy

def calculate_profit_factor(valid_trades):
    """Calculate profit factor (gross profits / gross losses)"""
    if valid_trades.empty:
        return 0
    
    gross_profits = valid_trades[valid_trades["pnl_R"] > 0]["pnl_R"].sum()
    gross_losses = abs(valid_trades[valid_trades["pnl_R"] < 0]["pnl_R"].sum())
    
    profit_factor = gross_profits / gross_losses if gross_losses != 0 else float('inf')
    return profit_factor

def analyze_streaks(valid_trades):
    """Analyze winning and losing streaks"""
    if valid_trades.empty:
        return None, None, None, None, None
    
    # Sort trades by date and time
    sorted_trades = valid_trades.sort_values(['date', 'confirm_time'])
    
    # Create list of wins (1) and losses (0)
    results = [1 if r > 0 else 0 for r in sorted_trades['pnl_R'].values]
    
    # Find streaks
    win_streaks = []
    loss_streaks = []
    
    current_streak = 1
    current_type = results[0] if len(results) > 0 else None
    
    for i in range(1, len(results)):
        if results[i] == results[i-1]:
            current_streak += 1
        else:
            if current_type == 1:
                win_streaks.append(current_streak)
            else:
                loss_streaks.append(current_streak)
            current_streak = 1
            current_type = results[i]
    
    # Add the last streak
    if current_type == 1:
        win_streaks.append(current_streak)
    elif current_type == 0:
        loss_streaks.append(current_streak)
    
    # Calculate streak statistics
    max_win_streak = max(win_streaks) if win_streaks else 0
    max_loss_streak = max(loss_streaks) if loss_streaks else 0
    avg_win_streak = sum(win_streaks) / len(win_streaks) if win_streaks else 0
    avg_loss_streak = sum(loss_streaks) / len(loss_streaks) if loss_streaks else 0
    
    return max_win_streak, max_loss_streak, avg_win_streak, avg_loss_streak, win_streaks + loss_streaks

def generate_equity_curve(valid_trades, output_dir, title):
    """Generate equity curve chart from trade data"""
    if valid_trades.empty:
        print("No valid trades to generate equity curve")
        return
    
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Cannot generate equity curve.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort trades by date and time
    sorted_trades = valid_trades.sort_values(['date', 'confirm_time'])
    
    # Calculate cumulative R
    sorted_trades['cumulative_R'] = sorted_trades['pnl_R'].cumsum()
    
    # Convert date to datetime for plotting
    sorted_trades['datetime'] = pd.to_datetime(sorted_trades['date']) + pd.to_timedelta(
        sorted_trades['confirm_time'].dt.hour * 3600 + 
        sorted_trades['confirm_time'].dt.minute * 60 + 
        sorted_trades['confirm_time'].dt.second, unit='s')
    
    # Create figure
    plt.figure(figsize=(12, 6))
    plt.plot(sorted_trades['datetime'], sorted_trades['cumulative_R'], 
             linestyle='-', marker='o', markersize=3)
    
    # Add moving average if enough points
    if len(sorted_trades) > 20:
        window = min(20, len(sorted_trades) // 5)
        sorted_trades['ma'] = sorted_trades['cumulative_R'].rolling(window=window).mean()
        plt.plot(sorted_trades['datetime'], sorted_trades['ma'], 'r--', 
                 label=f'{window}-Trade Moving Average')
    
    # Add drawdown
    running_max = sorted_trades['cumulative_R'].expanding().max()
    drawdown = sorted_trades['cumulative_R'] - running_max
    max_drawdown = drawdown.min()
    max_dd_idx = drawdown.idxmin()
    
    # Format the plot
    plt.grid(True, alpha=0.3)
    plt.title(f'Equity Curve - {title}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative R')
    
    # Format x-axis dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    
    # Add annotations
    total_r = sorted_trades['pnl_R'].sum()
    profit_factor = calculate_profit_factor(sorted_trades)
    win_rate, _, _, expectancy = calculate_expectancy(sorted_trades)
    
    annotation = (f'Total R: {total_r:.2f}\n'
                  f'Max Drawdown: {max_drawdown:.2f}R\n'
                  f'Profit Factor: {profit_factor:.2f}\n'
                  f'Win Rate: {win_rate:.1f}%\n'
                  f'Expectancy: {expectancy:.2f}R')
    
    plt.annotate(annotation, xy=(0.02, 0.02), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
    
    if len(sorted_trades) > 20:
        plt.legend()
    
    # Mark maximum drawdown
    if max_dd_idx in sorted_trades.index:
        plt.plot(sorted_trades.loc[max_dd_idx, 'datetime'], 
                 sorted_trades.loc[max_dd_idx, 'cumulative_R'], 
                 'rv', markersize=8, label='Max Drawdown')
    
    # Save the chart
    filename = f"{output_dir}/equity_curve_{title.replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Equity curve saved to {filename}")
    
    return max_drawdown

# Function to process a single parameter combination
def process_parameter_combo(combo_args):
    """Process a single parameter combination for parameter sweep"""
    tp_mult, sl_ex_ticks, df_subset, test_clusters, base_params = combo_args
    
    # Create parameters for this run
    sweep_params = base_params.copy()
    sweep_params['tp_multiple'] = tp_mult
    sweep_params['sl_distance'] = 0.1  # Fixed SL distance
    sweep_params['sl_extra_ticks'] = sl_ex_ticks
    
    # Run backtest with these parameters
    results = process_all_sessions(df_subset, test_clusters, sweep_params)
    results_df = pd.DataFrame(results)
    
    # Return empty result if no valid trades
    if results_df.empty or len(results_df[results_df["trade_entry"].notna()]) == 0:
        return {
            'tp_multiple': tp_mult,
            'sl_distance': 0.1,
            'sl_extra_ticks': sl_ex_ticks,
            'trade_count': 0,
            'win_rate': 0,
            'total_r': 0,
            'expectancy': 0,
            'profit_factor': 0
        }
    
    # Get performance metrics
    filtered_valid_trades = results_df[results_df["trade_entry"].notna()]
    
    if not filtered_valid_trades.empty:
        win_rate, avg_winner, avg_loser, expectancy = calculate_expectancy(filtered_valid_trades)
        total_r = filtered_valid_trades["pnl_R"].sum()
        profit_factor = calculate_profit_factor(filtered_valid_trades)
        trade_count = len(filtered_valid_trades)
        
        return {
            'tp_multiple': tp_mult,
            'sl_distance': 0.1,  # Fixed value
            'sl_extra_ticks': sl_ex_ticks,
            'trade_count': trade_count,
            'win_rate': win_rate,
            'total_r': total_r,
            'expectancy': expectancy,
            'profit_factor': profit_factor
        }
    else:
        return {
            'tp_multiple': tp_mult,
            'sl_distance': 0.1,
            'sl_extra_ticks': sl_ex_ticks,
            'trade_count': 0,
            'win_rate': 0,
            'total_r': 0,
            'expectancy': 0,
            'profit_factor': 0
        }

def perform_parameter_sweep(df, params):
    """
    Perform parameter sweep to find optimal parameter settings
    """
    print("\n" + "=" * 100)
    print(" " * 35 + "PARAMETER SWEEP ANALYSIS")
    print("=" * 100)
    
    # Define expanded parameter ranges to sweep
    tp_multiples = [0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 1.8, 2.0, 2.5]
    # Keep SL distance fixed at 0.1 as requested
    sl_distance_fixed = 0.1
    sl_extra_ticks_list = [1, 2, 3]
    
    # If cluster_levels are provided, use a subset for faster processing
    test_clusters = [params['cluster_levels'][0], params['cluster_levels'][len(params['cluster_levels'])//2], params['cluster_levels'][-1]]
    
    # Print parameter ranges
    print(f"Testing {len(tp_multiples) * len(sl_extra_ticks_list)} parameter combinations...")
    print(f"TP Multiples: {tp_multiples}")
    print(f"SL Distance: Fixed at {sl_distance_fixed}")
    print(f"SL Extra Ticks: {sl_extra_ticks_list}")
    print(f"Test Clusters: {test_clusters}")
    
    # Check if day-by-day analysis is enabled
    if params['perform_day_by_day_sweep']:
        return perform_day_by_day_sweep(df, params, tp_multiples, sl_extra_ticks_list, test_clusters)
    
    # Create combinations with fixed SL distance
    from itertools import product
    param_combinations = list(product(tp_multiples, sl_extra_ticks_list))
    
    total_combinations = len(param_combinations)
    print(f"Testing {total_combinations} parameter combinations...")
    
    # Use GPU acceleration if enabled and available
    if params['use_gpu'] and GPU_AVAILABLE:
        print("Using GPU acceleration for calculations...")
        # GPU implementation would go here, but since it requires custom CUDA kernels
        # We'll fall back to multiprocessing in this version
        
    # Use multiprocessing if enabled
    sweep_results = []
    if params['use_multiprocessing']:
        # Create a partial function with fixed arguments
        combo_args_list = [(tp, sl, df, test_clusters, params) for tp, sl in param_combinations]
        
        # Get number of processes
        num_processes = min(mp.cpu_count(), total_combinations)
        print(f"Using {num_processes} CPU cores for parallel processing...")
        
        # Create pool and map function to combinations
        with mp.Pool(processes=num_processes) as pool:
            # Process combinations with progress reporting
            results_iter = pool.imap(process_parameter_combo, combo_args_list)
            
            # Process results with progress bar
            for i, result in enumerate(results_iter):
                progress = (i + 1) / total_combinations * 100
                sys.stdout.write(f"\rProgress: {progress:.1f}% - Processed {i+1}/{total_combinations} combinations")
                sys.stdout.flush()
                sweep_results.append(result)
            
            print()  # New line after progress
    else:
        # Process combinations sequentially
        for i, (tp_mult, sl_ex_ticks) in enumerate(param_combinations):
            # Print progress
            progress = (i + 1) / total_combinations * 100
            sys.stdout.write(f"\rProgress: {progress:.1f}% - Testing TP={tp_mult}, SL={sl_distance_fixed}, Extra={sl_ex_ticks}")
            sys.stdout.flush()
            
            # Process this combination
            result = process_parameter_combo((tp_mult, sl_ex_ticks, df, test_clusters, params))
            sweep_results.append(result)
        
        print()  # New line after progress bar
    
    # Convert to DataFrame for analysis
    sweep_df = pd.DataFrame(sweep_results)
    
    if sweep_df.empty or sweep_df['trade_count'].sum() == 0:
        print("No valid parameter combinations found")
        return
    
    # Sort by different metrics and show top results
    print("\nTop 5 Parameter Sets by Total R:")
    print(sweep_df.sort_values('total_r', ascending=False).head(5).to_string(index=False))
    
    print("\nTop 5 Parameter Sets by Expectancy:")
    print(sweep_df.sort_values('expectancy', ascending=False).head(5).to_string(index=False))
    
    print("\nTop 5 Parameter Sets by Profit Factor:")
    print(sweep_df.sort_values('profit_factor', ascending=False).head(5).to_string(index=False))
    
    # Find optimal parameters (by expectancy)
    optimal = sweep_df.loc[sweep_df['expectancy'].idxmax()]
    
    print("\nOptimal Parameter Set (by Expectancy):")
    print(f"TP Multiple: {optimal['tp_multiple']}")
    print(f"SL Distance: {optimal['sl_distance']}")
    print(f"SL Extra Ticks: {int(optimal['sl_extra_ticks'])}")
    print(f"Resulting in:")
    print(f"  Trade Count: {int(optimal['trade_count'])}")
    print(f"  Win Rate: {optimal['win_rate']:.2f}%")
    print(f"  Total R: {optimal['total_r']:.2f}")
    print(f"  Expectancy: {optimal['expectancy']:.2f}R")
    print(f"  Profit Factor: {optimal['profit_factor']:.2f}")
    
    # Create heat map visualizations if matplotlib is available
    if MATPLOTLIB_AVAILABLE:
        try:
            plt.figure(figsize=(15, 10))
            
            # Group by TP and Extra Ticks, average the metrics for different visualizations
            # Expectancy heatmap
            plt.subplot(2, 2, 1)
            pivot_data = sweep_df.pivot_table(index='sl_extra_ticks', columns='tp_multiple', values='expectancy', aggfunc='mean')
            sns_heatmap = plt.imshow(pivot_data, cmap='RdYlGn')
            plt.colorbar(sns_heatmap, label='Expectancy')
            plt.title('Expectancy by TP and Extra Ticks')
            plt.xlabel('TP Multiple')
            plt.ylabel('SL Extra Ticks')
            plt.xticks(range(len(tp_multiples)), labels=tp_multiples)
            plt.yticks(range(len(sl_extra_ticks_list)), labels=sl_extra_ticks_list)
            
            # Win rate heatmap
            plt.subplot(2, 2, 2)
            pivot_data = sweep_df.pivot_table(index='sl_extra_ticks', columns='tp_multiple', values='win_rate', aggfunc='mean')
            sns_heatmap = plt.imshow(pivot_data, cmap='Blues')
            plt.colorbar(sns_heatmap, label='Win Rate %')
            plt.title('Win Rate by TP and Extra Ticks')
            plt.xlabel('TP Multiple')
            plt.ylabel('SL Extra Ticks')
            plt.xticks(range(len(tp_multiples)), labels=tp_multiples)
            plt.yticks(range(len(sl_extra_ticks_list)), labels=sl_extra_ticks_list)
            
            # Total R heatmap
            plt.subplot(2, 2, 3)
            pivot_data = sweep_df.pivot_table(index='sl_extra_ticks', columns='tp_multiple', values='total_r', aggfunc='sum')
            sns_heatmap = plt.imshow(pivot_data, cmap='plasma')
            plt.colorbar(sns_heatmap, label='Total R')
            plt.title('Total R by TP and Extra Ticks')
            plt.xlabel('TP Multiple')
            plt.ylabel('SL Extra Ticks')
            plt.xticks(range(len(tp_multiples)), labels=tp_multiples)
            plt.yticks(range(len(sl_extra_ticks_list)), labels=sl_extra_ticks_list)
            
            # Trade count heatmap
            plt.subplot(2, 2, 4)
            pivot_data = sweep_df.pivot_table(index='sl_extra_ticks', columns='tp_multiple', values='trade_count', aggfunc='sum')
            sns_heatmap = plt.imshow(pivot_data, cmap='Greens')
            plt.colorbar(sns_heatmap, label='Trade Count')
            plt.title('Trade Count by TP and Extra Ticks')
            plt.xlabel('TP Multiple')
            plt.ylabel('SL Extra Ticks')
            plt.xticks(range(len(tp_multiples)), labels=tp_multiples)
            plt.yticks(range(len(sl_extra_ticks_list)), labels=sl_extra_ticks_list)
            
            plt.tight_layout()
            
            # Save the chart
            os.makedirs(params['output_dir'], exist_ok=True)
            plt.savefig(f"{params['output_dir']}/parameter_sweep_heatmaps.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"\nHeatmap visualization saved to {params['output_dir']}/parameter_sweep_heatmaps.png")
        except Exception as e:
            print(f"Error creating heatmap: {e}")
    
    # Save parameter sweep results to CSV
    os.makedirs(params['output_dir'], exist_ok=True)
    sweep_df.to_csv(f"{params['output_dir']}/parameter_sweep_results.csv", index=False)
    print(f"Parameter sweep results saved to {params['output_dir']}/parameter_sweep_results.csv")
    
    return optimal

def perform_day_by_day_sweep(df, params, tp_multiples, sl_extra_ticks_list, test_clusters):
    """
    Perform parameter sweep separately for each day of the week to find day-specific optimal parameters
    """
    print("\n" + "=" * 100)
    print(" " * 30 + "DAY-BY-DAY PARAMETER SWEEP ANALYSIS")
    print("=" * 100)
    
    # Define weekday order
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    
    # Store results by day
    weekday_results = {}
    day_optimal_params = {}
    
    # Create combinations with fixed SL distance
    from itertools import product
    param_combinations = list(product(tp_multiples, sl_extra_ticks_list))
    
    # Process each day separately
    for day in weekday_order:
        print(f"\n--- Processing {day} ---")
        
        # Filter data for this day
        day_df = df[pd.to_datetime(df['date']).dt.day_name() == day]
        
        if len(day_df) == 0 or day_df.empty:
            print(f"No data for {day}, skipping")
            continue
        
        print(f"Found {len(day_df['date'].unique())} {day}s in dataset")
        
        # Process parameter combinations for this day
        day_sweep_results = []
        total_combinations = len(param_combinations)
        
        # Use multiprocessing if enabled
        if params['use_multiprocessing']:
            # Create args list for this day
            day_combo_args = [(tp, sl, day_df, test_clusters, params) for tp, sl in param_combinations]
            
            # Get number of processes
            num_processes = min(mp.cpu_count(), total_combinations)
            print(f"Using {num_processes} CPU cores for parallel processing...")
            
            # Process with multiprocessing pool
            with mp.Pool(processes=num_processes) as pool:
                day_sweep_results = list(pool.imap(process_parameter_combo, day_combo_args))
        else:
            # Process sequentially
            for i, (tp_mult, sl_ex_ticks) in enumerate(param_combinations):
                # Print progress
                progress = (i + 1) / total_combinations * 100
                sys.stdout.write(f"\rProgress: {progress:.1f}% - Testing TP={tp_mult}, Extra={sl_ex_ticks}")
                sys.stdout.flush()
                
                # Process this combination for this day
                result = process_parameter_combo((tp_mult, sl_ex_ticks, day_df, test_clusters, params))
                day_sweep_results.append(result)
            
            print()  # New line after progress
        
        # Convert to DataFrame for analysis
        day_sweep_df = pd.DataFrame(day_sweep_results)
        
        # Check if we have valid results
        if day_sweep_df.empty or day_sweep_df['trade_count'].sum() == 0:
            print(f"No valid parameter combinations found for {day}")
            continue
        
        # Find optimal parameters for this day (by expectancy)
        try:
            # Filter out combinations with too few trades
            min_trades_df = day_sweep_df[day_sweep_df['trade_count'] >= params['min_trades']]
            
            if min_trades_df.empty:
                print(f"No parameter combinations with at least {params['min_trades']} trades for {day}")
                optimal = day_sweep_df.loc[day_sweep_df['trade_count'].idxmax()]  # Use the one with most trades
            else:
                # Use the one with highest expectancy
                optimal = min_trades_df.loc[min_trades_df['expectancy'].idxmax()]
            
            # Store optimal parameters for this day
            day_optimal_params[day] = {
                'tp_multiple': optimal['tp_multiple'],
                'sl_distance': optimal['sl_distance'],
                'sl_extra_ticks': int(optimal['sl_extra_ticks']),
                'trade_count': int(optimal['trade_count']),
                'win_rate': optimal['win_rate'],
                'total_r': optimal['total_r'],
                'expectancy': optimal['expectancy'],
                'profit_factor': optimal['profit_factor']
            }
            
            # Store all results for this day
            weekday_results[day] = day_sweep_df
            
            # Show top 3 for this day
            print(f"\nTop 3 Parameter Sets for {day} by Expectancy:")
            top_by_expectancy = day_sweep_df.sort_values('expectancy', ascending=False).head(3)
            for i, row in top_by_expectancy.iterrows():
                print(f"TP={row['tp_multiple']}, Extra={int(row['sl_extra_ticks'])}: "
                      f"Trades={int(row['trade_count'])}, Win Rate={row['win_rate']:.1f}%, "
                      f"Total R={row['total_r']:.2f}, Expectancy={row['expectancy']:.3f}")
            
            # Create day-specific visualization
            if MATPLOTLIB_AVAILABLE:
                try:
                    plt.figure(figsize=(10, 6))
                    
                    # Expectancy heatmap for this day
                    pivot_data = day_sweep_df.pivot_table(index='sl_extra_ticks', columns='tp_multiple', 
                                                         values='expectancy', aggfunc='mean')
                    sns_heatmap = plt.imshow(pivot_data, cmap='RdYlGn')
                    plt.colorbar(sns_heatmap, label='Expectancy')
                    plt.title(f'Expectancy by Parameters for {day}')
                    plt.xlabel('TP Multiple')
                    plt.ylabel('SL Extra Ticks')
                    plt.xticks(range(len(tp_multiples)), labels=tp_multiples)
                    plt.yticks(range(len(sl_extra_ticks_list)), labels=sl_extra_ticks_list)
                    
                    # Mark optimal point
                    opt_tp_idx = tp_multiples.index(optimal['tp_multiple'])
                    opt_sl_idx = sl_extra_ticks_list.index(int(optimal['sl_extra_ticks']))
                    plt.plot(opt_tp_idx, opt_sl_idx, 'ro', markersize=12, mfc='none')
                    
                    # Save this day's heatmap
                    os.makedirs(params['output_dir'], exist_ok=True)
                    day_filename = f"{params['output_dir']}/parameter_sweep_{day}.png"
                    plt.savefig(day_filename, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    print(f"Heatmap for {day} saved to {day_filename}")
                except Exception as e:
                    print(f"Error creating heatmap for {day}: {e}")
        
        except Exception as e:
            print(f"Error processing optimal parameters for {day}: {e}")
    
    # Display summary of optimal parameters by day
    print("\n" + "=" * 100)
    print(" " * 20 + "OPTIMAL PARAMETERS BY DAY OF WEEK")
    print("=" * 100)
    
    for day in weekday_order:
        if day in day_optimal_params:
            opt = day_optimal_params[day]
            print(f"\n{day}:")
            print(f"  TP Multiple: {opt['tp_multiple']}")
            print(f"  SL Distance: {opt['sl_distance']}")
            print(f"  SL Extra Ticks: {opt['sl_extra_ticks']}")
            print(f"  Results:")
            print(f"    Trade Count: {opt['trade_count']}")
            print(f"    Win Rate: {opt['win_rate']:.2f}%")
            print(f"    Total R: {opt['total_r']:.2f}")
            print(f"    Expectancy: {opt['expectancy']:.2f}R")
            print(f"    Profit Factor: {opt['profit_factor']:.2f}")
    
    # Create a comparison chart across days
    if MATPLOTLIB_AVAILABLE and len(day_optimal_params) > 0:
        try:
            plt.figure(figsize=(15, 10))
            
            # Extract data for plotting
            days = []
            tp_values = []
            sl_extra_values = []
            win_rates = []
            total_rs = []
            expectancies = []
            
            for day in weekday_order:
                if day in day_optimal_params:
                    days.append(day)
                    tp_values.append(day_optimal_params[day]['tp_multiple'])
                    sl_extra_values.append(day_optimal_params[day]['sl_extra_ticks'])
                    win_rates.append(day_optimal_params[day]['win_rate'])
                    total_rs.append(day_optimal_params[day]['total_r'])
                    expectancies.append(day_optimal_params[day]['expectancy'])
            
            # TP Multiple comparison
            plt.subplot(2, 2, 1)
            plt.bar(days, tp_values, color='skyblue')
            plt.title('Optimal TP Multiple by Day')
            plt.ylabel('TP Multiple')
            plt.ylim(min(tp_multiples), max(tp_multiples))
            
            # SL Extra Ticks comparison
            plt.subplot(2, 2, 2)
            plt.bar(days, sl_extra_values, color='lightgreen')
            plt.title('Optimal SL Extra Ticks by Day')
            plt.ylabel('SL Extra Ticks')
            plt.ylim(min(sl_extra_ticks_list)-0.5, max(sl_extra_ticks_list)+0.5)
            
            # Win Rate comparison
            plt.subplot(2, 2, 3)
            plt.bar(days, win_rates, color='gold')
            plt.title('Win Rate by Day')
            plt.ylabel('Win Rate %')
            
            # Expectancy comparison
            plt.subplot(2, 2, 4)
            plt.bar(days, expectancies, color='salmon')
            plt.title('Expectancy by Day')
            plt.ylabel('Expectancy (R)')
            
            plt.tight_layout()
            
            # Save comparison chart
            comparison_filename = f"{params['output_dir']}/day_by_day_comparison.png"
            plt.savefig(comparison_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"\nDay-by-day comparison chart saved to {comparison_filename}")
        except Exception as e:
            print(f"Error creating comparison chart: {e}")
    
    # Save day-by-day optimal parameters to CSV
    try:
        # Convert to DataFrame for easier saving
        day_params_df = pd.DataFrame.from_dict(day_optimal_params, orient='index')
        day_params_df.index.name = 'day_of_week'
        day_params_df.reset_index(inplace=True)
        
        # Save to CSV
        params_filename = f"{params['output_dir']}/day_optimal_parameters.csv"
        day_params_df.to_csv(params_filename, index=False)
        print(f"Day-by-day optimal parameters saved to {params_filename}")
    except Exception as e:
        print(f"Error saving day-by-day parameters: {e}")
    
    # Return a map of optimal parameters by day
    return day_optimal_params

def perform_walk_forward_analysis(df, params):
    """
    Perform walk-forward analysis to test strategy robustness
    """
    print("\n" + "=" * 100)
    print(" " * 35 + "WALK FORWARD ANALYSIS")
    print("=" * 100)
    
    # Get all unique dates
    all_dates = sorted(df["date"].unique())
    
    # Check for minimum 8 years of data (assuming ~252 trading days per year)
    min_days_required = 252 * 8
    if len(all_dates) < min_days_required:
        print(f"Not enough data for walk-forward analysis (need at least 8 years / ~{min_days_required} trading days)")
        print(f"Current dataset has {len(all_dates)} trading days, which is approximately {len(all_dates)/252:.1f} years")
        return
    
    print(f"Dataset contains {len(all_dates)} trading days (approximately {len(all_dates)/252:.1f} years)")
    
    # Define walk-forward windows
    in_sample_days = int(252 * 6)  # 6 years for in-sample training
    out_sample_days = int(252 * 2)  # 2 years for out-sample testing
    
    print(f"Using {in_sample_days} trading days (~6 years) for in-sample training")
    print(f"Using {out_sample_days} trading days (~2 years) for out-sample testing")
    
    # Create windows
    windows = []
    start_idx = 0
    
    while start_idx + in_sample_days + out_sample_days <= len(all_dates):
        in_sample_start = all_dates[start_idx]
        in_sample_end = all_dates[start_idx + in_sample_days - 1]
        out_sample_start = all_dates[start_idx + in_sample_days]
        out_sample_end = all_dates[start_idx + in_sample_days + out_sample_days - 1]
        
        windows.append({
            'in_sample_start': in_sample_start,
            'in_sample_end': in_sample_end,
            'out_sample_start': out_sample_start,
            'out_sample_end': out_sample_end
        })
        
        start_idx += out_sample_days  # Move forward by out_sample_days
    
    print(f"Created {len(windows)} walk-forward windows")
    
    # Store results
    wf_results = []
    all_out_sample_trades = []  # Collect all out-sample trades for combined equity curve
    
    # If day-by-day analysis is enabled, we'll use day-specific optimal parameters
    day_by_day_mode = params['perform_day_by_day_sweep']
    
    for i, window in enumerate(windows):
        print(f"\nProcessing Window {i+1}/{len(windows)}")
        print(f"In-sample: {window['in_sample_start']} to {window['in_sample_end']}")
        print(f"Out-sample: {window['out_sample_start']} to {window['out_sample_end']}")
        
        # Filter data for in-sample period
        in_sample_df = df[(df["date"] >= window['in_sample_start']) & 
                           (df["date"] <= window['in_sample_end'])]
        
        # Find optimal parameters using in-sample data
        if day_by_day_mode:
            print("Finding optimal parameters by day using in-sample data...")
            day_optimal = perform_day_by_day_sweep(in_sample_df, params, 
                                                 [0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 1.8, 2.0, 2.5],
                                                 [1, 2, 3],
                                                 [params['cluster_levels'][0], 
                                                  params['cluster_levels'][len(params['cluster_levels'])//2], 
                                                  params['cluster_levels'][-1]])
            
            if not day_optimal:
                print("No optimal parameters found for this window, skipping...")
                continue
        else:
            print("Finding optimal parameters using in-sample data...")
            optimal = perform_parameter_sweep(in_sample_df, params)
            
            if optimal is None:
                print("No optimal parameters found for this window, skipping...")
                continue
        
        # Test on out-of-sample data
        out_sample_df = df[(df["date"] >= window['out_sample_start']) & 
                           (df["date"] <= window['out_sample_end'])]
        
        # If using day-by-day mode, we apply parameters by day
        if day_by_day_mode:
            # Process each day separately with its optimal parameters
            all_day_results = []
            
            for day, opt_params in day_optimal.items():
                print(f"Applying {day} parameters: TP={opt_params['tp_multiple']}, "
                      f"SL Extra={opt_params['sl_extra_ticks']}")
                
                # Filter out-sample data for this day
                day_out_df = out_sample_df[pd.to_datetime(out_sample_df['date']).dt.day_name() == day]
                
                if day_out_df.empty:
                    print(f"No out-sample data for {day}, skipping")
                    continue
                
                # Set day-specific parameters
                day_params = params.copy()
                day_params['tp_multiple'] = opt_params['tp_multiple']
                day_params['sl_distance'] = 0.1  # Keep SL distance fixed
                day_params['sl_extra_ticks'] = opt_params['sl_extra_ticks']
                
                # Run backtest with day-specific parameters
                day_results = process_all_sessions(day_out_df, params['cluster_levels'], day_params)
                all_day_results.extend(day_results)
            
            # Combine all day results
            out_results_df = pd.DataFrame(all_day_results)
        else:
            # Use single set of optimal parameters
            sweep_params = params.copy()
            sweep_params['tp_multiple'] = optimal['tp_multiple']
            sweep_params['sl_distance'] = 0.1  # Keep SL distance fixed at 0.1
            sweep_params['sl_extra_ticks'] = int(optimal['sl_extra_ticks'])
            
            print(f"Using parameters: TP={sweep_params['tp_multiple']}, SL={sweep_params['sl_distance']}, "
                  f"Extra={sweep_params['sl_extra_ticks']}")
            
            # Run backtest on out-of-sample data
            out_results = process_all_sessions(out_sample_df, params['cluster_levels'], sweep_params)
            out_results_df = pd.DataFrame(out_results)
        
        if not out_results_df.empty:
            filtered_trades = out_results_df[out_results_df["trade_entry"].notna()]
            
            if not filtered_trades.empty:
                # Store these trades for combined equity curve
                filtered_trades['window'] = i+1
                all_out_sample_trades.append(filtered_trades)
                
                win_rate, avg_winner, avg_loser, expectancy = calculate_expectancy(filtered_trades)
                total_r = filtered_trades["pnl_R"].sum()
                profit_factor = calculate_profit_factor(filtered_trades)
                trade_count = len(filtered_trades)
                
                # Calculate average trade R
                avg_r = total_r / trade_count if trade_count > 0 else 0
                
                # Calculate max drawdown
                running_max = filtered_trades['pnl_R'].cumsum().expanding().max()
                drawdown = filtered_trades['pnl_R'].cumsum() - running_max
                max_drawdown = drawdown.min() if not drawdown.empty else 0
                
                # Store either day-specific or general parameters
                if day_by_day_mode:
                    param_info = "Day-specific parameters"
                else:
                    param_info = {
                        'tp_multiple': sweep_params['tp_multiple'],
                        'sl_distance': sweep_params['sl_distance'],
                        'sl_extra_ticks': sweep_params['sl_extra_ticks']
                    }
                
                wf_results.append({
                    'window': i+1,
                    'in_sample_start': window['in_sample_start'],
                    'in_sample_end': window['in_sample_end'],
                    'out_sample_start': window['out_sample_start'],
                    'out_sample_end': window['out_sample_end'],
                    'parameters': param_info,
                    'trade_count': trade_count,
                    'win_rate': win_rate,
                    'total_r': total_r,
                    'avg_r': avg_r,
                    'expectancy': expectancy,
                    'profit_factor': profit_factor,
                    'max_drawdown': max_drawdown
                })
                
                print(f"Out-sample results: {trade_count} trades, {win_rate:.2f}% win rate, "
                     f"{total_r:.2f} total R, {expectancy:.2f} expectancy")
            else:
                print("No valid trades in out-sample period")
        else:
            print("No valid trade setups in out-sample period")
    
    # Convert to DataFrame for analysis
    wf_df = pd.DataFrame([{k: v for k, v in d.items() if k != 'parameters'} for d in wf_results])
    
    if wf_df.empty:
        print("No valid walk-forward results")
        return
    
    # Combine all out-sample trades for a complete equity curve
    if all_out_sample_trades:
        all_trades_df = pd.concat(all_out_sample_trades)
        
        # Sort by date and time
        all_trades_df = all_trades_df.sort_values(['date', 'confirm_time'])
        
        # Generate combined equity curve
        if params['generate_equity_curve'] and MATPLOTLIB_AVAILABLE:
            print("\nGenerating combined equity curve for all out-sample periods...")
            generate_equity_curve(all_trades_df, params['output_dir'], "Combined_Walk_Forward")
            
            # Split by direction if both directions were analyzed
            if params["analysis_mode"] == "both":
                long_trades = all_trades_df[all_trades_df['trade_direction'] == "long"]
                short_trades = all_trades_df[all_trades_df['trade_direction'] == "short"]
                
                if not long_trades.empty:
                    generate_equity_curve(long_trades, params['output_dir'], "Combined_Walk_Forward_Long")
                if not short_trades.empty:
                    generate_equity_curve(short_trades, params['output_dir'], "Combined_Walk_Forward_Short")
    
    # Calculate overall walk-forward metrics
    total_wf_trades = wf_df['trade_count'].sum()
    total_wf_r = wf_df['total_r'].sum()
    avg_wf_win_rate = wf_df['win_rate'].mean()
    avg_wf_expectancy = wf_df['expectancy'].mean()
    avg_wf_profit_factor = wf_df['profit_factor'].mean()
    
    print("\nWalk-Forward Analysis Summary:")
    print(f"Total Windows: {len(wf_df)}")
    print(f"Total Trades: {total_wf_trades}")
    print(f"Total R: {total_wf_r:.2f}")
    print(f"Average Win Rate: {avg_wf_win_rate:.2f}%")
    print(f"Average Expectancy: {avg_wf_expectancy:.2f}R")
    print(f"Average Profit Factor: {avg_wf_profit_factor:.2f}")
    
    # Count profitable windows
    profitable_windows = len(wf_df[wf_df['total_r'] > 0])
    print(f"Profitable Windows: {profitable_windows}/{len(wf_df)} ({profitable_windows/len(wf_df)*100:.2f}%)")
    
    # Plot window performance if matplotlib is available
    if MATPLOTLIB_AVAILABLE:
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(wf_df)), wf_df['total_r'], color=['g' if r > 0 else 'r' for r in wf_df['total_r']])
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.xlabel('Window')
        plt.ylabel('Total R')
        plt.title('Walk-Forward Analysis - Window Performance')
        plt.grid(axis='y', alpha=0.3)
        
        # Add labels
        for i, r in enumerate(wf_df['total_r']):
            plt.text(i, r + (0.5 if r > 0 else -0.5), f"{r:.1f}R", 
                    ha='center', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{params['output_dir']}/walk_forward_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Walk-forward performance chart saved to {params['output_dir']}/walk_forward_performance.png")
    
    # Save walk-forward results to CSV
    wf_df.to_csv(f"{params['output_dir']}/walk_forward_results.csv", index=False)
    print(f"Walk-forward results saved to {params['output_dir']}/walk_forward_results.csv")
    
    return wf_df

# =========================== CORE STRATEGY FUNCTIONS ===========================

def get_applicable_wdr(session_date, tuesday_dates, tuesday_wdr):
    """
    Find the most recent Tuesday's WDR that applies to this session date.
    For Tuesday sessions, use the previous Tuesday's WDR.
    """
    if pd.Timestamp(session_date).day_name() == 'Tuesday':
        # For Tuesday, find the previous Tuesday
        prev_tuesdays = [date for date in tuesday_dates if date < session_date]
        if prev_tuesdays:
            prev_tuesday = max(prev_tuesdays)  # Most recent previous Tuesday
            if prev_tuesday in tuesday_wdr:
                return prev_tuesday, tuesday_wdr[prev_tuesday]
    else:
        # For other days, find the most recent Tuesday on or before this date
        applicable_tuesdays = [date for date in tuesday_dates if date <= session_date]
        if applicable_tuesdays:
            most_recent_tuesday = max(applicable_tuesdays)  # Most recent Tuesday
            if most_recent_tuesday in tuesday_wdr:
                return most_recent_tuesday, tuesday_wdr[most_recent_tuesday]
    
    # No applicable WDR found
    return None, None

def calculate_m7b(session_df):
    """
    Calculate the M7 Box (M7B) for a session.
    M7B is defined as the range between the open price at 9:30 and close price at 10:25.
    """
    # Get the 9:30 candle (market open)
    market_open_df = session_df[session_df["timestamp"].dt.time == pd.to_datetime("09:30").time()]
    
    # Get the 10:25 candle (end of DR period)
    dr_end_df = session_df[session_df["timestamp"].dt.time == pd.to_datetime("10:25").time()]
    
    # Debug print if we can't find these candles
    if market_open_df.empty or dr_end_df.empty:
        session_date = session_df["date"].iloc[0] if not session_df.empty else "unknown"
        return None, None
    
    # Get the open price at 9:30 and close price at 10:25
    open_price = market_open_df.iloc[0]["open"]
    close_price = dr_end_df.iloc[0]["close"]
    
    # Define M7B upper and lower boundaries
    m7b_upper = max(open_price, close_price)
    m7b_lower = min(open_price, close_price)
    
    return m7b_upper, m7b_lower

def simulate_trade(post_trade_df, entry_price, TP, SL, direction, tick_size):
    """
    Simulate a trade from the entry candle until 15:55.
    For long trades:
       - If candle's low <= SL, then SL is triggered.
       - If candle's high >= TP, then TP is triggered.
    For short trades:
       - If candle's high >= SL, then SL is triggered.
       - If candle's low <= TP, then TP is triggered.
    If both occur in the same candle, assume stop loss.
    """
    for idx, row in post_trade_df.iterrows():
        if direction == "long":
            if row["low"] <= SL and row["high"] >= TP:
                return SL, "SL"
            elif row["low"] <= SL:
                return SL, "SL"
            elif row["high"] >= TP:
                return TP, "TP"
        elif direction == "short":
            if row["high"] >= SL and row["low"] <= TP:
                return SL, "SL"
            elif row["high"] >= SL:
                return SL, "SL"
            elif row["low"] <= TP:
                return TP, "TP"
    return post_trade_df.iloc[-1]["close"], "EXIT"

def get_long_session_metrics(session_df, cluster, params, tuesday_dates, tuesday_wdr):
    """Process long trade setups for a session"""
    # Get time windows from parameters
    long_confirm_start = pd.to_datetime(params["long_confirm_start"]).time()
    long_confirm_end = pd.to_datetime(params["long_confirm_end"]).time()
    tick_size = params["tick_size"]
    
    dr_df = session_df[(session_df["timestamp"].dt.time >= pd.to_datetime("09:30").time()) &
                      (session_df["timestamp"].dt.time <= pd.to_datetime("10:25").time())]
    if dr_df.empty:
        return None
    DR_high = dr_df["high"].max()
    DR_low = dr_df["low"].min()
    IDR_high = dr_df["close"].max()
    IDR_low = dr_df["close"].min()
    SD = IDR_high - IDR_low if (IDR_high - IDR_low) != 0 else np.nan
    
    # Get the session date
    session_date = session_df["date"].iloc[0]
    day_of_week = pd.Timestamp(session_date).day_name()
    
    # Check WDR - just store info, don't filter yet
    wdr_tuesday = None
    wdr_high = None
    wdr_low = None
    wdr_reason = None
    
    if params["use_wdr_filter"]:
        wdr_tuesday, wdr_info = get_applicable_wdr(session_date, tuesday_dates, tuesday_wdr)
        if wdr_info:
            wdr_high = wdr_info["dr_high"]
            wdr_low = wdr_info["dr_low"]
    
    # Calculate M7B (M7 Box)
    m7b_upper, m7b_lower = calculate_m7b(session_df)
    m7b_invalid = False
    m7b_reason = None

    confirm_df = session_df[(session_df["timestamp"].dt.time >= long_confirm_start) &
                            (session_df["timestamp"].dt.time <= long_confirm_end)]
    confirm_signal = confirm_df[confirm_df["close"] > DR_high]
    if confirm_signal.empty:
        return None
    confirm_row = confirm_signal.iloc[0]
    confirm_time = confirm_row["timestamp"]

    session_close = session_df.iloc[-1]["close"]
    if session_close < DR_low:
        return None

    entry_window_df = session_df[(session_df["timestamp"].dt.time >= pd.to_datetime("10:40").time()) &
                                (session_df["timestamp"].dt.time <= pd.to_datetime("14:00").time()) &
                                (session_df["timestamp"] > confirm_time)]
    if entry_window_df.empty:
        return None

    baseline = IDR_high
    candidate_entry = baseline + cluster * SD
    TP = baseline + params["tp_multiple"] * SD  # Use TP multiple from parameters

    # Don't skip trade simulation based on WDR here - just run it normally
    candidate_entries = entry_window_df[entry_window_df["low"] <= candidate_entry]
    if candidate_entries.empty:
        trade_entry = None
        trade_result = None
    else:
        candidate_entry_index = candidate_entries.index[0]
        pre_candidate_df = entry_window_df.loc[:candidate_entry_index]
        if not pre_candidate_df.empty and pre_candidate_df["high"].max() >= TP:
            trade_entry = None
            trade_result = None
        else:
            # For long, ensure that between confirmation and candidate entry,
            # the price never falls below candidate_entry - sl_distance * SD.
            check_df = entry_window_df.loc[entry_window_df.index < candidate_entry_index]
            long_threshold = candidate_entry - params["sl_distance"] * SD  # Use SL distance from parameters
            if not check_df.empty and check_df["low"].min() < long_threshold:
                trade_entry = None
                trade_result = None
            else:
                trade_entry = candidate_entry
                SL = trade_entry - params["sl_distance"] * SD - (params["sl_extra_ticks"] * tick_size)  # Use SL extra ticks from parameters
                entry_time = entry_window_df.loc[candidate_entry_index, "timestamp"]
                
                # Check M7B filter - For LONG trades, invalidate if entry is below M7B lower boundary
                if params["use_m7b_filter"] and m7b_lower is not None and candidate_entry < m7b_lower:
                    m7b_invalid = True
                    m7b_reason = "Entry below M7B lower boundary"
                
                post_trade_df = session_df[(session_df["timestamp"] >= entry_time) &
                                        (session_df["timestamp"].dt.time <= pd.to_datetime("15:55").time())]
                if post_trade_df.empty:
                    trade_result = None
                else:
                    exit_price, exit_reason = simulate_trade(post_trade_df, trade_entry, TP, SL, "long", tick_size)
                    risk = trade_entry - SL
                    trade_result = {
                        "exit_reason": exit_reason,
                        "pnl_price": exit_price - trade_entry,
                        "pnl_R": (exit_price - trade_entry) / risk if risk != 0 else np.nan,
                        "pnl_ticks": (exit_price - trade_entry) / tick_size
                    }

    post_confirm_df = session_df[(session_df["timestamp"] > confirm_time) &
                                (session_df["timestamp"].dt.time <= pd.to_datetime("15:55").time())]
    if post_confirm_df.empty:
        return None
    max_high = post_confirm_df["high"].max()
    extreme_row = post_confirm_df[post_confirm_df["high"] == max_high].iloc[0]
    extension_value = max_high - baseline
    retracement_window = post_confirm_df[post_confirm_df["timestamp"] < extreme_row["timestamp"]]
    if retracement_window.empty:
        return None
    retracement_value = retracement_window["low"].min() - baseline
    extension_SD = extension_value / SD if pd.notna(SD) and SD != 0 else np.nan
    retracement_SD = retracement_value / SD if pd.notna(SD) and SD != 0 else np.nan

    confirm_time_str = confirm_time.strftime("%H:%M")

    # Now check WDR conditions AFTER all processing is done
    wdr_session_invalid = False
    wdr_cluster_invalid = False
    
    if params["use_wdr_filter"] and wdr_high is not None and wdr_low is not None:
        # Check if both IDR high and low are inside WDR
        if (wdr_low <= IDR_high <= wdr_high) and (wdr_low <= IDR_low <= wdr_high):
            wdr_session_invalid = True
            wdr_reason = "IDR inside WDR"
        # Check if TP is inside WDR
        elif wdr_low <= TP <= wdr_high:
            wdr_session_invalid = True
            wdr_reason = "TP inside WDR"
        # Check if cluster is inside WDR
        elif wdr_low <= candidate_entry <= wdr_high:
            wdr_cluster_invalid = True
            wdr_reason = "Cluster inside WDR"

    output = {
        "date": session_date,
        "trade_direction": "long",
        "day_of_week": day_of_week,
        "confirm_time": confirm_time,
        "confirm_time_str": confirm_time_str,
        "DR_high": DR_high,
        "DR_low": DR_low,
        "IDR_high": IDR_high,
        "IDR_low": IDR_low,
        "SD": SD,
        "session_close": session_close,
        "baseline": baseline,
        "TP": TP,
        "candidate_cluster": cluster,
        "candidate_entry": candidate_entry,
        "retracement": retracement_value,
        "retracement_SD": retracement_SD,
        "extension": extension_value,
        "extension_SD": extension_SD,
        "wdr_session_invalid": wdr_session_invalid,
        "wdr_cluster_invalid": wdr_cluster_invalid,
        "wdr_reason": wdr_reason,
    }
    
    # Add M7B information to output
    if m7b_upper is not None and m7b_lower is not None:
        output.update({
            "m7b_upper": m7b_upper,
            "m7b_lower": m7b_lower,
            "m7b_invalid": m7b_invalid,
            "m7b_reason": m7b_reason,
        })
    
    if wdr_tuesday is not None:
        output.update({
            "wdr_tuesday": wdr_tuesday,
            "wdr_high": wdr_high,
            "wdr_low": wdr_low,
        })
    
    # Store trade results with M7B filter applied
    if trade_result is not None:
        # First store the original unfiltered values
        output_values = {
            "unfiltered_trade_entry": trade_entry,
            "unfiltered_exit_reason": trade_result["exit_reason"],
            "unfiltered_pnl_price": trade_result["pnl_price"],
            "unfiltered_pnl_R": trade_result["pnl_R"],
            "unfiltered_pnl_ticks": trade_result["pnl_ticks"]
        }
        
        # Then store the filtered values (NaN if m7b_invalid is True)
        if m7b_invalid:
            output_values.update({
                "trade_entry": np.nan,
                "SL": np.nan,
                "trade_exit": np.nan,
                "exit_reason": np.nan,
                "pnl_price": np.nan,
                "pnl_R": np.nan,
                "pnl_ticks": np.nan
            })
        else:
            output_values.update({
                "trade_entry": trade_entry,
                "SL": SL,
                "trade_exit": exit_price,
                "exit_reason": trade_result["exit_reason"],
                "pnl_price": trade_result["pnl_price"],
                "pnl_R": trade_result["pnl_R"],
                "pnl_ticks": trade_result["pnl_ticks"]
            })
        
        output.update(output_values)
    else:
        output.update({
            "trade_entry": np.nan,
            "SL": np.nan,
            "trade_exit": np.nan,
            "exit_reason": np.nan,
            "pnl_price": np.nan,
            "pnl_R": np.nan,
            "pnl_ticks": np.nan,
            "unfiltered_trade_entry": np.nan,
            "unfiltered_exit_reason": np.nan,
            "unfiltered_pnl_price": np.nan,
            "unfiltered_pnl_R": np.nan,
            "unfiltered_pnl_ticks": np.nan
        })
    
    return output

def get_short_session_metrics(session_df, cluster, params, tuesday_dates, tuesday_wdr):
    """Process short trade setups for a session"""
    # Get time windows from parameters
    short_confirm_start = pd.to_datetime(params["short_confirm_start"]).time()
    short_confirm_end = pd.to_datetime(params["short_confirm_end"]).time()
    tick_size = params["tick_size"]
    
    dr_df = session_df[(session_df["timestamp"].dt.time >= pd.to_datetime("09:30").time()) &
                      (session_df["timestamp"].dt.time <= pd.to_datetime("10:25").time())]
    if dr_df.empty:
        return None
    DR_high = dr_df["high"].max()
    DR_low = dr_df["low"].min()
    IDR_high = dr_df["close"].max()
    IDR_low = dr_df["close"].min()
    SD = IDR_high - IDR_low if (IDR_high - IDR_low) != 0 else np.nan
    
    # Get the session date
    session_date = session_df["date"].iloc[0]
    day_of_week = pd.Timestamp(session_date).day_name()
    
    # Check WDR - just store info, don't filter yet
    wdr_tuesday = None
    wdr_high = None
    wdr_low = None
    wdr_reason = None
    
    if params["use_wdr_filter"]:
        wdr_tuesday, wdr_info = get_applicable_wdr(session_date, tuesday_dates, tuesday_wdr)
        if wdr_info:
            wdr_high = wdr_info["dr_high"]
            wdr_low = wdr_info["dr_low"]
            
    # Calculate M7B (M7 Box)
    m7b_upper, m7b_lower = calculate_m7b(session_df)
    m7b_invalid = False
    m7b_reason = None

    confirm_df = session_df[(session_df["timestamp"].dt.time >= short_confirm_start) &
                            (session_df["timestamp"].dt.time <= short_confirm_end)]
    confirm_signal = confirm_df[confirm_df["close"] < DR_low]
    if confirm_signal.empty:
        return None
    confirm_row = confirm_signal.iloc[0]
    confirm_time = confirm_row["timestamp"]

    session_close = session_df.iloc[-1]["close"]
    if session_close > DR_high:
        return None

    entry_window_df = session_df[(session_df["timestamp"].dt.time >= pd.to_datetime("10:40").time()) &
                                (session_df["timestamp"].dt.time <= pd.to_datetime("14:00").time()) &
                                (session_df["timestamp"] > confirm_time)]
    if entry_window_df.empty:
        return None

    baseline = IDR_low  # For short, baseline = IDR_low.
    candidate_entry = baseline - cluster * SD
    TP = baseline - params["tp_multiple"] * SD  # Use TP multiple from parameters

    # Don't skip trade simulation based on WDR here - just run it normally
    candidate_entries = entry_window_df[entry_window_df["high"] >= candidate_entry]
    if candidate_entries.empty:
        trade_entry = None
        trade_result = None
    else:
        candidate_entry_index = candidate_entries.index[0]
        pre_candidate_df = entry_window_df.loc[:candidate_entry_index]
        if not pre_candidate_df.empty and pre_candidate_df["low"].min() <= TP:
            trade_entry = None
            trade_result = None
        else:
            # For short, candidate valid only if between confirmation and candidate entry,
            # the price never goes above candidate_entry + sl_distance * SD.
            check_df = entry_window_df.loc[entry_window_df.index < candidate_entry_index]
            short_threshold = candidate_entry + params["sl_distance"] * SD  # Use SL distance from parameters
            if not check_df.empty and check_df["high"].max() > short_threshold:
                trade_entry = None
                trade_result = None
            else:
                trade_entry = candidate_entry
                SL = trade_entry + params["sl_distance"] * SD + (params["sl_extra_ticks"] * tick_size)  # Use SL extra ticks from parameters
                entry_time = entry_window_df.loc[candidate_entry_index, "timestamp"]
                
                # Check M7B filter - For SHORT trades, invalidate if entry is above M7B upper boundary
                if params["use_m7b_filter"] and m7b_upper is not None and candidate_entry > m7b_upper:
                    m7b_invalid = True
                    m7b_reason = "Entry above M7B upper boundary"
                
                post_trade_df = session_df[(session_df["timestamp"] >= entry_time) &
                                        (session_df["timestamp"].dt.time <= pd.to_datetime("15:55").time())]
                if post_trade_df.empty:
                    trade_result = None
                else:
                    exit_price, exit_reason = simulate_trade(post_trade_df, trade_entry, TP, SL, "short", tick_size)
                    risk = SL - trade_entry
                    trade_result = {
                        "exit_reason": exit_reason,
                        "pnl_price": trade_entry - exit_price,
                        "pnl_R": (trade_entry - exit_price) / risk if risk != 0 else np.nan,
                        "pnl_ticks": (trade_entry - exit_price) / tick_size
                    }

    post_confirm_df = session_df[(session_df["timestamp"] > confirm_time) &
                                (session_df["timestamp"].dt.time <= pd.to_datetime("15:55").time())]
    if post_confirm_df.empty:
        return None
    min_low = post_confirm_df["low"].min()
    max_high = post_confirm_df["high"].max()
    extension_value = baseline - min_low  # Favorable move (down from baseline).
    retracement_value = max_high - baseline  # Adverse move (up from baseline).
    extension_SD = extension_value / SD if pd.notna(SD) and SD != 0 else np.nan
    retracement_SD = retracement_value / SD if pd.notna(SD) and SD != 0 else np.nan

    confirm_time_str = confirm_time.strftime("%H:%M")
    
    # Now check WDR conditions AFTER all processing is done
    wdr_session_invalid = False
    wdr_cluster_invalid = False
    
    if params["use_wdr_filter"] and wdr_high is not None and wdr_low is not None:
        # Check if both IDR high and low are inside WDR
        if (wdr_low <= IDR_high <= wdr_high) and (wdr_low <= IDR_low <= wdr_high):
            wdr_session_invalid = True
            wdr_reason = "IDR inside WDR"
        # Check if TP is inside WDR
        elif wdr_low <= TP <= wdr_high:
            wdr_session_invalid = True
            wdr_reason = "TP inside WDR"
        # Check if cluster is inside WDR
        elif wdr_low <= candidate_entry <= wdr_high:
            wdr_cluster_invalid = True
            wdr_reason = "Cluster inside WDR"

    output = {
        "date": session_date,
        "trade_direction": "short",
        "day_of_week": day_of_week,
        "confirm_time": confirm_time,
        "confirm_time_str": confirm_time_str,
        "DR_high": DR_high,
        "DR_low": DR_low,
        "IDR_high": IDR_high,
        "IDR_low": IDR_low,
        "SD": SD,
        "session_close": session_close,
        "baseline": baseline,
        "TP": TP,
        "candidate_cluster": cluster,
        "candidate_entry": candidate_entry,
        "retracement": retracement_value,
        "retracement_SD": retracement_SD,
        "extension": extension_value,
        "extension_SD": extension_SD,
        "wdr_session_invalid": wdr_session_invalid,
        "wdr_cluster_invalid": wdr_cluster_invalid,
        "wdr_reason": wdr_reason,
    }
    
    # Add M7B information to output
    if m7b_upper is not None and m7b_lower is not None:
        output.update({
            "m7b_upper": m7b_upper,
            "m7b_lower": m7b_lower,
            "m7b_invalid": m7b_invalid,
            "m7b_reason": m7b_reason,
        })
    
    if wdr_tuesday is not None:
        output.update({
            "wdr_tuesday": wdr_tuesday,
            "wdr_high": wdr_high,
            "wdr_low": wdr_low,
        })
    
    # Store trade results with M7B filter applied
    if trade_result is not None:
        # First store the original unfiltered values
        output_values = {
            "unfiltered_trade_entry": trade_entry,
            "unfiltered_exit_reason": trade_result["exit_reason"],
            "unfiltered_pnl_price": trade_result["pnl_price"],
            "unfiltered_pnl_R": trade_result["pnl_R"],
            "unfiltered_pnl_ticks": trade_result["pnl_ticks"]
        }
        
        # Then store the filtered values (NaN if m7b_invalid is True)
        if m7b_invalid:
            output_values.update({
                "trade_entry": np.nan,
                "SL": np.nan,
                "trade_exit": np.nan,
                "exit_reason": np.nan,
                "pnl_price": np.nan,
                "pnl_R": np.nan,
                "pnl_ticks": np.nan
            })
        else:
            output_values.update({
                "trade_entry": trade_entry,
                "SL": SL,
                "trade_exit": exit_price,
                "exit_reason": trade_result["exit_reason"],
                "pnl_price": trade_result["pnl_price"],
                "pnl_R": trade_result["pnl_R"],
                "pnl_ticks": trade_result["pnl_ticks"]
            })
        
        output.update(output_values)
    else:
        output.update({
            "trade_entry": np.nan,
            "SL": np.nan,
            "trade_exit": np.nan,
            "exit_reason": np.nan,
            "pnl_price": np.nan,
            "pnl_R": np.nan,
            "pnl_ticks": np.nan,
            "unfiltered_trade_entry": np.nan,
            "unfiltered_exit_reason": np.nan,
            "unfiltered_pnl_price": np.nan,
            "unfiltered_pnl_R": np.nan,
                "unfiltered_pnl_ticks": np.nan
            })
        
    return output

def process_all_sessions(df, cluster_levels, params):
    """Process all sessions for all candidate clusters"""
    # Build Tuesday WDR dictionary
    # Get all unique dates
    all_dates = sorted(df["date"].unique())

    # Extract all Tuesdays
    tuesday_dates = []
    for date in all_dates:
        if pd.Timestamp(date).day_name() == 'Tuesday':
            tuesday_dates.append(date)

    # Create WDR for each Tuesday
    tuesday_wdr = {}
    for tuesday_date in tuesday_dates:
        tuesday_df = df[df["date"] == tuesday_date]
        
        # Calculate DR high and low for this Tuesday (9:30-10:25)
        dr_df = tuesday_df[(tuesday_df["timestamp"].dt.time >= pd.to_datetime("09:30").time()) &
                            (tuesday_df["timestamp"].dt.time <= pd.to_datetime("10:25").time())]
        
        if not dr_df.empty:
            dr_high = dr_df["high"].max()
            dr_low = dr_df["low"].min()
            
            tuesday_wdr[tuesday_date] = {
                "dr_high": dr_high,
                "dr_low": dr_low
            }
    
    # Process all sessions for all candidate clusters
    results = []
    for date, group in df.groupby("date"):
        if params["analysis_mode"] in ["long", "both"]:
            for cluster in cluster_levels:
                metrics = get_long_session_metrics(group, cluster, params, tuesday_dates, tuesday_wdr)
                if metrics is not None:
                    results.append(metrics)
        if params["analysis_mode"] in ["short", "both"]:
            for cluster in cluster_levels:
                metrics = get_short_session_metrics(group, cluster, params, tuesday_dates, tuesday_wdr)
                if metrics is not None:
                    results.append(metrics)
    
    return results

# =========================== DISPLAY FUNCTIONS ===========================

def display_raw_data(results_df, params):
    """Display raw configuration data"""
    if not params["show_raw_data"]:
        return
    
    # Define weekday order for sorting
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    
    print("\n" + "=" * 100)
    print(" " * 30 + "RAW DATA - ALL CONFIGURATIONS")
    print("=" * 100)
    
    # Group by day_of_week, confirmation time, trade_direction, and candidate_cluster
    grouped = results_df.groupby(['day_of_week', 'confirm_time_str', 'trade_direction', 'candidate_cluster'])
    
    # Sort days of week in proper order
    sorted_groups = []
    
    for name, group in grouped:
        day, conf_time, direction, cluster = name
        day_index = weekday_order.index(day) if day in weekday_order else 999
        sorted_groups.append((day_index, conf_time, direction, cluster, group))
    
    # Sort by day, time, direction, and cluster
    sorted_groups.sort()
    
    # Print raw data in the format requested
    for _, conf_time, direction, cluster, group in sorted_groups:
        day = group['day_of_week'].iloc[0]  # Get actual day name
        total_in_group = len(group)
        valid = group[group["trade_entry"].notna()]
        valid_count = len(valid)
        
        if not valid.empty:
            # Use the corrected expectancy calculation
            win_rate, avg_winner, avg_loser, grp_expectancy = calculate_expectancy(valid)
            
            # Calculate additional metrics
            profit_factor = calculate_profit_factor(valid) if params["calculate_profit_factor"] else None
            
            # Also calculate TP win rate for reference
            tp_win_rate = (valid["exit_reason"] == "TP").mean() * 100
            grp_total_R = valid["pnl_R"].sum()
            
            # Highlight profitability with simple markers
            profit_marker = '+' if grp_total_R > 0 else '-' if grp_total_R < 0 else ' '
            
            output_line = (f"{day} at {conf_time} ({str(direction).capitalize()}, Cluster: {float(cluster):+.2f}): "
                  f"{total_in_group} sessions, Valid Occurrences: {valid_count}, "
                  f"Profit Win Rate: {win_rate:.2f}% (TP Rate: {tp_win_rate:.2f}%), "
                  f"Total R: {profit_marker}{abs(grp_total_R):.2f}, Expectancy: {grp_expectancy:.2f}")
            
            if profit_factor is not None:
                output_line += f", Profit Factor: {profit_factor:.2f}"
                
            print(output_line)
        else:
            print(f"{day} at {conf_time} ({str(direction).capitalize()}, Cluster: {float(cluster):+.2f}): "
                  f"{total_in_group} sessions, Valid Occurrences: {valid_count}, "
                  f"Profit Win Rate: N/A, Total R: N/A, Expectancy: N/A")

def display_overall_stats(results_df, filtered_df, params):
    """Display overall strategy statistics"""
    if not params["show_overall_stats"]:
        return
    
    print("\n" + "=" * 100)
    print(" " * 35 + "OVERALL STATISTICS")
    print("=" * 100)
    
    # Create a copy of results_df with unfiltered trade data for comparison
    unfiltered_results_df = results_df.copy()
    if 'unfiltered_trade_entry' in unfiltered_results_df.columns and 'unfiltered_pnl_R' in unfiltered_results_df.columns and 'unfiltered_exit_reason' in unfiltered_results_df.columns:
        # Copy unfiltered data to the columns used for analysis
        mask = unfiltered_results_df['unfiltered_trade_entry'].notna()
        unfiltered_results_df.loc[mask, 'trade_entry'] = unfiltered_results_df.loc[mask, 'unfiltered_trade_entry']
        unfiltered_results_df.loc[mask, 'pnl_R'] = unfiltered_results_df.loc[mask, 'unfiltered_pnl_R']
        unfiltered_results_df.loc[mask, 'exit_reason'] = unfiltered_results_df.loc[mask, 'unfiltered_exit_reason']
    
    # Print stats for unfiltered data
    if params["use_wdr_filter"] or params["use_m7b_filter"]:
        print("\n--- WITHOUT FILTERS ---")
    
    # Unfiltered stats - using the unfiltered data
    unfiltered_valid_trades_df = unfiltered_results_df[unfiltered_results_df["trade_entry"].notna()]
    median_retracement = unfiltered_results_df["retracement_SD"].median()
    median_extension = unfiltered_results_df["extension_SD"].median()
    
    print(f"Total valid session entries: {len(unfiltered_results_df)}")
    print(f"Median Retracement (in SD units): {median_retracement:.2f}")
    print(f"Median Maximum Extension (in SD units): {median_extension:.2f}")

    if not unfiltered_valid_trades_df.empty:
        # Use GPU-accelerated expectancy calculation if available
        if params['use_gpu'] and GPU_AVAILABLE:
            try:
                profit_win_rate, avg_winner, avg_loser, overall_expectancy = calculate_expectancy_gpu(unfiltered_valid_trades_df)
            except Exception as e:
                print(f"GPU calculation failed, falling back to CPU: {e}")
                profit_win_rate, avg_winner, avg_loser, overall_expectancy = calculate_expectancy(unfiltered_valid_trades_df)
        else:
            profit_win_rate, avg_winner, avg_loser, overall_expectancy = calculate_expectancy(unfiltered_valid_trades_df)
        
        # Calculate additional metrics
        if params["calculate_profit_factor"]:
            profit_factor = calculate_profit_factor(unfiltered_valid_trades_df)
        
        if params["analyze_streaks"]:
            max_win_streak, max_loss_streak, avg_win_streak, avg_loss_streak, _ = analyze_streaks(unfiltered_valid_trades_df)
        
        # Also calculate TP hit rate for reference
        tp_win_rate = (unfiltered_valid_trades_df["exit_reason"] == "TP").mean() * 100
        total_R = unfiltered_valid_trades_df["pnl_R"].sum()
        
        print(f"\nValid trades: {len(unfiltered_valid_trades_df)}")
        print(f"Overall Profit Win Rate: {profit_win_rate:.2f}%")
        print(f"Overall TP Hit Rate: {tp_win_rate:.2f}%")
        print(f"Overall Total R (sum of pnl_R): {total_R:.2f}")
        print(f"Overall Average Winner (in R): {avg_winner:.2f}")
        print(f"Overall Average Loser (in R): {avg_loser:.2f}")
        print(f"Overall Expectancy (in R): {overall_expectancy:.2f}")
        
        if params["calculate_profit_factor"]:
            print(f"Overall Profit Factor: {profit_factor:.2f}")
        
        if params["analyze_streaks"]:
            print("\nStreaks Analysis:")
            print(f"Max Winning Streak: {max_win_streak}")
            print(f"Max Losing Streak: {max_loss_streak}")
            print(f"Average Winning Streak: {avg_win_streak:.2f}")
            print(f"Average Losing Streak: {avg_loss_streak:.2f}")
    else:
        print("No trades were simulated in the valid sessions.")
    
    # Print stats for filtered data if filtering is enabled
    if (params["use_wdr_filter"] or params["use_m7b_filter"]) and params["show_overall_stats"]:
        print("\n--- WITH FILTERS ---")
        filtered_valid_trades = filtered_df[filtered_df["trade_entry"].notna()]
        filtered_median_retracement = filtered_df["retracement_SD"].median()
        filtered_median_extension = filtered_df["extension_SD"].median()
        
        filtered_session_count = len(filtered_df)
        if params["use_wdr_filter"] and 'wdr_session_invalid' in filtered_df.columns:
            filtered_session_count -= filtered_df['wdr_session_invalid'].sum()
        
        print(f"Total valid session entries: {filtered_session_count}")
        print(f"Median Retracement (in SD units): {filtered_median_retracement:.2f}")
        print(f"Median Maximum Extension (in SD units): {filtered_median_extension:.2f}")

        if not filtered_valid_trades.empty:
            # Use corrected expectancy calculation
            filtered_profit_win_rate, filtered_avg_winner, filtered_avg_loser, filtered_expectancy = calculate_expectancy(filtered_valid_trades)
            
            # Calculate additional metrics for filtered data
            if params["calculate_profit_factor"]:
                filtered_profit_factor = calculate_profit_factor(filtered_valid_trades)
            
            if params["analyze_streaks"]:
                filtered_max_win_streak, filtered_max_loss_streak, filtered_avg_win_streak, filtered_avg_loss_streak, _ = analyze_streaks(filtered_valid_trades)
            
            # Also calculate TP hit rate for reference
            filtered_tp_win_rate = (filtered_valid_trades["exit_reason"] == "TP").mean() * 100
            filtered_total_R = filtered_valid_trades["pnl_R"].sum()
            
            print(f"\nValid trades: {len(filtered_valid_trades)}")
            print(f"Overall Profit Win Rate: {filtered_profit_win_rate:.2f}%")
            print(f"Overall TP Hit Rate: {filtered_tp_win_rate:.2f}%")
            print(f"Overall Total R (sum of pnl_R): {filtered_total_R:.2f}")
            print(f"Overall Average Winner (in R): {filtered_avg_winner:.2f}")
            print(f"Overall Average Loser (in R): {filtered_avg_loser:.2f}")
            print(f"Overall Expectancy (in R): {filtered_expectancy:.2f}")
            
            if params["calculate_profit_factor"]:
                print(f"Overall Profit Factor: {filtered_profit_factor:.2f}")
            
            if params["analyze_streaks"]:
                print("\nStreaks Analysis (Filtered):")
                print(f"Max Winning Streak: {filtered_max_win_streak}")
                print(f"Max Losing Streak: {filtered_max_loss_streak}")
                print(f"Average Winning Streak: {filtered_avg_win_streak:.2f}")
                print(f"Average Losing Streak: {filtered_avg_loss_streak:.2f}")
            
            # Calculate improvement safely
            try:
                improvement_R = filtered_total_R - total_R
                improvement_R_pct = (improvement_R/abs(total_R))*100 if total_R != 0 else float('inf')
            except:
                improvement_R = 0
                improvement_R_pct = 0
                
            try:
                improvement_win_rate = filtered_profit_win_rate - profit_win_rate
                improvement_win_rate_pct = (improvement_win_rate/profit_win_rate)*100 if profit_win_rate != 0 else float('inf')
            except:
                improvement_win_rate = 0
                improvement_win_rate_pct = 0
                
            try:
                improvement_expectancy = filtered_expectancy - overall_expectancy
                improvement_expectancy_pct = (improvement_expectancy/abs(overall_expectancy))*100 if overall_expectancy != 0 else float('inf')
            except:
                improvement_expectancy = 0
                improvement_expectancy_pct = 0
            
            print("\nImprovement from filtering:")
            print(f"Total R: {improvement_R:.2f} ({improvement_R_pct:.2f}% change)")
            print(f"Win Rate: {improvement_win_rate:.2f}% ({improvement_win_rate_pct:.2f}% change)")
            print(f"Expectancy: {improvement_expectancy:.2f} ({improvement_expectancy_pct:.2f}% change)")
            
            if params["calculate_profit_factor"] and 'profit_factor' in locals() and profit_factor != 0:
                improvement_pf = filtered_profit_factor - profit_factor
                improvement_pf_pct = (improvement_pf/profit_factor)*100
                print(f"Profit Factor: {improvement_pf:.2f} ({improvement_pf_pct:.2f}% change)")
        else:
            print("No trades were simulated in the filtered sessions.")
    
    return unfiltered_valid_trades_df, filtered_valid_trades

def display_day_performance(unfiltered_valid_trades_df, filtered_valid_trades, params):
    """Display performance by day of week"""
    if not params["show_day_performance"]:
        return
    
    # Define weekday order for sorting
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    
    print("\n" + "=" * 100)
    print(" " * 32 + "PERFORMANCE BY DAY OF WEEK")
    print("=" * 100)
    
    # First show unfiltered day of week performance
    if params["use_wdr_filter"] or params["use_m7b_filter"]:
        print("\n--- WITHOUT FILTERS ---")
    
    # Calculate day of week performance for unfiltered data
    if not unfiltered_valid_trades_df.empty:
        # First, handle the case when analysis_mode is "both"
        if params["analysis_mode"] == "both":
            print("\n" + "-" * 50)
            print(" " * 15 + "LONG TRADES")
            print("-" * 50)
            
            # Filter for long trades only
            long_trades_df = unfiltered_valid_trades_df[unfiltered_valid_trades_df['trade_direction'] == "long"]
            
            if not long_trades_df.empty:
                for day in weekday_order:
                    if day in long_trades_df['day_of_week'].values:
                        day_data = long_trades_df[long_trades_df['day_of_week'] == day]
                        
                        trade_count = len(day_data)
                        win_rate, _, _, expectancy = calculate_expectancy(day_data)
                        tp_rate = (day_data["exit_reason"] == "TP").mean() * 100
                        total_r = day_data['pnl_R'].sum()
                        avg_r = day_data['pnl_R'].mean()
                        
                        # Calculate profit factor if requested
                        profit_factor_str = ""
                        if params["calculate_profit_factor"]:
                            profit_factor = calculate_profit_factor(day_data)
                            profit_factor_str = f" | Profit Factor: {profit_factor:6.2f}"
                        
                        # Profit indicator
                        profit_indicator = "+" if total_r > 0 else "-" if total_r < 0 else " "
                        
                        # Format the output line
                        print(f"{day:<10} | Trades: {trade_count:3d} | Profit Rate: {win_rate:6.2f}% | TP Rate: {tp_rate:6.2f}% | "
                              f"Total R: {profit_indicator}{abs(total_r):6.2f} | Avg R: {avg_r:6.2f} | Expectancy: {expectancy:6.2f}{profit_factor_str}")
            else:
                print("No valid long trades to analyze by day of week.")
                
            print("\n" + "-" * 50)
            print(" " * 15 + "SHORT TRADES")
            print("-" * 50)
            
            # Filter for short trades only
            short_trades_df = unfiltered_valid_trades_df[unfiltered_valid_trades_df['trade_direction'] == "short"]
            
            if not short_trades_df.empty:
                for day in weekday_order:
                    if day in short_trades_df['day_of_week'].values:
                        day_data = short_trades_df[short_trades_df['day_of_week'] == day]
                        
                        trade_count = len(day_data)
                        win_rate, _, _, expectancy = calculate_expectancy(day_data)
                        tp_rate = (day_data["exit_reason"] == "TP").mean() * 100
                        total_r = day_data['pnl_R'].sum()
                        avg_r = day_data['pnl_R'].mean()
                        
                        # Calculate profit factor if requested
                        profit_factor_str = ""
                        if params["calculate_profit_factor"]:
                            profit_factor = calculate_profit_factor(day_data)
                            profit_factor_str = f" | Profit Factor: {profit_factor:6.2f}"
                        
                        # Profit indicator
                        profit_indicator = "+" if total_r > 0 else "-" if total_r < 0 else " "
                        
                        # Format the output line
                        print(f"{day:<10} | Trades: {trade_count:3d} | Profit Rate: {win_rate:6.2f}% | TP Rate: {tp_rate:6.2f}% | "
                              f"Total R: {profit_indicator}{abs(total_r):6.2f} | Avg R: {avg_r:6.2f} | Expectancy: {expectancy:6.2f}{profit_factor_str}")
            else:
                print("No valid short trades to analyze by day of week.")
        
        # If analysis_mode is "long" or "short", keep the original logic
        else:
            for day in weekday_order:
                if day in unfiltered_valid_trades_df['day_of_week'].values:
                    day_data = unfiltered_valid_trades_df[unfiltered_valid_trades_df['day_of_week'] == day]
                    
                    trade_count = len(day_data)
                    win_rate, _, _, expectancy = calculate_expectancy(day_data)
                    tp_rate = (day_data["exit_reason"] == "TP").mean() * 100
                    total_r = day_data['pnl_R'].sum()
                    avg_r = day_data['pnl_R'].mean()
                    
                    # Calculate profit factor if requested
                    profit_factor_str = ""
                    if params["calculate_profit_factor"]:
                        profit_factor = calculate_profit_factor(day_data)
                        profit_factor_str = f" | Profit Factor: {profit_factor:6.2f}"
                    
                    # Profit indicator
                    profit_indicator = "+" if total_r > 0 else "-" if total_r < 0 else " "
                    
                    # Format the output line
                    print(f"{day:<10} | Trades: {trade_count:3d} | Profit Rate: {win_rate:6.2f}% | TP Rate: {tp_rate:6.2f}% | "
                          f"Total R: {profit_indicator}{abs(total_r):6.2f} | Avg R: {avg_r:6.2f} | Expectancy: {expectancy:6.2f}{profit_factor_str}")
    else:
        print("No valid trades to analyze by day of week.")
    
    # Then show filtered day of week performance (if enabled)
    if (params["use_wdr_filter"] or params["use_m7b_filter"]) and not filtered_valid_trades.empty:
        print("\n--- WITH FILTERS ---")
        
        # First, handle the case when analysis_mode is "both"
        if params["analysis_mode"] == "both":
            print("\n" + "-" * 50)
            print(" " * 15 + "LONG TRADES (Filtered)")
            print("-" * 50)
            
            # Filter for long trades only
            long_trades_df = filtered_valid_trades[filtered_valid_trades['trade_direction'] == "long"]
            
            if not long_trades_df.empty:
                for day in weekday_order:
                    if day in long_trades_df['day_of_week'].values:
                        day_data = long_trades_df[long_trades_df['day_of_week'] == day]
                        
                        trade_count = len(day_data)
                        win_rate, _, _, expectancy = calculate_expectancy(day_data)
                        tp_rate = (day_data["exit_reason"] == "TP").mean() * 100
                        total_r = day_data['pnl_R'].sum()
                        avg_r = day_data['pnl_R'].mean()
                        
                        # Calculate profit factor if requested
                        profit_factor_str = ""
                        if params["calculate_profit_factor"]:
                            profit_factor = calculate_profit_factor(day_data)
                            profit_factor_str = f" | Profit Factor: {profit_factor:6.2f}"
                        
                        # Profit indicator
                        profit_indicator = "+" if total_r > 0 else "-" if total_r < 0 else " "
                        
                        # Format the output line
                        print(f"{day:<10} | Trades: {trade_count:3d} | Profit Rate: {win_rate:6.2f}% | TP Rate: {tp_rate:6.2f}% | "
                              f"Total R: {profit_indicator}{abs(total_r):6.2f} | Avg R: {avg_r:6.2f} | Expectancy: {expectancy:6.2f}{profit_factor_str}")
            else:
                print("No valid long trades to analyze by day of week with filters.")
                
            print("\n" + "-" * 50)
            print(" " * 15 + "SHORT TRADES (Filtered)")
            print("-" * 50)
            
            # Filter for short trades only
            short_trades_df = filtered_valid_trades[filtered_valid_trades['trade_direction'] == "short"]
            
            if not short_trades_df.empty:
                for day in weekday_order:
                    if day in short_trades_df['day_of_week'].values:
                        day_data = short_trades_df[short_trades_df['day_of_week'] == day]
                        
                        trade_count = len(day_data)
                        win_rate, _, _, expectancy = calculate_expectancy(day_data)
                        tp_rate = (day_data["exit_reason"] == "TP").mean() * 100
                        total_r = day_data['pnl_R'].sum()
                        avg_r = day_data['pnl_R'].mean()
                        
                        # Calculate profit factor if requested
                        profit_factor_str = ""
                        if params["calculate_profit_factor"]:
                            profit_factor = calculate_profit_factor(day_data)
                            profit_factor_str = f" | Profit Factor: {profit_factor:6.2f}"
                        
                        # Profit indicator
                        profit_indicator = "+" if total_r > 0 else "-" if total_r < 0 else " "
                        
                        # Format the output line
                        print(f"{day:<10} | Trades: {trade_count:3d} | Profit Rate: {win_rate:6.2f}% | TP Rate: {tp_rate:6.2f}% | "
                              f"Total R: {profit_indicator}{abs(total_r):6.2f} | Avg R: {avg_r:6.2f} | Expectancy: {expectancy:6.2f}{profit_factor_str}")
            else:
                print("No valid short trades to analyze by day of week with filters.")
        
        # If analysis_mode is "long" or "short", keep the original logic but for filtered data
        else:
            for day in weekday_order:
                if day in filtered_valid_trades['day_of_week'].values:
                    day_data = filtered_valid_trades[filtered_valid_trades['day_of_week'] == day]
                    
                    trade_count = len(day_data)
                    win_rate, _, _, expectancy = calculate_expectancy(day_data)
                    tp_rate = (day_data["exit_reason"] == "TP").mean() * 100
                    total_r = day_data['pnl_R'].sum()
                    avg_r = day_data['pnl_R'].mean()
                    
                    # Calculate profit factor if requested
                    profit_factor_str = ""
                    if params["calculate_profit_factor"]:
                        profit_factor = calculate_profit_factor(day_data)
                        profit_factor_str = f" | Profit Factor: {profit_factor:6.2f}"
                    
                    # Profit indicator
                    profit_indicator = "+" if total_r > 0 else "-" if total_r < 0 else " "
                    
                    # Format the output line
                    print(f"{day:<10} | Trades: {trade_count:3d} | Profit Rate: {win_rate:6.2f}% | TP Rate: {tp_rate:6.2f}% | "
                          f"Total R: {profit_indicator}{abs(total_r):6.2f} | Avg R: {avg_r:6.2f} | Expectancy: {expectancy:6.2f}{profit_factor_str}")

def display_top_configs(unfiltered_results_df, filtered_df, unfiltered_valid_trades_df, filtered_valid_trades, params):
    """Display top configurations"""
    if not params["show_top_configs"]:
        return
    
    print("\n" + "=" * 100)
    print(" " * 30 + f"TOP {params['top_configs']} MOST PROFITABLE CONFIGURATIONS")
    print("=" * 100)
    
    # First show unfiltered top configurations
    if params["use_wdr_filter"] or params["use_m7b_filter"]:
        print("\n--- WITHOUT FILTERS ---")
    
    # Group for unfiltered statistics
    unfiltered_grouped = unfiltered_results_df.groupby(['day_of_week', 'confirm_time_str', 'trade_direction', 'candidate_cluster'])
    
    # Calculate metrics for each configuration (unfiltered)
    config_metrics_df = None
    filtered_config_metrics_df = None
    
    if not unfiltered_valid_trades_df.empty:
        config_metrics = []
        
        for name, group in unfiltered_grouped:
            day, conf_time, direction, cluster = name
            valid = group[group["trade_entry"].notna()]
            valid_count = len(valid)
            
            if valid_count >= params["min_trades"]:  # Only consider configurations with at least min_trades_for_analysis trades
                # Use corrected expectancy calculation
                win_rate, avg_winner, avg_loser, expectancy = calculate_expectancy(valid)
                total_R = valid["pnl_R"].sum()
                
                # Add profit factor if requested
                if params["calculate_profit_factor"]:
                    profit_factor = calculate_profit_factor(valid)
                else:
                    profit_factor = None
                
                config_metrics.append({
                    'day': day,
                    'time': conf_time,
                    'direction': direction,
                    'cluster': cluster,
                    'trades': valid_count,
                    'win_rate': win_rate,
                    'total_R': total_R,
                    'expectancy': expectancy,
                    'profit_factor': profit_factor
                })
        
        # Sort by total R (most profitable first)
        config_metrics_df = pd.DataFrame(config_metrics)
        if not config_metrics_df.empty:
            sorted_configs = config_metrics_df.sort_values('total_R', ascending=False).head(params["top_configs"])
            
            # Print the top configurations
            for i, row in sorted_configs.iterrows():
                # Profit indicator
                profit_indicator = "+" if row['total_R'] > 0 else "-" if row['total_R'] < 0 else " "
                exp_indicator = "+" if row['expectancy'] > 0 else "-" if row['expectancy'] < 0 else " "
                
                output_line = (f"#{i+1:2d}: {row['day']:<9} at {row['time']} ({row['direction'].capitalize():<5}, "
                      f"Cluster: {float(row['cluster']):+.2f}): "
                      f"Trades: {row['trades']:3d}, Profit Rate: {row['win_rate']:6.2f}%, "
                      f"Total R: {profit_indicator}{abs(row['total_R']):6.2f}, Expectancy: {exp_indicator}{abs(row['expectancy']):5.2f}")
                
                if params["calculate_profit_factor"] and row['profit_factor'] is not None:
                    output_line += f", Profit Factor: {row['profit_factor']:.2f}"
                
                print(output_line)
        else:
            print("No configurations with sufficient trades to analyze.")
    else:
        print("No valid trades found for analysis.")
    
    # Then show filtered top configurations (if enabled)
    if (params["use_wdr_filter"] or params["use_m7b_filter"]) and filtered_valid_trades is not None and not filtered_valid_trades.empty:
        print("\n--- WITH FILTERS ---")
        
        # Group and calculate metrics for filtered configurations
        filtered_grouped = filtered_df.groupby(['day_of_week', 'confirm_time_str', 'trade_direction', 'candidate_cluster'])
        filtered_config_metrics = []
        
        for name, group in filtered_grouped:
            day, conf_time, direction, cluster = name
            valid = group[group["trade_entry"].notna()]
            valid_count = len(valid)
            
            if valid_count >= params["min_trades"]:
                # Use corrected expectancy calculation
                win_rate, avg_winner, avg_loser, expectancy = calculate_expectancy(valid)
                total_R = valid["pnl_R"].sum()
                
                # Add profit factor if requested
                if params["calculate_profit_factor"]:
                    profit_factor = calculate_profit_factor(valid)
                else:
                    profit_factor = None
                
                filtered_config_metrics.append({
                    'day': day,
                    'time': conf_time,
                    'direction': direction,
                    'cluster': cluster,
                    'trades': valid_count,
                    'win_rate': win_rate,
                    'total_R': total_R,
                    'expectancy': expectancy,
                    'profit_factor': profit_factor
                })
        
        # Sort by total R (most profitable first)
        filtered_config_metrics_df = pd.DataFrame(filtered_config_metrics)
        if not filtered_config_metrics_df.empty:
            filtered_sorted_configs = filtered_config_metrics_df.sort_values('total_R', ascending=False).head(params["top_configs"])
            
            # Print the top filtered configurations
            for i, row in filtered_sorted_configs.iterrows():
                # Profit indicator
                profit_indicator = "+" if row['total_R'] > 0 else "-" if row['total_R'] < 0 else " "
                exp_indicator = "+" if row['expectancy'] > 0 else "-" if row['expectancy'] < 0 else " "
                
                output_line = (f"#{i+1:2d}: {row['day']:<9} at {row['time']} ({row['direction'].capitalize():<5}, "
                      f"Cluster: {float(row['cluster']):+.2f}): "
                      f"Trades: {row['trades']:3d}, Profit Rate: {row['win_rate']:6.2f}%, "
                      f"Total R: {profit_indicator}{abs(row['total_R']):6.2f}, Expectancy: {exp_indicator}{abs(row['expectancy']):5.2f}")
                
                if params["calculate_profit_factor"] and row['profit_factor'] is not None:
                    output_line += f", Profit Factor: {row['profit_factor']:.2f}"
                
                print(output_line)
        else:
            print("No configurations with sufficient trades to analyze after filtering.")
    
    return config_metrics_df, filtered_config_metrics_df

def display_target_configs(config_metrics_df, filtered_config_metrics_df, params):
    """Display configurations meeting target criteria"""
    if not params["show_target_configs"] or (params["target_win_rate"] is None and params["target_expectancy"] is None):
        return
    
    print("\n" + "=" * 100)
    print(" " * 30 + "CONFIGURATIONS MEETING TARGET CRITERIA")
    print("=" * 100)
    print(f"Target Win Rate: {params['target_win_rate']}%, Target Expectancy: {params['target_expectancy']}R")
    
    # First show unfiltered configurations meeting criteria
    if params["use_wdr_filter"] or params["use_m7b_filter"]:
        print("\n--- WITHOUT FILTERS ---")
    
    # Filter configurations meeting criteria (unfiltered)
    if config_metrics_df is not None and not config_metrics_df.empty:
        criteria_configs = config_metrics_df.copy()
        
        if params["target_win_rate"] is not None:
            criteria_configs = criteria_configs[criteria_configs['win_rate'] >= params["target_win_rate"]]
        if params["target_expectancy"] is not None:
            criteria_configs = criteria_configs[criteria_configs['expectancy'] >= params["target_expectancy"]]
        
        if not criteria_configs.empty:
            # Sort by total R
            sorted_criteria = criteria_configs.sort_values('total_R', ascending=False)
            
            for i, row in sorted_criteria.iterrows():
                # Profit indicator
                profit_indicator = "+" if row['total_R'] > 0 else "-" if row['total_R'] < 0 else " "
                exp_indicator = "+" if row['expectancy'] > 0 else "-" if row['expectancy'] < 0 else " "
                
                output_line = (f"{row['day']:<9} at {row['time']} ({row['direction'].capitalize():<5}, "
                      f"Cluster: {float(row['cluster']):+.2f}): "
                      f"Trades: {row['trades']:3d}, Profit Rate: {row['win_rate']:6.2f}%, "
                      f"Total R: {profit_indicator}{abs(row['total_R']):6.2f}, Expectancy: {exp_indicator}{abs(row['expectancy']):5.2f}")
                
                if params["calculate_profit_factor"] and 'profit_factor' in row and row['profit_factor'] is not None:
                    output_line += f", Profit Factor: {row['profit_factor']:.2f}"
                
                print(output_line)
        else:
            print("No configurations meet the target criteria.")
    else:
        print("No configuration metrics available for target filtering.")
    
    # Then show filtered configurations meeting criteria (if enabled)
    if (params["use_wdr_filter"] or params["use_m7b_filter"]) and filtered_config_metrics_df is not None and not filtered_config_metrics_df.empty:
        print("\n--- WITH FILTERS ---")
        
        filtered_criteria = filtered_config_metrics_df.copy()
        
        if params["target_win_rate"] is not None:
            filtered_criteria = filtered_criteria[filtered_criteria['win_rate'] >= params["target_win_rate"]]
        if params["target_expectancy"] is not None:
            filtered_criteria = filtered_criteria[filtered_criteria['expectancy'] >= params["target_expectancy"]]
        
        if not filtered_criteria.empty:
            # Sort by total R
            sorted_filtered_criteria = filtered_criteria.sort_values('total_R', ascending=False)
            
            for i, row in sorted_filtered_criteria.iterrows():
                # Profit indicator
                profit_indicator = "+" if row['total_R'] > 0 else "-" if row['total_R'] < 0 else " "
                exp_indicator = "+" if row['expectancy'] > 0 else "-" if row['expectancy'] < 0 else " "
                
                output_line = (f"{row['day']:<9} at {row['time']} ({row['direction'].capitalize():<5}, "
                      f"Cluster: {float(row['cluster']):+.2f}): "
                      f"Trades: {row['trades']:3d}, Profit Rate: {row['win_rate']:6.2f}%, "
                      f"Total R: {profit_indicator}{abs(row['total_R']):6.2f}, Expectancy: {exp_indicator}{abs(row['expectancy']):5.2f}")
                
                if params["calculate_profit_factor"] and 'profit_factor' in row and row['profit_factor'] is not None:
                    output_line += f", Profit Factor: {row['profit_factor']:.2f}"
                
                print(output_line)
        else:
            print("No configurations meet the target criteria after filtering.")

def display_filter_stats(results_df, params):
    """Display filter statistics"""
    if not params["show_filter_stats"]:
        return
    
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    
    # Print M7B statistics if filter is enabled
    if params["use_m7b_filter"] and 'm7b_invalid' in results_df.columns:
        # Fixed: Use fillna to handle NaN values in boolean filtering
        m7b_counts = results_df[results_df['m7b_invalid'].fillna(False) & results_df['unfiltered_trade_entry'].notna()].shape[0]
        
        print("\n" + "=" * 100)
        print(" " * 35 + "M7B FILTER STATISTICS")
        print("=" * 100)
        
        # Print overall filter count
        unfiltered_total = results_df['unfiltered_trade_entry'].notna().sum()
        filtered_total = results_df['trade_entry'].notna().sum()
        filtered_out = unfiltered_total - filtered_total
        pct_filtered = (filtered_out / unfiltered_total * 100) if unfiltered_total > 0 else 0
        
        print(f"\nM7B filter removed {filtered_out} of {unfiltered_total} trades ({pct_filtered:.2f}%)")
        
        # Print sample filtered trades
        if 'm7b_upper' in results_df.columns and 'm7b_lower' in results_df.columns:
            # Fixed: Use fillna in boolean filtering
            sample_data = results_df[results_df['m7b_invalid'].fillna(False) & results_df['unfiltered_trade_entry'].notna()].head(5)
            if not sample_data.empty:
                print("\nSample of filtered trades (first 5):")
                for _, row in sample_data.iterrows():
                    direction = row['trade_direction']
                    entry = row['unfiltered_trade_entry']
                    if pd.notna(entry):  # Make sure entry is not NaN
                        if direction == 'long':
                            boundary = row['m7b_lower']
                            if pd.notna(boundary):  # Make sure boundary is not NaN
                                print(f"Long trade entry: {entry:.2f}, M7B lower: {boundary:.2f}, Entry < M7B lower: {entry < boundary}")
                        else:
                            boundary = row['m7b_upper']
                            if pd.notna(boundary):  # Make sure boundary is not NaN
                                print(f"Short trade entry: {entry:.2f}, M7B upper: {boundary:.2f}, Entry > M7B upper: {entry > boundary}")
        
        # Print by reason if available
        if 'm7b_reason' in results_df.columns:
            # Count invalidation reasons - Fixed: Use fillna in boolean filtering
            m7b_invalid_trades = results_df[results_df['m7b_invalid'].fillna(False) & results_df['unfiltered_trade_entry'].notna()]
            if not m7b_invalid_trades.empty:
                reason_counts = m7b_invalid_trades['m7b_reason'].value_counts()
                print("\nM7B invalidation reasons:")
                for reason, count in reason_counts.items():
                    if reason and pd.notna(reason) and count > 0:  # Only print non-empty reasons
                        pct = count / m7b_counts * 100 if m7b_counts > 0 else 0
                        print(f"  - {reason}: {count} ({pct:.2f}%)")
        
        # By day of week
        print("\nM7B filtered trades by day of week:")
        # Group by day of week and count trades and filtered trades - Fixed: Use fillna in boolean filtering
        day_counts = {}
        for day in weekday_order:
            day_trades = results_df[(results_df['day_of_week'] == day) & results_df['unfiltered_trade_entry'].notna()]
            # Use fillna to handle NaN values in boolean filtering
            day_filtered = day_trades[day_trades['m7b_invalid'].fillna(False)].shape[0]
            day_total = day_trades.shape[0]
            if day_total > 0:
                day_counts[day] = (day_filtered, day_total, day_filtered/day_total*100)
        
        for day, (filtered, total, pct) in day_counts.items():
            print(f"  - {day}: {filtered}/{total} ({pct:.2f}%)")
        
        # By trade direction
        print("\nM7B filtered trades by direction:")
        # Group by trade direction and count trades and filtered trades - Fixed: Use fillna in boolean filtering
        direction_counts = {}
        for direction in ['long', 'short']:
            dir_trades = results_df[(results_df['trade_direction'] == direction) & results_df['unfiltered_trade_entry'].notna()]
            # Use fillna to handle NaN values in boolean filtering
            dir_filtered = dir_trades[dir_trades['m7b_invalid'].fillna(False)].shape[0]
            dir_total = dir_trades.shape[0]
            if dir_total > 0:
                direction_counts[direction] = (dir_filtered, dir_total, dir_filtered/dir_total*100)
        
        for direction, (filtered, total, pct) in direction_counts.items():
            print(f"  - {direction.capitalize()}: {filtered}/{total} ({pct:.2f}%)")

    # Print WDR statistics if filter is enabled
    if params["use_wdr_filter"] and 'wdr_session_invalid' in results_df.columns:
        session_counts = results_df['wdr_session_invalid'].sum()
        # Fixed: Use fillna in boolean filtering for wdr_cluster_invalid
        cluster_counts = results_df[~results_df['wdr_session_invalid'] & results_df['wdr_cluster_invalid'].fillna(False)].shape[0]
        
        print("\n" + "=" * 100)
        print(" " * 35 + "WDR FILTER STATISTICS")
        print("=" * 100)
        
        # Print by reason if available
        if 'wdr_reason' in results_df.columns:
            # Count session invalidation reasons
            session_invalid = results_df[results_df['wdr_session_invalid']]
            if not session_invalid.empty:
                reason_counts = session_invalid['wdr_reason'].value_counts()
                print("\nSession invalidation reasons:")
                for reason, count in reason_counts.items():
                    if reason and pd.notna(reason) and count > 0:  # Only print non-empty reasons
                        pct = count / session_counts * 100 if session_counts > 0 else 0
                        print(f"  - {reason}: {count} ({pct:.2f}%)")
            
            # Count cluster invalidation reasons - Fixed: Use fillna in boolean filtering
            cluster_invalid = results_df[~results_df['wdr_session_invalid'] & results_df['wdr_cluster_invalid'].fillna(False)]
            if not cluster_invalid.empty:
                cluster_reason_counts = cluster_invalid['wdr_reason'].value_counts()
                print("\nCluster invalidation reasons:")
                for reason, count in cluster_reason_counts.items():
                    if reason and pd.notna(reason) and count > 0:  # Only print non-empty reasons
                        pct = count / cluster_counts * 100 if cluster_counts > 0 else 0
                        print(f"  - {reason}: {count} ({pct:.2f}%)")
        
        # By day of week
        print("\nWDR filtered sessions by day of week:")
        day_total = results_df.groupby('day_of_week').size()
        day_invalid = session_invalid.groupby('day_of_week').size() if 'session_invalid' in locals() and not session_invalid.empty else pd.Series()
        
        for day in weekday_order:
            if day in day_total.index:
                invalid_count = day_invalid.get(day, 0)
                total_count = day_total[day]
                invalid_pct = invalid_count / total_count * 100 if total_count > 0 else 0
                print(f"  - {day}: {invalid_count}/{total_count} ({invalid_pct:.2f}%)")
        
        # By cluster level - Fixed: Use fillna in boolean filtering
        if 'cluster_invalid' in locals() and not cluster_invalid.empty:
            print("\nClusters filtered by level:")
            cluster_configs = cluster_invalid.groupby(['candidate_cluster', 'trade_direction']).size()
            
            for (cluster, direction), count in sorted(cluster_configs.items()):
                similar_configs = results_df[(results_df['candidate_cluster'] == cluster) & 
                                            (results_df['trade_direction'] == direction)]
                total_count = len(similar_configs)
                filter_pct = count / total_count * 100 if total_count > 0 else 0
                print(f"  - Cluster {cluster:+.1f} ({direction.capitalize()}): {count}/{total_count} ({filter_pct:.2f}%)")

# =========================== MAIN FUNCTION ===========================

def main():
    # Get user parameters through interactive prompts
    params = get_user_input()
    
    # Import necessary visualization libraries if needed
    if params['generate_equity_curve'] or params['perform_param_sweep'] or params['perform_walk_forward']:
        if not MATPLOTLIB_AVAILABLE:
            print("Warning: Matplotlib is not available. Visualizations will be disabled.")
            params['generate_equity_curve'] = False
            params['perform_param_sweep'] = False
            params['perform_walk_forward'] = False
    
    # Create output directory if needed and doesn't exist
    if params['generate_equity_curve'] or params['perform_param_sweep'] or params['perform_walk_forward']:
        os.makedirs(params['output_dir'], exist_ok=True)
    
    # Data Import & Preprocessing
    print(f"Loading data from {params['data_file']}...")
    cols = ["date_str", "time_str", "open", "high", "low", "close", "volume"]

    try:
        # Use GPU acceleration if enabled
        if params['use_gpu'] and GPU_AVAILABLE:
            print("Using GPU for data loading and preprocessing...")
            # Use cuDF to read the CSV if available, otherwise use CuPy + pandas
            try:
                if CUDF_AVAILABLE:
                    # cuDF version
                    df = cudf.read_csv(params['data_file'], delimiter=";", header=None, names=cols)
                    # Convert timestamp
                    df["timestamp"] = cudf.to_datetime(df["date_str"] + " " + df["time_str"], format="%d/%m/%Y %H:%M")
                    df = df.drop(columns=["date_str", "time_str"])
                    df["date"] = df["timestamp"].dt.date
                    
                    # Convert back to pandas for the rest of the processing since all our functions expect pandas
                    df = df.to_pandas()
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df["timestamp"] = df["timestamp"].dt.tz_localize("America/New_York")
                else:
                    # CuPy + pandas version (CuPy for computation, pandas for dataframes)
                    print("Using pandas for dataframes and CuPy for computations")
                    df = pd.read_csv(params['data_file'], delimiter=";", header=None, names=cols)
                    df["timestamp"] = pd.to_datetime(df["date_str"] + " " + df["time_str"], format="%d/%m/%Y %H:%M")
                    df.drop(columns=["date_str", "time_str"], inplace=True)
                    df["timestamp"] = df["timestamp"].dt.tz_localize("America/New_York")
                    df["date"] = df["timestamp"].dt.date
                    # We'll use CuPy for numerical computations later
            except Exception as e:
                print(f"Error using GPU for data loading: {e}")
                print("Falling back to CPU for data loading...")
                # Fall back to pandas
                df = pd.read_csv(params['data_file'], delimiter=";", header=None, names=cols)
                df["timestamp"] = pd.to_datetime(df["date_str"] + " " + df["time_str"], format="%d/%m/%Y %H:%M")
                df.drop(columns=["date_str", "time_str"], inplace=True)
                df["timestamp"] = df["timestamp"].dt.tz_localize("America/New_York")
                df["date"] = df["timestamp"].dt.date
        else:
            # Standard pandas loading
            df = pd.read_csv(params['data_file'], delimiter=";", header=None, names=cols)
            df["timestamp"] = pd.to_datetime(df["date_str"] + " " + df["time_str"], format="%d/%m/%Y %H:%M")
            df.drop(columns=["date_str", "time_str"], inplace=True)
            df["timestamp"] = df["timestamp"].dt.tz_localize("America/New_York")
            df["date"] = df["timestamp"].dt.date
    except Exception as e:
        print(f"Error loading data file: {e}")
        sys.exit(1)
    
    print(f"Loaded {len(df)} data points across {len(df['date'].unique())} trading days")
    
    # Filter by selected days if specified
    if params['selected_days'] is not None:
        print(f"Filtering for selected days: {params['selected_days']}")
        filtered_dates = []
        for date in df["date"].unique():
            if pd.Timestamp(date).day_name() in params['selected_days']:
                filtered_dates.append(date)
        df = df[df["date"].isin(filtered_dates)]
        print(f"After day filtering: {len(df)} data points across {len(df['date'].unique())} trading days")
    
    # Display strategy parameters
    print("\nStrategy Parameters:")
    print(f"Analysis Mode: {params['analysis_mode']}")
    print(f"Cluster Levels: {params['cluster_levels']}")
    print(f"TP Multiple: {params['tp_multiple']}")
    print(f"SL Distance: {params['sl_distance']}")
    print(f"SL Extra Ticks: {params['sl_extra_ticks']}")
    print(f"Use WDR Filter: {params['use_wdr_filter']}")
    print(f"Use M7B Filter: {params['use_m7b_filter']}")
    print(f"Long Confirm Window: {params['long_confirm_start']} to {params['long_confirm_end']}")
    print(f"Short Confirm Window: {params['short_confirm_start']} to {params['short_confirm_end']}")
    
    # Run parameter sweep if requested
    if params['perform_param_sweep']:
        perform_parameter_sweep(df, params)
    
    # Run walk-forward analysis if requested
    if params['perform_walk_forward']:
        perform_walk_forward_analysis(df, params)
    
    # Process all sessions for all candidate clusters
    print("\nProcessing all trading sessions...")
    results = process_all_sessions(df, params['cluster_levels'], params)
    results_df = pd.DataFrame(results)
    
    if results_df.empty:
        print("No valid sessions were processed.")
        return
    
    # Create a copy of the data with filters applied
    filtered_df = results_df.copy()
    
    # Apply WDR filter if enabled
    if params["use_wdr_filter"] and 'wdr_session_invalid' in filtered_df.columns and 'wdr_cluster_invalid' in filtered_df.columns:
        # Track filtering statistics
        total_sessions = len(results_df)
        sessions_to_filter = filtered_df['wdr_session_invalid'].sum()
        # Use fillna to handle NaN values in boolean filtering
        clusters_to_filter = filtered_df[~filtered_df['wdr_session_invalid'] & filtered_df['wdr_cluster_invalid'].fillna(False)].shape[0]
        
        # Now apply the WDR filtering by setting trade values to NaN based on WDR rules
        session_mask = filtered_df['wdr_session_invalid']
        # Use fillna to handle NaN values in boolean filtering
        cluster_mask = filtered_df['wdr_cluster_invalid'].fillna(False)
        
        # Force these columns to NaN for filtered rows
        for col in ['trade_entry', 'SL', 'trade_exit', 'exit_reason', 'pnl_price', 'pnl_R', 'pnl_ticks']:
            filtered_df.loc[session_mask | cluster_mask, col] = np.nan
        
        # Print initial filtering stats
        print(f"\nWDR Filtering Applied:")
        print(f"  - {sessions_to_filter} sessions invalid ({sessions_to_filter/total_sessions*100 if total_sessions > 0 else 0:.2f}%)")
        print(f"  - {clusters_to_filter} additional clusters invalid")
        
        # Count trades before and after filtering
        unfiltered_trades = results_df['trade_entry'].notna().sum()
        filtered_trades = filtered_df['trade_entry'].notna().sum()
        trades_removed = unfiltered_trades - filtered_trades
        
        print(f"Trades affected: {trades_removed}/{unfiltered_trades} ({trades_removed/unfiltered_trades*100 if unfiltered_trades > 0 else 0:.2f}% removed)")
    
    # Display results based on enabled output sections
    # Note: Some functions return values needed by other functions, so we need to capture them
    unfiltered_valid_trades_df, filtered_valid_trades = display_overall_stats(results_df, filtered_df, params)
    display_day_performance(unfiltered_valid_trades_df, filtered_valid_trades, params)
    config_metrics_df, filtered_config_metrics_df = display_top_configs(results_df, filtered_df, unfiltered_valid_trades_df, filtered_valid_trades, params)
    display_target_configs(config_metrics_df, filtered_config_metrics_df, params)
    display_raw_data(results_df, params)
    display_filter_stats(results_df, params)
    
    # Generate equity curves if requested
    if params['generate_equity_curve']:
        print("\nGenerating equity curves...")
        if not unfiltered_valid_trades_df.empty:
            generate_equity_curve(unfiltered_valid_trades_df, params['output_dir'], "All_Trades_Unfiltered")
            
            # Generate separate curves for long and short trades if in 'both' mode
            if params["analysis_mode"] == "both":
                long_trades = unfiltered_valid_trades_df[unfiltered_valid_trades_df['trade_direction'] == "long"]
                short_trades = unfiltered_valid_trades_df[unfiltered_valid_trades_df['trade_direction'] == "short"]
                
                if not long_trades.empty:
                    generate_equity_curve(long_trades, params['output_dir'], "Long_Trades_Unfiltered")
                if not short_trades.empty:
                    generate_equity_curve(short_trades, params['output_dir'], "Short_Trades_Unfiltered")
        
        if (params["use_wdr_filter"] or params["use_m7b_filter"]) and not filtered_valid_trades.empty:
            generate_equity_curve(filtered_valid_trades, params['output_dir'], "All_Trades_Filtered")
            
            # Generate separate curves for long and short trades if in 'both' mode
            if params["analysis_mode"] == "both":
                long_trades = filtered_valid_trades[filtered_valid_trades['trade_direction'] == "long"]
                short_trades = filtered_valid_trades[filtered_valid_trades['trade_direction'] == "short"]
                
                if not long_trades.empty:
                    generate_equity_curve(long_trades, params['output_dir'], "Long_Trades_Filtered")
                if not short_trades.empty:
                    generate_equity_curve(short_trades, params['output_dir'], "Short_Trades_Filtered")
        
        # Generate equity curves for top configurations
        if config_metrics_df is not None and not config_metrics_df.empty:
            top_config = config_metrics_df.sort_values('total_R', ascending=False).iloc[0]
            day, time, direction, cluster = top_config['day'], top_config['time'], top_config['direction'], top_config['cluster']
            
            top_trades = unfiltered_valid_trades_df[
                (unfiltered_valid_trades_df['day_of_week'] == day) & 
                (unfiltered_valid_trades_df['confirm_time_str'] == time) & 
                (unfiltered_valid_trades_df['trade_direction'] == direction) & 
                (unfiltered_valid_trades_df['candidate_cluster'] == cluster)
            ]
            
            if not top_trades.empty:
                generate_equity_curve(top_trades, params['output_dir'], f"Top_Config_{day}_{time}_{direction}_Cluster{cluster}")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
