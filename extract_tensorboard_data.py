import os
import re
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOG_DIR = "./rumorProject/data/logs/tensorboard/"

def parse_run_name(run_name):
    params = {
        'simple_model': run_name.startswith('Simple_'),
        'seqlen': None,
        'dimvj': None,
        'dimh': None,
        'frac': None,
        'lr': None,
        'reg': None,
        'regall': None,
        'alpha': None,
        'balance': None # Added for balance parameter
    }
    
    # Remove 'Simple_' prefix if present
    clean_name = run_name.replace('Simple_', '')

    # Regex to extract parameters
    match = re.search(r'seqlen(\d+)_dimvj(\d+)_dimh(\d+)_frac([\d\.]+)_lr([\d\.]+)_reg([\d\.]+)_regall(True|False)(_alpha(\d+))?(_balance(True|False))?', clean_name)
    if match:
        params['seqlen'] = int(match.group(1))
        params['dimvj'] = int(match.group(2))
        params['dimh'] = int(match.group(3))
        params['frac'] = float(match.group(4))
        params['lr'] = float(match.group(5))
        params['reg'] = float(match.group(6))
        params['regall'] = match.group(7) == 'True'
        if match.group(9): # alpha
            params['alpha'] = int(match.group(9))
        if match.group(11): # balance
            params['balance'] = match.group(11) == 'True'
    else:
        # Handle older formats or runs without all parameters
        # Attempt to parse partial information if full regex fails
        seqlen_match = re.search(r'seqlen(\d+)', clean_name)
        if seqlen_match:
            params['seqlen'] = int(seqlen_match.group(1))
        dimvj_match = re.search(r'dimvj(\d+)', clean_name)
        if dimvj_match:
            params['dimvj'] = int(dimvj_match.group(1))
        dimh_match = re.search(r'dimh(\d+)', clean_name)
        if dimh_match:
            params['dimh'] = int(dimh_match.group(1))
        frac_match = re.search(r'frac([\d\.]+)', clean_name)
        if frac_match:
            params['frac'] = float(frac_match.group(1))
        lr_match = re.search(r'lr([\d\.]+)', clean_name)
        if lr_match:
            params['lr'] = float(lr_match.group(1))
        reg_match = re.search(r'reg([\d\.]+)', clean_name)
        if reg_match:
            params['reg'] = float(reg_match.group(1))
        regall_match = re.search(r'regall(True|False)', clean_name)
        if regall_match:
            params['regall'] = regall_match.group(1) == 'True'
        alpha_match = re.search(r'alpha(\d+)', clean_name)
        if alpha_match:
            params['alpha'] = int(alpha_match.group(1))
        balance_match = re.search(r'balance(True|False)', clean_name)
        if balance_match:
            params['balance'] = balance_match.group(1) == 'True'

    return params


import os
import re
import json
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOG_DIR = "./rumorProject/data/logs/tensorboard/"
OUTPUT_DATA_FILE = "./report_results/tensorboard_data.json"

def parse_run_name(run_name):
    params = {
        'simple_model': run_name.startswith('Simple_'),
        'seqlen': None,
        'dimvj': None,
        'dimh': None,
        'frac': None,
        'lr': None,
        'reg': None,
        'regall': None,
        'alpha': None,
        'balance': None
    }
    
    clean_name = run_name.replace('Simple_', '')

    match = re.search(r'seqlen(\d+)_dimvj(\d+)_dimh(\d+)_frac([\d\.]+)_lr([\d\.]+)_reg([\d\.]+)_regall(True|False)(_alpha(\d+))?(_balance(True|False))?', clean_name)
    if match:
        params['seqlen'] = int(match.group(1))
        params['dimvj'] = int(match.group(2))
        params['dimh'] = int(match.group(3))
        params['frac'] = float(match.group(4))
        params['lr'] = float(match.group(5))
        params['reg'] = float(match.group(6))
        params['regall'] = match.group(7) == 'True'
        if match.group(9):
            params['alpha'] = int(match.group(9))
        if match.group(11):
            params['balance'] = match.group(11) == 'True'
    else:
        # Attempt to parse partial information if full regex fails
        seqlen_match = re.search(r'seqlen(\d+)', clean_name)
        if seqlen_match: params['seqlen'] = int(seqlen_match.group(1))
        dimvj_match = re.search(r'dimvj(\d+)', clean_name)
        if dimvj_match: params['dimvj'] = int(dimvj_match.group(1))
        dimh_match = re.search(r'dimh(\d+)', clean_name)
        if dimh_match: params['dimh'] = int(dimh_match.group(1))
        frac_match = re.search(r'frac([\d\.]+)', clean_name)
        if frac_match: params['frac'] = float(frac_match.group(1))
        lr_match = re.search(r'lr([\d\.]+)', clean_name)
        if lr_match: params['lr'] = float(lr_match.group(1))
        reg_match = re.search(r'reg([\d\.]+)', clean_name)
        if reg_match: params['reg'] = float(reg_match.group(1))
        regall_match = re.search(r'regall(True|False)', clean_name)
        if regall_match: params['regall'] = regall_match.group(1) == 'True'
        alpha_match = re.search(r'alpha(\d+)', clean_name)
        if alpha_match: params['alpha'] = int(alpha_match.group(1))
        balance_match = re.search(r'balance(True|False)', clean_name)
        if balance_match: params['balance'] = balance_match.group(1) == 'True'

    return params


def extract_tensorboard_data(log_dir):
    all_runs_data = []
    for run_name in os.listdir(log_dir):
        run_path = os.path.join(log_dir, run_name)
        if os.path.isdir(run_path):
            try:
                event_acc = EventAccumulator(run_path)
                event_acc.Reload()

                run_params = parse_run_name(run_name)
                run_data = {
                    'run_name': run_name,
                    'params': run_params,
                    'max_accuracy': None,
                    'accuracy_history': [],
                    'train_accuracy_history': [],
                    'validation_loss_history': [],
                    'train_loss_history': []
                }

                if 'Accuracy/Validation' in event_acc.Tags()['scalars']:
                    val_accuracies = event_acc.Scalars('Accuracy/Validation')
                    if val_accuracies:
                        run_data['max_accuracy'] = max([s.value for s in val_accuracies])
                        run_data['accuracy_history'] = [{'step': s.step, 'value': s.value} for s in val_accuracies]
                
                if 'Accuracy/Train' in event_acc.Tags()['scalars']:
                    train_accuracies = event_acc.Scalars('Accuracy/Train')
                    if train_accuracies:
                        run_data['train_accuracy_history'] = [{'step': s.step, 'value': s.value} for s in train_accuracies]

                if 'Loss/Validation' in event_acc.Tags()['scalars']:
                    val_losses = event_acc.Scalars('Loss/validation') # Note: 'Loss/validation' is used in trainer.py
                    if val_losses:
                        run_data['validation_loss_history'] = [{'step': s.step, 'value': s.value} for s in val_losses]

                if 'Loss/Train' in event_acc.Tags()['scalars']:
                    train_losses = event_acc.Scalars('Loss/Train')
                    if train_losses:
                        run_data['train_loss_history'] = [{'step': s.step, 'value': s.value} for s in train_losses]
                
                if run_data['max_accuracy'] is not None: # Only add runs that have at least validation accuracy
                    all_runs_data.append(run_data)
                else:
                    print(f"Warning: No 'Accuracy/Validation' found for {run_name}")
            except Exception as e:
                print(f"Error processing {run_name}: {e}")
    return all_runs_data

if __name__ == "__main__":
    print(f"Extracting data from: {LOG_DIR}")
    all_results = extract_tensorboard_data(LOG_DIR)

    # Sort all results by max_accuracy in descending order
    sorted_results = sorted(all_results, key=lambda x: x['max_accuracy'], reverse=True)

    # Select top 6 runs for the table and plot, prioritizing diversity
    selected_runs = []
    csi_count = 0
    simple_csi_count = 0
    regall_true_count = 0
    regall_false_count = 0

    excluded_run_name = "seqlen101_dimvj102_dimh50_frac1_lr0.001_reg0.01_regallTrue_balanceTrue"
    
    filtered_results = [run for run in sorted_results if run['run_name'] != excluded_run_name]

    # Select top 6 runs for the table and plot, prioritizing diversity
    selected_runs = []
    csi_count = 0
    simple_csi_count = 0
    regall_true_count = 0
    regall_false_count = 0

    for run in filtered_results: # Iterate over filtered results
        if len(selected_runs) >= 6:
            break
        
        is_csi = not run['params']['simple_model']
        is_regall_true = run['params']['regall']
        
        # Try to get a mix: at least 2 of each type (CSI/Simple, RegAll True/False) if possible
        if is_csi and csi_count < 3: # Aim for up to 3 CSI models
            selected_runs.append(run)
            csi_count += 1
        elif not is_csi and simple_csi_count < 3: # Aim for up to 3 Simple CSI models
            selected_runs.append(run)
            simple_csi_count += 1
        elif is_regall_true and regall_true_count < 3: # If not added by model type, try by RegAll
            selected_runs.append(run)
            regall_true_count += 1
        elif not is_regall_true and regall_false_count < 3:
            selected_runs.append(run)
            regall_false_count += 1
        elif len(selected_runs) < 6: # Fallback: just add if less than 6
            selected_runs.append(run)

    # Ensure unique runs in case of fallback adding duplicates (should be less likely with filter)
    unique_selected_runs = []
    seen_run_names = set()
    for run in selected_runs:
        if run['run_name'] not in seen_run_names:
            unique_selected_runs.append(run)
            seen_run_names.add(run['run_name'])
    selected_runs = unique_selected_runs[:6] # Final trim to 6 if more were added by fallback

    print("\n--- Selected Runs for Table and Plot ---")
    for run in selected_runs:
        print(f"Run: {run['run_name']}, Max Accuracy: {run['max_accuracy']:.4f}, Params: {run['params']}")

    # Save selected data to a JSON file for plotting script
    with open(OUTPUT_DATA_FILE, 'w') as f:
        json.dump(selected_runs, f, indent=4)
    print(f"\nSelected data saved to {OUTPUT_DATA_FILE}")

    # Print data for LaTeX table
    print("\n--- LaTeX Table Data ---")
    for run in selected_runs:
        model_type = "CSI" if not run['params']['simple_model'] else "Simple CSI (CI)"
        seqlen = run['params'].get('seqlen', 'N/A')
        dimvj = run['params'].get('dimvj', 'N/A')
        dimh = run['params'].get('dimh', 'N/A')
        lr = run['params'].get('lr', 'N/A')
        regall = "True" if run['params'].get('regall') else "False" if run['params'].get('regall') is not None else "N/A"
        
        print(f"{model_type} & {run['max_accuracy']:.2f}\% & {seqlen} & {dimvj} & {dimh} & {lr} & {regall} \\\\")

    # Comparison of RegAll (re-using previous logic for summary)
    csi_models = {run['run_name']: run for run in all_results if not run['params']['simple_model']}
    simple_csi_models = {run['run_name']: run for run in all_results if run['params']['simple_model']}

    print("\n--- Comparison: RegAll Enabled vs Disabled (Summary) ---")
    regall_enabled_csi = {k: v for k, v in csi_models.items() if v['params']['regall'] is True}
    regall_disabled_csi = {k: v for k, v in csi_models.items() if v['params']['regall'] is False}

    if regall_enabled_csi:
        max_regall_enabled_acc = max(v['max_accuracy'] for v in regall_enabled_csi.values())
        print(f"Max CSI with RegAll=True: {max_regall_enabled_acc:.4f}")
    else:
        print("No CSI runs with RegAll=True found.")

    if regall_disabled_csi:
        max_regall_disabled_acc = max(v['max_accuracy'] for v in regall_disabled_csi.values())
        print(f"Max CSI with RegAll=False: {max_regall_disabled_acc:.4f}")
    else:
        print("No CSI runs with RegAll=False found.")

    regall_enabled_simple_csi = {k: v for k, v in simple_csi_models.items() if v['params']['regall'] is True}
    regall_disabled_simple_csi = {k: v for k, v in simple_csi_models.items() if v['params']['regall'] is False}

    if regall_enabled_simple_csi:
        max_regall_enabled_simple_acc = max(v['max_accuracy'] for v in regall_enabled_simple_csi.values())
        print(f"Max Simple CSI (CI) with RegAll=True: {max_regall_enabled_simple_acc:.4f}")
    else:
        print("No Simple CSI (CI) runs with RegAll=True found.")

    if regall_disabled_simple_csi:
        max_regall_disabled_simple_acc = max(v['max_accuracy'] for v in regall_disabled_simple_csi.values())
        print(f"Max Simple CSI (CI) with RegAll=False: {max_regall_disabled_simple_acc:.4f}")
    else:
        print("No Simple CSI (CI) runs with RegAll=False found.")
