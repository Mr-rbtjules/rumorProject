import json
import matplotlib.pyplot as plt
import os

DATA_FILE = "./report_results/tensorboard_data.json"
VALIDATION_ACCURACY_PLOT_FILE = "./report_results/validation_accuracy_plot.png"
LOSS_OVERFITTING_PLOT_FILE = "./report_results/loss_overfitting_plot.png"

def plot_validation_accuracies(data):
    plt.figure(figsize=(12, 7))
    
    for run in data:
        history = run['accuracy_history']
        steps = [item['step'] for item in history]
        values = [item['value'] for item in history]
        
        model_type = "CSI" if not run['params']['simple_model'] else "Simple CSI (CI)"
        run_label = f"{model_type} (Max Acc: {run['max_accuracy']:.2f}%)"
        
        # Add key hyperparameters to label for clarity
        params = run['params']
        label_parts = [run_label]
        if params.get('seqlen') is not None: label_parts.append(f"SeqLen:{params['seqlen']}")
        if params.get('dimvj') is not None: label_parts.append(f"DimVJ:{params['dimvj']}")
        if params.get('dimh') is not None: label_parts.append(f"DimH:{params['dimh']}")
        if params.get('lr') is not None: label_parts.append(f"LR:{params['lr']}")
        if params.get('regall') is not None: label_parts.append(f"RegAll:{params['regall']}")
        
        plt.plot(steps, values, label=", ".join(label_parts))

    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.title('Evolution of Validation Accuracy for Selected Models')
    plt.legend(loc='lower right', fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(VALIDATION_ACCURACY_PLOT_FILE)
    print(f"Validation accuracy plot saved to {VALIDATION_ACCURACY_PLOT_FILE}")

def plot_loss_overfitting_analysis(data):
    # Find the best performing model based on max_accuracy
    best_run = max(data, key=lambda x: x['max_accuracy'])
    
    model_type = "CSI" if not best_run['params']['simple_model'] else "Simple CSI (CI)"
    run_label_base = f"{model_type} (Max Acc: {best_run['max_accuracy']:.2f}%)"
    
    plt.figure(figsize=(10, 7)) # Single plot for loss
    
    # Plot Loss
    if best_run['train_loss_history']:
        train_loss_steps = [item['step'] for item in best_run['train_loss_history']]
        train_loss_values = [item['value'] for item in best_run['train_loss_history']]
        plt.plot(train_loss_steps, train_loss_values, label='Training Loss', color='blue')

    if best_run['validation_loss_history']:
        val_loss_steps = [item['step'] for item in best_run['validation_loss_history']]
        val_loss_values = [item['value'] for item in best_run['validation_loss_history']]
        plt.plot(val_loss_steps, val_loss_values, label='Validation Loss', color='orange')
    else:
        print(f"Warning: No validation loss history found for the best model: {best_run['run_name']}")


    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss Evolution for Best Model: {run_label_base}')
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(LOSS_OVERFITTING_PLOT_FILE)
    print(f"Loss overfitting analysis plot saved to {LOSS_OVERFITTING_PLOT_FILE}")

if __name__ == "__main__":
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            all_runs_data = json.load(f)
        
        # Plot 1: All validation accuracies
        plot_validation_accuracies(all_runs_data)
        
        # Plot 2: Best model's train/val loss
        plot_loss_overfitting_analysis(all_runs_data)
    else:
        print(f"Error: Data file not found at {DATA_FILE}. Please run extract_tensorboard_data.py first.")
