# Rumor Detection with CSI Model

This project implements CSI model for detecting rumors in social media. The database 

## How to Run

1. **Install dependencies** (you'll need PyTorch, pandas, scikit-learn, sentence-transformers, and other standard ML libraries)

2. **Prepare the PHEME dataset** and place it in the directory specified by `RP.config.RAW_DATA_DIR` in the config.py file

3. **Run the training**:
   ```python
   python main.py
   ```

The code will:
- Automatically detect your device (MPS/CUDA/CPU)
- Load and preprocess the PHEME rumor dataset
- Train the CSI model
- Save the best model checkpoint as `best_model.pth`

## Configuration

Modify parameters in `main.py`:
- `bin_size`: Time bin size for temporal aggregation
- `dim_hidden`: hidden dimension
- `dim_v_j`: Article representation dimension
- `learning_rate`: Training learning rate
- `lambda_reg`: L2 regularization strength
