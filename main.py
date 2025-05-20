import rumorProject as RP
import torch


"""
potential issues : non related events, users might not be
connected that much (different assumptions than in csi implementation)
check distribution in eta delta t maybe outiliers and need stonger 
normalization, also it should be 0 for the lstm not for wa ?? so x_tt ??




-

"""


if __name__ == "__main__":


    """data = pkg.DataBase(1, 50, use_checkpoints=False)
    print(data.article_seq["rumours-552783238415265792"])"""

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    

    database = RP.DataBase(
        bin_size=1,
        save_file_name="rumor_data_julesv4" # Changed save file name to force recomputation
    )


    
    trainer = RP.Trainer( 
        database=database, 
        device=device,
        dim_hidden=50,
        dim_v_j=100,
        learning_rate=0.001,
        lambda_reg=0.0001, # Reduced regularization strength
        reg_all=False # Revert to original regularization on Wu only
    )
    
    trainer.train(num_epochs=100)
