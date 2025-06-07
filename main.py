import rumorProject as RP
import torch
torch.manual_seed(RP.config.SEED_TORCH)


"""
weibo:
number of tweets: 3805656
number of events: 4664
number of users: 2856741

pheme:
number of  tweets: 102440 
Number of threads: 5802
Number of users: 49345



"""


if __name__ == "__main__":


    


    if torch.backends.mps.is_available():
        torch.mps.manual_seed(RP.config.SEED_TORCH)
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    


    """database = RP.WeiboDataBase(
        bin_size=1,
        save_df_file_name="weibo_df_cache1",
        save_precomputed_file_name="weibo_precomputed_1",
        device=device,
        event_fraction=1 # Reduced event fraction
    )"""

    database = RP.DataBase(
        bin_size=1,
        save_file_name="precompute_final1", # Changed save file name to force recomputation
        device=device
    )

    
    trainer = RP.Trainer( 
        database=database, 
        device=device,
        dim_hidden=50,
        dim_v_j=100,
        learning_rate=0.001,
        max_seq_len=20, #101 for saying now we biase p_j learning
        lambda_reg=0.001, # Reduced regularization strength
        reg_all=False, # Revert to original regularization on Wu only
        simple_model=False, # Use simple model for faster training
        alpha=10
    )
    
    trainer.train(num_epochs=100)
