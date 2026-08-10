from __future__ import annotations

import torch
from TADFVAE.to_rjt import AutoEncoder, AETrainer, make_npz_file, get_dataset_dataloader, score
# print(torch.cuda.device_count())  

class TADF_score_try():
    def __init__(self):
        super().__init__()
        # 定义 self.device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 你的 AutoEncoder, AETrainer
        self.autoencoder = AutoEncoder(200).to(self.device)
        
        # 路径要根据自己的实际情况来
        save_model = "/home/xinxinniu/2025_Mol_Prompt/RJT-RL-master/rjt_rl/rl/rewards/TADFVAE/trained_model/TADF_self_check.pt"        
        self.trainer = AETrainer(
            self.autoencoder,
            None,
            optimizer_name=None,
            lr=1e-3,
            n_epochs=1,
            lr_milestones=(),
            batch_size=1,
            weight_decay=0.0,
            save_model=save_model,
            device=self.device,
            lr_decay=1.0,
        )
        self.trainer.ae_net.load_state_dict(torch.load(save_model))
        self.trainer.ae_net.eval()

    def calc_TADF_score(self, smiles) -> float:  
        data = make_npz_file(smiles)
        try:
            _, unlabel_xs, _, _ = get_dataset_dataloader(data, batch_size=1, num_workers=1)
            unlab = score(self.trainer, unlabel_xs, self.device).cpu().detach().numpy().reshape(-1)
        except Exception as e:
            print("Error in calc_TADF_score:", e)
            unlab = 0
        return unlab
    
import pandas as pd
def calculate_TADF_scores_from_csv(csv_file: str):
    # Load CSV file
    df = pd.read_csv(csv_file)
    # Instantiate the TADF scorer
    tadf_scorer = TADF_score_try()
    
    # Calculate TADF scores for each SMILES string in the 'smiles' column
    scores = []
    for smiles in df['smiles_step-1']:
        try:
            score = tadf_scorer.calc_TADF_score(smiles)
            scores.append(score[0])  # Append the first score (as the result is an array)
        except:
            scores.append(-100)
    
    # Add the scores to the DataFrame
    df['TADF_score'] = scores
    return df
    
csv_file = '/home/xinxinniu/2025_Mol_Prompt/chemprop-uncertainty/results/S1_exc_0321/EST_CO_C=O_CN_C#N_est_unique_smiles_step-1.csv'  # Replace with your CSV file path
result_df = calculate_TADF_scores_from_csv(csv_file)
result_df.to_csv('/home/xinxinniu/2025_Mol_Prompt/chemprop-uncertainty/results/S1_exc_0321/EST_CO_C=O_CN_C#N_est_TADF_smiles_step-1.csv',index=None)
