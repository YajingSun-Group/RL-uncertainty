from __future__ import annotations

import networkx as nx
from rdkit import Chem
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rjt_rl.rl.envs.states import State
from .base_mol_reward import BaseMolReward
from argparse import Namespace
import csv
from typing import List, Optional
from .TADFVAE.to_rjt import *
import numpy as np
import torch
from tqdm import tqdm

from chemprop.utils import load_checkpoint, load_scalers,load_args

class MPNNReward0(BaseMolReward):
    "value"
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # cmd = set_cuda_visible_device(0)
        # os.environ["CUDA_VISIBLE_DEVICES"] = cmd
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(self.device,'self.device')
        self.model1 = load_checkpoint(config.prop_model1, cuda=self.device)
        self.model2 = load_checkpoint(config.prop_model2, cuda=self.device)
        self.model3 = load_checkpoint(config.prop_model3, cuda=self.device)
        self.scaler, features_scaler = load_scalers(config.prop_model1)
        train_args = load_args(config.prop_model1)
        self.model1.eval()
        self.model2.eval()
        self.model3.eval()

    def calc_score(self, state: State) -> float:
        mol = state.mol    
        smiles = [Chem.MolToSmiles(mol)] #'c1ccc(-c2nc(-c3ccccc3)nc(-c3ccc(-n4c5ccccc5c5c4c4ccccc4n5-c4ccccc4)cc3)n2)cc1'
        # if 'FORMAT' in self.save_model:
        #     data = mol_to_graph_data_obj_simple(mol)
        # elif 'Cho' in self.save_model:
        #     data = mol_to_graph_data_obj_simple_2(mol=mol,sol=Chem.MolFromSmiles('ClCCl'))
        # else:
        #     raise ValueError(f"penalized_logp_reward.py Unsupported save_model type: {self.save_model}")
        # data = data.to(self.device)
        with torch.no_grad():
            pred1,logvar1 = self.model1(smiles)
            pred2,logvar2 = self.model2(smiles)
            pred3,logvar3 = self.model3(smiles)
            pred1,pred2,pred3= pred1.detach().cpu().numpy(),pred2.detach().cpu().numpy(),pred3.detach().cpu().numpy()
            logvar1 = torch.exp(logvar1)
            logvar2 = torch.exp(logvar2)
            logvar3 = torch.exp(logvar3)
            logvar1,logvar2,logvar3= logvar1.detach().cpu().numpy(),logvar2.detach().cpu().numpy(),logvar3.detach().cpu().numpy()
            pred1 = self.scaler.inverse_transform(pred1).tolist()
            pred2 = self.scaler.inverse_transform(pred2).tolist()
            pred3 = self.scaler.inverse_transform(pred3).tolist()
            logvar1 = self.scaler.inverse_transform_variance(logvar1)
            logvar2 = self.scaler.inverse_transform_variance(logvar2)
            logvar3 = self.scaler.inverse_transform_variance(logvar3)

            all_preds = np.zeros((1,1,3))
            all_preds[:, :, 0] = pred1
            all_preds[:, :, 1] = pred2
            all_preds[:, :, 2] = pred3

            avg_preds = (np.array(pred1) + np.array(pred2) + np.array(pred3)) / 3
            avg_ale_uncs = (np.array(logvar1) + np.array(logvar2) + np.array(logvar3)) / 3
            avg_epi_uncs = np.var(all_preds, axis=2)

            # avg_preds,avg_ale_uncs,avg_epi_uncs= avg_preds.detach().cpu().numpy(),avg_ale_uncs.detach().cpu().numpy(),avg_epi_uncs.detach().cpu().numpy()
            # print(avg_preds,'pre2')
            state.score_dict = {"avg_preds":avg_preds[0][0],"avg_ale_uncs":avg_ale_uncs[0][0],"avg_epi_uncs":avg_epi_uncs[0][0],}
        if self.config.EPI>0:
            # if avg_epi_uncs[0][0]>0.5: #sa
            if avg_epi_uncs[0][0]>50:  #TPSA
                return 0
            else:
                return avg_preds[0][0] #EPI_value
        elif self.config.ALE>0:
            # if avg_ale_uncs[0][0]>0.5: #sa
            if avg_ale_uncs[0][0]>50: #TPSA
                return 0
            else:
                return avg_preds[0][0] #ALE_value
        else:
            return avg_preds[0][0] #随机
        


class MPNNReward(BaseMolReward):
    ""
    def __init__(self, config):
        super().__init__(config)
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # cmd = set_cuda_visible_device(0)
        # os.environ["CUDA_VISIBLE_DEVICES"] = cmd
        self.config = config
        # self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = "cuda:1" if torch.cuda.is_available() else "cpu"
        print(self.device,'self.device')
        self.model1 = load_checkpoint(config.prop_model1, cuda=self.device)
        self.model2 = load_checkpoint(config.prop_model2, cuda=self.device)
        self.model3 = load_checkpoint(config.prop_model3, cuda=self.device)
        self.scaler, features_scaler = load_scalers(config.prop_model1)
        train_args = load_args(config.prop_model1)
        self.model1.eval()
        self.model2.eval()
        self.model3.eval()
    
    @staticmethod
    def reward_exp(reward, avg_uncs, lambda_a):
        return reward * np.exp(-lambda_a * avg_uncs)
    # def reward_exp(pred, uncertainty, lambd):
    #     if pred <= 0:
    #         return 0.0
    #     return (1.0 / pred) * np.exp(-lambd * uncertainty)


    @staticmethod
    def reward_k(reward, avg_uncs, k):
        return reward / (1 + (avg_uncs) ** k)
    # def reward_k(pred, uncertainty, k):
    #     if pred <= 0:
    #         return 0.0
    #     return (1.0 / pred) / (1 + k * uncertainty)

    
    def calc_score(self, state: State) -> float:
        mol = state.mol    
        smiles = [Chem.MolToSmiles(mol)] #'c1ccc(-c2nc(-c3ccccc3)nc(-c3ccc(-n4c5ccccc5c5c4c4ccccc4n5-c4ccccc4)cc3)n2)cc1'
        # if 'FORMAT' in self.save_model:
        #     data = mol_to_graph_data_obj_simple(mol)
        # elif 'Cho' in self.save_model:
        #     data = mol_to_graph_data_obj_simple_2(mol=mol,sol=Chem.MolFromSmiles('ClCCl'))
        # else:
        #     raise ValueError(f"penalized_logp_reward.py Unsupported save_model type: {self.save_model}")
        # data = data.to(self.device)
        with torch.no_grad():
            pred1,logvar1 = self.model1(smiles)
            pred2,logvar2 = self.model2(smiles)
            pred3,logvar3 = self.model3(smiles)
            pred1,pred2,pred3= pred1.detach().cpu().numpy(),pred2.detach().cpu().numpy(),pred3.detach().cpu().numpy()
            logvar1 = torch.exp(logvar1)
            logvar2 = torch.exp(logvar2)
            logvar3 = torch.exp(logvar3)
            logvar1,logvar2,logvar3= logvar1.detach().cpu().numpy(),logvar2.detach().cpu().numpy(),logvar3.detach().cpu().numpy()
            pred1 = self.scaler.inverse_transform(pred1).tolist()
            pred2 = self.scaler.inverse_transform(pred2).tolist()
            pred3 = self.scaler.inverse_transform(pred3).tolist()
            logvar1 = self.scaler.inverse_transform_variance(logvar1)
            logvar2 = self.scaler.inverse_transform_variance(logvar2)
            logvar3 = self.scaler.inverse_transform_variance(logvar3)

            all_preds = np.zeros((1,1,3))
            all_preds[:, :, 0] = pred1
            all_preds[:, :, 1] = pred2
            all_preds[:, :, 2] = pred3

            avg_preds = (np.array(pred1) + np.array(pred2) + np.array(pred3)) / 3
            avg_ale_uncs = (np.array(logvar1) + np.array(logvar2) + np.array(logvar3)) / 3
            avg_epi_uncs = np.var(all_preds, axis=2)
            state.score_dict = {"avg_preds":avg_preds[0][0],"avg_ale_uncs":avg_ale_uncs[0][0],"avg_epi_uncs":avg_epi_uncs[0][0],}
        if (self.config.ALE + self.config.EPI) == 0.0:
            return avg_preds[0][0] #
        elif self.config.eq_exp>0:
            if self.config.EPI > 0 and self.config.ALE > 0:
                assert self.config.lambda_ale == self.config.lambda_epi
                reward_ALE_EPI=self.reward_exp(avg_preds[0][0],avg_ale_uncs[0][0]+avg_epi_uncs[0][0],self.config.lambda_ale)
                state.score_dict.update({"eqexp_ale_epi_preds":reward_ALE_EPI,})
                return reward_ALE_EPI#
            elif self.config.ALE > 0: 
                reward_ALE=self.reward_exp(avg_preds[0][0],avg_ale_uncs[0][0],self.config.lambda_ale)
                state.score_dict.update({"eqexp_ale_preds":reward_ALE,})
                return reward_ALE#
            elif self.config.EPI > 0:
                reward_EPI=self.reward_exp(avg_preds[0][0],avg_epi_uncs[0][0],self.config.lambda_epi)
                state.score_dict.update({"eqexp_epi_preds":reward_EPI,})
                return reward_EPI#
            
        elif self.config.eq_k>0:
            if self.config.EPI > 0 and self.config.ALE > 0:
                assert self.config.k_ale == self.config.k_epi
                reward_ALE_EPI=self.reward_k(avg_preds[0][0],avg_ale_uncs[0][0]+avg_epi_uncs[0][0],self.config.k_ale)
                state.score_dict.update({"eqk_ale_epi_preds":reward_ALE_EPI,})
                return reward_ALE_EPI#
            elif self.config.ALE > 0: 
                reward_ALE=self.reward_k(avg_preds[0][0],avg_ale_uncs[0][0],self.config.k_ale)
                state.score_dict.update({"eqk_ale_preds":reward_ALE,})
                return reward_ALE#
            elif self.config.EPI > 0:
                reward_EPI=self.reward_k(avg_preds[0][0],avg_epi_uncs[0][0],self.config.k_epi)
                state.score_dict.update({"eqk_epi_preds":reward_EPI,})
                return reward_EPI#            
        else :
            raise ValueError("Unexpected condition in config.ALE and config.EPI")
            
class MPNNReward_ale_epi(BaseMolReward):
    ""
    def __init__(self, config):
        super().__init__(config)
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # cmd = set_cuda_visible_device(0)
        # os.environ["CUDA_VISIBLE_DEVICES"] = cmd
        self.config = config
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(self.device,'self.device')
        self.model1 = load_checkpoint(config.prop_model1, cuda=self.device)
        self.model2 = load_checkpoint(config.prop_model2, cuda=self.device)
        self.model3 = load_checkpoint(config.prop_model3, cuda=self.device)
        self.scaler, features_scaler = load_scalers(config.prop_model1)
        train_args = load_args(config.prop_model1)
        self.model1.eval()
        self.model2.eval()
        self.model3.eval()
    
    @staticmethod
    def reward_exp(reward, avg_uncs, lambda_a):
        return reward * np.exp(-lambda_a * avg_uncs)
    # def reward_exp(pred, uncertainty, lambd):
    #     if pred <= 0:
    #         return 0.0
    #     return (1.0 / pred) * np.exp(-lambd * uncertainty)


    @staticmethod
    def reward_k(reward, avg_uncs, k):
        return reward / (1 + (avg_uncs) ** k)
    # def reward_k(pred, uncertainty, k):
    #     if pred <= 0:
    #         return 0.0
    #     return (1.0 / pred) / (1 + k * uncertainty)
    
    def calc_score(self, state: State) -> float:
        mol = state.mol    
        smiles = [Chem.MolToSmiles(mol)] #'c1ccc(-c2nc(-c3ccccc3)nc(-c3ccc(-n4c5ccccc5c5c4c4ccccc4n5-c4ccccc4)cc3)n2)cc1'
        # if 'FORMAT' in self.save_model:
        #     data = mol_to_graph_data_obj_simple(mol)
        # elif 'Cho' in self.save_model:
        #     data = mol_to_graph_data_obj_simple_2(mol=mol,sol=Chem.MolFromSmiles('ClCCl'))
        # else:
        #     raise ValueError(f"penalized_logp_reward.py Unsupported save_model type: {self.save_model}")
        # data = data.to(self.device)
        with torch.no_grad():
            pred1,logvar1 = self.model1(smiles)
            pred2,logvar2 = self.model2(smiles)
            pred3,logvar3 = self.model3(smiles)
            pred1,pred2,pred3= pred1.detach().cpu().numpy(),pred2.detach().cpu().numpy(),pred3.detach().cpu().numpy()
            logvar1 = torch.exp(logvar1)
            logvar2 = torch.exp(logvar2)
            logvar3 = torch.exp(logvar3)
            logvar1,logvar2,logvar3= logvar1.detach().cpu().numpy(),logvar2.detach().cpu().numpy(),logvar3.detach().cpu().numpy()
            pred1 = self.scaler.inverse_transform(pred1).tolist()
            pred2 = self.scaler.inverse_transform(pred2).tolist()
            pred3 = self.scaler.inverse_transform(pred3).tolist()
            logvar1 = self.scaler.inverse_transform_variance(logvar1)
            logvar2 = self.scaler.inverse_transform_variance(logvar2)
            logvar3 = self.scaler.inverse_transform_variance(logvar3)

            all_preds = np.zeros((1,1,3))
            all_preds[:, :, 0] = pred1
            all_preds[:, :, 1] = pred2
            all_preds[:, :, 2] = pred3

            avg_preds = (np.array(pred1) + np.array(pred2) + np.array(pred3)) / 3
            avg_ale_uncs = (np.array(logvar1) + np.array(logvar2) + np.array(logvar3)) / 3
            avg_epi_uncs = np.var(all_preds, axis=2)
            state.score_dict = {"avg_preds":avg_preds[0][0],"avg_ale_uncs":avg_ale_uncs[0][0],"avg_epi_uncs":avg_epi_uncs[0][0],}
        assert self.config.EPI > 0 and self.config.ALE > 0      
        if self.config.lambda_ale>0 and self.config.lambda_epi>0:
            reward_ALE=self.reward_exp(avg_preds[0][0],avg_ale_uncs[0][0],self.config.lambda_ale)
            reward_EPI=self.reward_exp(avg_preds[0][0],avg_epi_uncs[0][0],self.config.lambda_epi)
            state.score_dict.update({"eqexp_ale_preds":reward_ALE,"eqexp_epi_preds":reward_EPI,})
            return (reward_ALE+reward_EPI)/2#

        elif self.config.lambda_ale>0 and self.config.k_epi>0:
            reward_ALE=self.reward_exp(avg_preds[0][0],avg_ale_uncs[0][0],self.config.lambda_ale)
            reward_EPI=self.reward_k(avg_preds[0][0],avg_epi_uncs[0][0],self.config.k_epi)
            state.score_dict.update({"eqexp_ale_preds":reward_ALE,"eqk_epi_preds":reward_EPI,})
            return (reward_ALE+reward_EPI)/2#
        
        elif self.config.k_ale>0 and self.config.lambda_epi>0:
            reward_ALE=self.reward_k(avg_preds[0][0],avg_ale_uncs[0][0],self.config.k_ale)
            reward_EPI=self.reward_exp(avg_preds[0][0],avg_epi_uncs[0][0],self.config.lambda_epi)
            state.score_dict.update({"eqk_ale_preds":reward_ALE,"eqexp_epi_preds":reward_EPI,})
            return (reward_ALE+reward_EPI)/2#
        
        elif self.config.k_ale>0 and self.config.k_epi>0:
            reward_ALE=self.reward_k(avg_preds[0][0],avg_ale_uncs[0][0],self.config.k_ale)
            reward_EPI=self.reward_k(avg_preds[0][0],avg_epi_uncs[0][0],self.config.k_epi)
            state.score_dict.update({"eqk_ale_preds":reward_ALE,"eqk_epi_preds":reward_EPI,})
            return (reward_ALE+reward_EPI)/2#
        elif self.config.eq_exp>0 and self.config.eq_k==0:  
            reward_ALE=self.reward_exp(avg_preds[0][0],avg_ale_uncs[0][0],self.config.lambda_ale)
            reward_EPI=self.reward_exp(avg_preds[0][0],avg_epi_uncs[0][0],self.config.lambda_epi)
            state.score_dict.update({"eqexp_ale_preds":reward_ALE,"eqexp_epi_preds":reward_EPI,})
            return (reward_ALE+reward_EPI)/2#
        elif self.config.eq_exp==0 and self.config.eq_k>0:  
            reward_ALE=self.reward_k(avg_preds[0][0],avg_ale_uncs[0][0],self.config.k_ale)
            reward_EPI=self.reward_k(avg_preds[0][0],avg_epi_uncs[0][0],self.config.k_epi)
            state.score_dict.update({"eqexp_ale_preds":reward_ALE,"eqexp_epi_preds":reward_EPI,})
            return (reward_ALE+reward_EPI)/2#
        else :
            raise ValueError("Unexpected condition in config.ALE and config.EPI")

class MPNNReward_ale_epi_2(BaseMolReward):
    ""
    def __init__(self, config):
        super().__init__(config)
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # cmd = set_cuda_visible_device(0)
        # os.environ["CUDA_VISIBLE_DEVICES"] = cmd
        self.config = config
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(self.device,'self.device')
        self.model1 = load_checkpoint(config.prop_model1, cuda=self.device)
        self.model2 = load_checkpoint(config.prop_model2, cuda=self.device)
        self.model3 = load_checkpoint(config.prop_model3, cuda=self.device)
        self.model1_1 = load_checkpoint(config.prop_model1_1, cuda=self.device)
        self.model2_1 = load_checkpoint(config.prop_model2_1, cuda=self.device)
        self.model3_1 = load_checkpoint(config.prop_model3_1, cuda=self.device)
        self.scaler, features_scaler = load_scalers(config.prop_model1)
        self.scaler_1, features_scaler_1 = load_scalers(config.prop_model1_1)
        train_args = load_args(config.prop_model1)
        
        self.autoencoder = AutoEncoder(200).to(self.device)
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
        self.trainer.ae_net.load_state_dict(torch.load(save_model, weights_only=True))
        self.trainer.ae_net.eval()

        self.model1.eval()
        self.model2.eval()
        self.model3.eval()
        self.model1_1.eval()
        self.model2_1.eval()
        self.model3_1.eval()
    
    @staticmethod
    def reward_exp(reward, avg_uncs, lambda_a):
        return reward * np.exp(-lambda_a * avg_uncs)

    @staticmethod
    def reward_exp_fu(pred, uncertainty, lambd):
        if pred <= 0:
            return 0.0
        return (1.0 / pred) * np.exp(-lambd * uncertainty)


    @staticmethod
    def reward_k(reward, avg_uncs, k):
        return reward / (1 + (avg_uncs) ** k)
    
    @staticmethod
    def reward_k_fu(pred, uncertainty, k):
        if pred <= 0:
            return 0.0
        return (1.0 / pred) / (1 + k * uncertainty)

    def calc_TADF_score(self, state: State) -> float:
        mol = state.mol    
        smiles = Chem.MolToSmiles(mol)#'c1ccc(-c2nc(-c3ccccc3)nc(-c3ccc(-n4c5ccccc5c5c4c4ccccc4n5-c4ccccc4)cc3)n2)cc1'
        data = make_npz_file(smiles)
        try:
            _, unlabel_xs, _,_ = get_dataset_dataloader(
                data, batch_size=1, num_workers=1
            )
            unlab = score(self.trainer, unlabel_xs, self.device).cpu().detach().numpy().reshape(-1)
        except:
            unlab = 0
        return unlab
    
    def calc_score(self, state: State) -> float:
        mol = state.mol
        smiles = [Chem.MolToSmiles(mol)]

        with torch.no_grad():
            preds, logvars = {}, {}
            for name in ['model1', 'model2', 'model3']:
                pred, logvar = getattr(self, name)(smiles)
                preds[name] = pred.detach().cpu().numpy()
                logvars[name] = torch.exp(logvar).detach().cpu().numpy()

            for name in ['model1_1', 'model2_1', 'model3_1']:
                pred, logvar = getattr(self, name)(smiles)
                preds[name] = pred.detach().cpu().numpy()
                logvars[name] = torch.exp(logvar).detach().cpu().numpy()

        for i in range(1, 4):
            preds[f'model{i}'] = self.scaler.inverse_transform(preds[f'model{i}']).tolist()
            logvars[f'model{i}'] = self.scaler.inverse_transform_variance(logvars[f'model{i}'])

            preds[f'model{i}_1'] = self.scaler_1.inverse_transform(preds[f'model{i}_1']).tolist()
            logvars[f'model{i}_1'] = self.scaler_1.inverse_transform_variance(logvars[f'model{i}_1'])

        def get_avg_stats(prefix="", suffix=""):
            pred_arr = np.array([preds[f'{prefix}{i}{suffix}'] for i in range(1, 4)])  # shape: (3,1,1)
            logvar_arr = np.array([logvars[f'{prefix}{i}{suffix}'] for i in range(1, 4)])
            all_preds = np.transpose(pred_arr, (1, 2, 0))

            avg_pred = np.mean(pred_arr, axis=0)
            avg_ale = np.mean(logvar_arr, axis=0)
            avg_epi = np.var(all_preds, axis=2)
            return avg_pred[0][0], avg_ale[0][0], avg_epi[0][0]

        
        avg_preds, avg_ale_uncs, avg_epi_uncs = get_avg_stats('model')
        avg_preds_1, avg_ale_uncs_1, avg_epi_uncs_1 = get_avg_stats('model', "_1")

        state.score_dict = {
            "avg_preds": avg_preds,
            "avg_ale_uncs": avg_ale_uncs,
            "avg_epi_uncs": avg_epi_uncs,
            "avg_preds_1": avg_preds_1,
            "avg_ale_uncs_1": avg_ale_uncs_1,
            "avg_epi_uncs_1": avg_epi_uncs_1,
        }

        def compute_reward(pred, ale, epi, config, prefix=""):
            if getattr(config, f'lambda_ale{prefix}', 0) > 0 and getattr(config, f'lambda_epi{prefix}', 0) > 0:
                r_ale = self.reward_exp(pred, ale, getattr(config, f'lambda_ale{prefix}'))
                r_epi = self.reward_exp(pred, epi, getattr(config, f'lambda_epi{prefix}'))
            elif getattr(config, f'lambda_ale{prefix}', 0) > 0 and getattr(config, f'k_epi{prefix}', 0) > 0:
                r_ale = self.reward_exp(pred, ale, getattr(config, f'lambda_ale{prefix}'))
                r_epi = self.reward_k(pred, epi, getattr(config, f'k_epi{prefix}'))
            elif getattr(config, f'k_ale{prefix}', 0) > 0 and getattr(config, f'lambda_epi{prefix}', 0) > 0:
                r_ale = self.reward_k(pred, ale, getattr(config, f'k_ale{prefix}'))
                r_epi = self.reward_exp(pred, epi, getattr(config, f'lambda_epi{prefix}'))
            elif getattr(config, f'k_ale{prefix}', 0) > 0 and getattr(config, f'k_epi{prefix}', 0) > 0:
                r_ale = self.reward_k(pred, ale, getattr(config, f'k_ale{prefix}'))
                r_epi = self.reward_k(pred, epi, getattr(config, f'k_epi{prefix}'))
            else:
                raise ValueError(f"Invalid reward config for prefix: {prefix}")
            return r_ale, r_epi

        assert self.config.ALE > 0 and self.config.EPI > 0
        assert self.config.ALE_1 > 0 and self.config.EPI_1 > 0

        if self.config.lambda_ale > 0 and self.config.lambda_epi > 0:
            reward_ALE = self.reward_exp_fu(avg_preds, avg_ale_uncs, self.config.lambda_ale)
            reward_EPI = self.reward_exp_fu(avg_preds, avg_epi_uncs, self.config.lambda_epi)
        elif self.config.lambda_ale > 0 and self.config.k_epi > 0:
            reward_ALE = self.reward_exp_fu(avg_preds, avg_ale_uncs, self.config.lambda_ale)
            reward_EPI = self.reward_k_fu(avg_preds, avg_epi_uncs, self.config.k_epi)
        elif self.config.k_ale > 0 and self.config.lambda_epi > 0:
            reward_ALE = self.reward_k_fu(avg_preds, avg_ale_uncs, self.config.k_ale)
            reward_EPI = self.reward_exp_fu(avg_preds, avg_epi_uncs, self.config.lambda_epi)
        elif self.config.k_ale > 0 and self.config.k_epi > 0:
            reward_ALE = self.reward_k_fu(avg_preds, avg_ale_uncs, self.config.k_ale)
            reward_EPI = self.reward_k_fu(avg_preds, avg_epi_uncs, self.config.k_epi)
        else:
            raise ValueError("Invalid reward config for main model")

        reward_ALE_1, reward_EPI_1 = compute_reward(avg_preds_1, avg_ale_uncs_1, avg_epi_uncs_1, self.config, "_1")

        # 6. 更新分数并返回最终奖励
        state.score_dict.update({
            "eqexp_ale_preds": reward_ALE,
            "eqexp_epi_preds": reward_EPI,
            "eqexp_ale_preds_1": reward_ALE_1,
            "eqexp_epi_preds_1": reward_EPI_1,
        })

        atom_num = mol.GetNumAtoms()
        TADF_score = self.calc_TADF_score(state)
        if atom_num > 60 or TADF_score < 85:
            final_reward = 0.0
        elif (reward_ALE_1 + reward_EPI_1)/2 >4:
            final_reward = (reward_ALE + reward_EPI) / 2
        else:
            final_reward = (reward_ALE + reward_EPI) / 3
        state.score_dict.update({
            "atom_num": atom_num,
            "TADF_score": TADF_score,})

        return final_reward
    
class MPNNReward_small_est(BaseMolReward):
    ""
    def __init__(self, config):
        super().__init__(config)
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # cmd = set_cuda_visible_device(0)
        # os.environ["CUDA_VISIBLE_DEVICES"] = cmd
        self.config = config
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(self.device,'self.device')
        self.model1 = load_checkpoint(config.prop_model1, cuda=self.device)
        self.model2 = load_checkpoint(config.prop_model2, cuda=self.device)
        self.model3 = load_checkpoint(config.prop_model3, cuda=self.device)
        self.model1_1 = load_checkpoint(config.prop_model1_1, cuda=self.device)
        self.model2_1 = load_checkpoint(config.prop_model2_1, cuda=self.device)
        self.model3_1 = load_checkpoint(config.prop_model3_1, cuda=self.device)
        self.scaler, features_scaler = load_scalers(config.prop_model1)
        self.scaler_1, features_scaler_1 = load_scalers(config.prop_model1_1)
        train_args = load_args(config.prop_model1)
        
        self.autoencoder = AutoEncoder(200).to(self.device)
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
        self.trainer.ae_net.load_state_dict(torch.load(save_model, weights_only=True))
        self.trainer.ae_net.eval()

        self.model1.eval()
        self.model2.eval()
        self.model3.eval()
        self.model1_1.eval()
        self.model2_1.eval()
        self.model3_1.eval()
    
    @staticmethod
    def reward_exp(reward, avg_uncs, lambda_a):
        return reward * np.exp(-lambda_a * avg_uncs)

    @staticmethod
    def reward_exp_fu(pred, uncertainty, lambd):
        if pred <= 0:
            return 0.0
        return (1.0 / pred) * np.exp(-lambd * uncertainty)


    @staticmethod
    def reward_k(reward, avg_uncs, k):
        return reward / (1 + (avg_uncs) ** k)
    
    @staticmethod
    def reward_k_fu(pred, uncertainty, k):
        if pred <= 0:
            return 0.0
        return (1.0 / pred) / (1 + k * uncertainty)

    def calc_TADF_score(self, state: State) -> float:
        mol = state.mol    
        smiles = Chem.MolToSmiles(mol)#'c1ccc(-c2nc(-c3ccccc3)nc(-c3ccc(-n4c5ccccc5c5c4c4ccccc4n5-c4ccccc4)cc3)n2)cc1'
        data = make_npz_file(smiles)
        try:
            _, unlabel_xs, _,_ = get_dataset_dataloader(
                data, batch_size=1, num_workers=1
            )
            unlab = score(self.trainer, unlabel_xs, self.device).cpu().detach().numpy().reshape(-1)
        except:
            unlab = 0
        return unlab
    
    def calc_score(self, state: State) -> float:
        mol = state.mol
        smiles = [Chem.MolToSmiles(mol)]

        with torch.no_grad():
            preds, logvars = {}, {}
            for name in ['model1', 'model2', 'model3']:
                pred, logvar = getattr(self, name)(smiles)
                preds[name] = pred.detach().cpu().numpy()
                logvars[name] = torch.exp(logvar).detach().cpu().numpy()

            for name in ['model1_1', 'model2_1', 'model3_1']:
                pred, logvar = getattr(self, name)(smiles)
                preds[name] = pred.detach().cpu().numpy()
                logvars[name] = torch.exp(logvar).detach().cpu().numpy()

        for i in range(1, 4):
            preds[f'model{i}'] = self.scaler.inverse_transform(preds[f'model{i}']).tolist()
            logvars[f'model{i}'] = self.scaler.inverse_transform_variance(logvars[f'model{i}'])

            preds[f'model{i}_1'] = self.scaler_1.inverse_transform(preds[f'model{i}_1']).tolist()
            logvars[f'model{i}_1'] = self.scaler_1.inverse_transform_variance(logvars[f'model{i}_1'])

        def get_avg_stats(prefix="", suffix=""):
            pred_arr = np.array([preds[f'{prefix}{i}{suffix}'] for i in range(1, 4)])  # shape: (3,1,1)
            logvar_arr = np.array([logvars[f'{prefix}{i}{suffix}'] for i in range(1, 4)])
            all_preds = np.transpose(pred_arr, (1, 2, 0))

            avg_pred = np.mean(pred_arr, axis=0)
            avg_ale = np.mean(logvar_arr, axis=0)
            avg_epi = np.var(all_preds, axis=2)
            return avg_pred[0][0], avg_ale[0][0], avg_epi[0][0]

        
        avg_preds, avg_ale_uncs, avg_epi_uncs = get_avg_stats('model')
        avg_preds_1, avg_ale_uncs_1, avg_epi_uncs_1 = get_avg_stats('model', "_1")

        state.score_dict = {
            "avg_preds": avg_preds,
            "avg_ale_uncs": avg_ale_uncs,
            "avg_epi_uncs": avg_epi_uncs,
            "avg_preds_1": avg_preds_1,
            "avg_ale_uncs_1": avg_ale_uncs_1,
            "avg_epi_uncs_1": avg_epi_uncs_1,
        }

        def compute_reward(pred, ale, epi, config, prefix=""):
            if getattr(config, f'lambda_ale{prefix}', 0) > 0 and getattr(config, f'lambda_epi{prefix}', 0) > 0:
                r_ale = self.reward_exp(pred, ale, getattr(config, f'lambda_ale{prefix}'))
                r_epi = self.reward_exp(pred, epi, getattr(config, f'lambda_epi{prefix}'))
            elif getattr(config, f'lambda_ale{prefix}', 0) > 0 and getattr(config, f'k_epi{prefix}', 0) > 0:
                r_ale = self.reward_exp(pred, ale, getattr(config, f'lambda_ale{prefix}'))
                r_epi = self.reward_k(pred, epi, getattr(config, f'k_epi{prefix}'))
            elif getattr(config, f'k_ale{prefix}', 0) > 0 and getattr(config, f'lambda_epi{prefix}', 0) > 0:
                r_ale = self.reward_k(pred, ale, getattr(config, f'k_ale{prefix}'))
                r_epi = self.reward_exp(pred, epi, getattr(config, f'lambda_epi{prefix}'))
            elif getattr(config, f'k_ale{prefix}', 0) > 0 and getattr(config, f'k_epi{prefix}', 0) > 0:
                r_ale = self.reward_k(pred, ale, getattr(config, f'k_ale{prefix}'))
                r_epi = self.reward_k(pred, epi, getattr(config, f'k_epi{prefix}'))
            else:
                raise ValueError(f"Invalid reward config for prefix: {prefix}")
            return r_ale, r_epi

        assert self.config.ALE > 0 and self.config.EPI > 0
        assert self.config.ALE_1 > 0 and self.config.EPI_1 > 0

        if self.config.lambda_ale > 0 and self.config.lambda_epi > 0:
            reward_ALE = self.reward_exp_fu(avg_preds, avg_ale_uncs, self.config.lambda_ale)
            reward_EPI = self.reward_exp_fu(avg_preds, avg_epi_uncs, self.config.lambda_epi)
        elif self.config.lambda_ale > 0 and self.config.k_epi > 0:
            reward_ALE = self.reward_exp_fu(avg_preds, avg_ale_uncs, self.config.lambda_ale)
            reward_EPI = self.reward_k_fu(avg_preds, avg_epi_uncs, self.config.k_epi)
        elif self.config.k_ale > 0 and self.config.lambda_epi > 0:
            reward_ALE = self.reward_k_fu(avg_preds, avg_ale_uncs, self.config.k_ale)
            reward_EPI = self.reward_exp_fu(avg_preds, avg_epi_uncs, self.config.lambda_epi)
        elif self.config.k_ale > 0 and self.config.k_epi > 0:
            reward_ALE = self.reward_k_fu(avg_preds, avg_ale_uncs, self.config.k_ale)
            reward_EPI = self.reward_k_fu(avg_preds, avg_epi_uncs, self.config.k_epi)
        else:
            raise ValueError("Invalid reward config for main model")

        reward_ALE_1, reward_EPI_1 = compute_reward(avg_preds_1, avg_ale_uncs_1, avg_epi_uncs_1, self.config, "_1")

        # 6. 更新分数并返回最终奖励
        state.score_dict.update({
            "eqexp_ale_preds": reward_ALE,
            "eqexp_epi_preds": reward_EPI,
            "eqexp_ale_preds_1": reward_ALE_1,
            "eqexp_epi_preds_1": reward_EPI_1,
        })

        atom_num = mol.GetNumAtoms()
        TADF_score = self.calc_TADF_score(state)
        if atom_num > 100:
            final_reward = 0.0
        elif TADF_score < 0:
            final_reward = 0.0
        elif TADF_score < 30:
            final_reward = 0.3
        elif TADF_score < 60:
            final_reward = 0.6            
        elif TADF_score < 85:
            final_reward = 0.85

        elif (reward_ALE_1 + reward_EPI_1)/2 >4:
            final_reward = (reward_ALE + reward_EPI) / 2
        else:
            final_reward = (reward_ALE + reward_EPI) / 3
        state.score_dict.update({
            "atom_num": atom_num,
            "TADF_score": TADF_score,})

        return final_reward


class MPNNReward_T2(BaseMolReward):
    ""
    def __init__(self, config):
        super().__init__(config)
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # cmd = set_cuda_visible_device(0)
        # os.environ["CUDA_VISIBLE_DEVICES"] = cmd
        self.config = config
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(self.device,'self.device')
        self.model1 = load_checkpoint(config.prop_model1, cuda=self.device)
        self.model2 = load_checkpoint(config.prop_model2, cuda=self.device)
        self.model3 = load_checkpoint(config.prop_model3, cuda=self.device)
        self.model1_1 = load_checkpoint(config.prop_model1_1, cuda=self.device)
        self.model2_1 = load_checkpoint(config.prop_model2_1, cuda=self.device)
        self.model3_1 = load_checkpoint(config.prop_model3_1, cuda=self.device)
        self.model1_2 = load_checkpoint(config.prop_model1_2, cuda=self.device)
        self.model2_2 = load_checkpoint(config.prop_model2_2, cuda=self.device)
        self.model3_2 = load_checkpoint(config.prop_model3_2, cuda=self.device)
        self.model1_3 = load_checkpoint(config.prop_model1_3, cuda=self.device)
        self.model2_3 = load_checkpoint(config.prop_model2_3, cuda=self.device)
        self.model3_3 = load_checkpoint(config.prop_model3_3, cuda=self.device)

        self.scaler, features_scaler = load_scalers(config.prop_model1)
        self.scaler_1, features_scaler_1 = load_scalers(config.prop_model1_1)
        self.scaler_2, features_scaler_2 = load_scalers(config.prop_model1_2)
        self.scaler_3, features_scaler_3 = load_scalers(config.prop_model1_3)
        
        self.model1.eval()
        self.model2.eval()
        self.model3.eval()
        self.model1_1.eval()
        self.model2_1.eval()
        self.model3_1.eval()
        self.model1_2.eval()
        self.model2_2.eval()
        self.model3_2.eval()
        self.model1_3.eval()
        self.model2_3.eval()
        self.model3_3.eval()

    @staticmethod
    def reward_exp(reward, avg_uncs, lambda_a):
        return reward * np.exp(-lambda_a * avg_uncs)

    @staticmethod
    def reward_k(reward, avg_uncs, k):
        return reward / (1 + (avg_uncs) ** k)
     
    def calc_score(self, state: State) -> float:
        mol = state.mol
        smiles = [Chem.MolToSmiles(mol)]

        with torch.no_grad():
            preds, logvars = {}, {}
            for name in ['model1', 'model2', 'model3']:
                pred, logvar = getattr(self, name)(smiles)
                preds[name] = pred.detach().cpu().numpy()
                logvars[name] = torch.exp(logvar).detach().cpu().numpy()

            for name in ['model1_1', 'model2_1', 'model3_1']:
                pred, logvar = getattr(self, name)(smiles)
                preds[name] = pred.detach().cpu().numpy()
                logvars[name] = torch.exp(logvar).detach().cpu().numpy()

            for name in ['model1_2', 'model2_2', 'model3_2']:
                pred, logvar = getattr(self, name)(smiles)
                preds[name] = pred.detach().cpu().numpy()
                logvars[name] = torch.exp(logvar).detach().cpu().numpy()

            for name in ['model1_3', 'model2_3', 'model3_3']:
                pred, logvar = getattr(self, name)(smiles)
                preds[name] = pred.detach().cpu().numpy()
                logvars[name] = torch.exp(logvar).detach().cpu().numpy()

        for i in range(1, 4):
            preds[f'model{i}'] = self.scaler.inverse_transform(preds[f'model{i}']).tolist()
            logvars[f'model{i}'] = self.scaler.inverse_transform_variance(logvars[f'model{i}'])

            preds[f'model{i}_1'] = self.scaler_1.inverse_transform(preds[f'model{i}_1']).tolist()
            logvars[f'model{i}_1'] = self.scaler_1.inverse_transform_variance(logvars[f'model{i}_1'])

            preds[f'model{i}_2'] = self.scaler_2.inverse_transform(preds[f'model{i}_2']).tolist()
            logvars[f'model{i}_2'] = self.scaler_2.inverse_transform_variance(logvars[f'model{i}_2'])

            preds[f'model{i}_3'] = self.scaler_3.inverse_transform(preds[f'model{i}_3']).tolist()
            logvars[f'model{i}_3'] = self.scaler_3.inverse_transform_variance(logvars[f'model{i}_3'])

        def get_avg_stats(prefix="", suffix=""):
            pred_arr = np.array([preds[f'{prefix}{i}{suffix}'] for i in range(1, 4)])  
            logvar_arr = np.array([logvars[f'{prefix}{i}{suffix}'] for i in range(1, 4)])
            all_preds = np.transpose(pred_arr, (1, 2, 0))

            avg_pred = np.mean(pred_arr, axis=0)
            avg_ale = np.mean(logvar_arr, axis=0)
            avg_epi = np.var(all_preds, axis=2)
            return avg_pred[0][0], avg_ale[0][0], avg_epi[0][0]

        
        avg_preds, avg_ale_uncs, avg_epi_uncs = get_avg_stats('model')
        avg_preds_1, avg_ale_uncs_1, avg_epi_uncs_1 = get_avg_stats('model', "_1")
        avg_preds_2, avg_ale_uncs_2, avg_epi_uncs_2 = get_avg_stats('model', "_2")
        avg_preds_3, avg_ale_uncs_3, avg_epi_uncs_3 = get_avg_stats('model', "_3")

        state.score_dict = {
            "ES1": avg_preds,
            "avg_ale_uncs": avg_ale_uncs,
            "avg_epi_uncs": avg_epi_uncs,
            "ET1": avg_preds_1,
            "avg_ale_uncs_1": avg_ale_uncs_1,
            "avg_epi_uncs_1": avg_epi_uncs_1,
            "DS1": avg_preds_2,
            "avg_ale_uncs_2": avg_ale_uncs_2,
            "avg_epi_uncs_2": avg_epi_uncs_2,
            "DT1": avg_preds_3,
            "avg_ale_uncs_3": avg_ale_uncs_3,
            "avg_epi_uncs_3": avg_epi_uncs_3,}

        def compute_reward(pred, ale, epi, config, prefix=""):
            if getattr(config, f'lambda_ale{prefix}', 0) > 0 and getattr(config, f'lambda_epi{prefix}', 0) > 0:
                r_ale = self.reward_exp(pred, ale, getattr(config, f'lambda_ale{prefix}'))
                r_epi = self.reward_exp(pred, epi, getattr(config, f'lambda_epi{prefix}'))
            elif getattr(config, f'lambda_ale{prefix}', 0) > 0 and getattr(config, f'k_epi{prefix}', 0) > 0:
                r_ale = self.reward_exp(pred, ale, getattr(config, f'lambda_ale{prefix}'))
                r_epi = self.reward_k(pred, epi, getattr(config, f'k_epi{prefix}'))
            elif getattr(config, f'k_ale{prefix}', 0) > 0 and getattr(config, f'lambda_epi{prefix}', 0) > 0:
                r_ale = self.reward_k(pred, ale, getattr(config, f'k_ale{prefix}'))
                r_epi = self.reward_exp(pred, epi, getattr(config, f'lambda_epi{prefix}'))
            elif getattr(config, f'k_ale{prefix}', 0) > 0 and getattr(config, f'k_epi{prefix}', 0) > 0:
                r_ale = self.reward_k(pred, ale, getattr(config, f'k_ale{prefix}'))
                r_epi = self.reward_k(pred, epi, getattr(config, f'k_epi{prefix}'))
            else:
                raise ValueError(f"Invalid reward config for prefix: {prefix}")
            return r_ale, r_epi

        assert self.config.ALE > 0 and self.config.EPI > 0
        assert self.config.ALE_1 > 0 and self.config.EPI_1 > 0
        assert self.config.ALE_2 > 0 and self.config.EPI_2 > 0
        assert self.config.ALE_3 > 0 and self.config.EPI_3 > 0

        if self.config.lambda_ale > 0 and self.config.lambda_epi > 0:
            reward_ALE = self.reward_exp(avg_preds, avg_ale_uncs, self.config.lambda_ale)
            reward_EPI = self.reward_exp(avg_preds, avg_epi_uncs, self.config.lambda_epi)
        elif self.config.lambda_ale > 0 and self.config.k_epi > 0:
            reward_ALE = self.reward_exp(avg_preds, avg_ale_uncs, self.config.lambda_ale)
            reward_EPI = self.reward_k(avg_preds, avg_epi_uncs, self.config.k_epi)
        elif self.config.k_ale > 0 and self.config.lambda_epi > 0:
            reward_ALE = self.reward_k(avg_preds, avg_ale_uncs, self.config.k_ale)
            reward_EPI = self.reward_exp(avg_preds, avg_epi_uncs, self.config.lambda_epi)
        elif self.config.k_ale > 0 and self.config.k_epi > 0:
            reward_ALE = self.reward_k(avg_preds, avg_ale_uncs, self.config.k_ale)
            reward_EPI = self.reward_k(avg_preds, avg_epi_uncs, self.config.k_epi)
        else:
            raise ValueError("Invalid reward config for main model")

        reward_ALE_1, reward_EPI_1 = compute_reward(avg_preds_1, avg_ale_uncs_1, avg_epi_uncs_1, self.config, "_1")
        reward_ALE_2, reward_EPI_2 = compute_reward(avg_preds_2, avg_ale_uncs_2, avg_epi_uncs_2, self.config, "_2")
        reward_ALE_3, reward_EPI_3 = compute_reward(avg_preds_3, avg_ale_uncs_3, avg_epi_uncs_3, self.config, "_3")
        # 6. 更新分数并返回最终奖励
        state.score_dict.update({
            "eqexp_ale_preds": reward_ALE,
            "eqexp_epi_preds": reward_EPI,
            "eqexp_ale_preds_1": reward_ALE_1,
            "eqexp_epi_preds_1": reward_EPI_1,
            "eqexp_ale_preds_2": reward_ALE_2,
            "eqexp_epi_preds_2": reward_EPI_2,    
            "eqexp_ale_preds_3": reward_ALE_3,
            "eqexp_epi_preds_3": reward_EPI_3,     })

        atom_num = mol.GetNumAtoms()
        if atom_num > 100 or (reward_ALE_2 + reward_EPI_2)/2 <3 or (reward_ALE_3 + reward_EPI_3)/2 >3:
            final_reward = 0
        elif (reward_ALE + reward_EPI)/2 <3.8 and (reward_ALE_1 + reward_EPI_1)/2 >= 1.5 and (reward_ALE_2 + reward_EPI_2)/2 > 4:
            final_reward = ((reward_ALE + reward_EPI) / 2) - (reward_ALE_1 + reward_EPI_1) + 6       
        # elif (reward_ALE + reward_EPI)/2 <3.8 and (reward_ALE_1 + reward_EPI_1)/2 >= 1.5:
        #     final_reward = ((reward_ALE + reward_EPI) / 2) - (reward_ALE_1 + reward_EPI_1) +4
        # elif (reward_ALE + reward_EPI)/2 <3.8 or (reward_ALE_1 + reward_EPI_1)/2 >= 1.5:
        #     final_reward = ((reward_ALE + reward_EPI) / 2) - (reward_ALE_1 + reward_EPI_1) +2
        else:
            final_reward = 0
        state.score_dict.update({
            "atom_num": atom_num,})

        return final_reward


class MPNNReward_T2_noopt(BaseMolReward):
    ""
    def __init__(self, config):
        super().__init__(config)
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # cmd = set_cuda_visible_device(0)
        # os.environ["CUDA_VISIBLE_DEVICES"] = cmd
        self.config = config
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(self.device,'self.device')
        self.model1 = load_checkpoint(config.prop_model1, cuda=self.device)
        self.model2 = load_checkpoint(config.prop_model2, cuda=self.device)
        self.model3 = load_checkpoint(config.prop_model3, cuda=self.device)
        self.model1_1 = load_checkpoint(config.prop_model1_1, cuda=self.device)
        self.model2_1 = load_checkpoint(config.prop_model2_1, cuda=self.device)
        self.model3_1 = load_checkpoint(config.prop_model3_1, cuda=self.device)
        self.model1_2 = load_checkpoint(config.prop_model1_2, cuda=self.device)
        self.model2_2 = load_checkpoint(config.prop_model2_2, cuda=self.device)
        self.model3_2 = load_checkpoint(config.prop_model3_2, cuda=self.device)
        self.model1_3 = load_checkpoint(config.prop_model1_3, cuda=self.device)
        self.model2_3 = load_checkpoint(config.prop_model2_3, cuda=self.device)
        self.model3_3 = load_checkpoint(config.prop_model3_3, cuda=self.device)

        self.scaler, features_scaler = load_scalers(config.prop_model1)
        self.scaler_1, features_scaler_1 = load_scalers(config.prop_model1_1)
        self.scaler_2, features_scaler_2 = load_scalers(config.prop_model1_2)
        self.scaler_3, features_scaler_3 = load_scalers(config.prop_model1_3)
        
        self.model1.eval()
        self.model2.eval()
        self.model3.eval()
        self.model1_1.eval()
        self.model2_1.eval()
        self.model3_1.eval()
        self.model1_2.eval()
        self.model2_2.eval()
        self.model3_2.eval()
        self.model1_3.eval()
        self.model2_3.eval()
        self.model3_3.eval()

    @staticmethod
    def reward_exp(reward, avg_uncs, lambda_a):
        return reward * np.exp(-lambda_a * avg_uncs)

    @staticmethod
    def reward_k(reward, avg_uncs, k):
        return reward / (1 + (avg_uncs) ** k)
     
    def calc_score(self, state: State) -> float:
        mol = state.mol
        smiles = [Chem.MolToSmiles(mol)]

        with torch.no_grad():
            preds, logvars = {}, {}
            for name in ['model1', 'model2', 'model3']:
                pred, logvar = getattr(self, name)(smiles)
                preds[name] = pred.detach().cpu().numpy()
                logvars[name] = torch.exp(logvar).detach().cpu().numpy()

            for name in ['model1_1', 'model2_1', 'model3_1']:
                pred, logvar = getattr(self, name)(smiles)
                preds[name] = pred.detach().cpu().numpy()
                logvars[name] = torch.exp(logvar).detach().cpu().numpy()

            for name in ['model1_2', 'model2_2', 'model3_2']:
                pred, logvar = getattr(self, name)(smiles)
                preds[name] = pred.detach().cpu().numpy()
                logvars[name] = torch.exp(logvar).detach().cpu().numpy()

            for name in ['model1_3', 'model2_3', 'model3_3']:
                pred, logvar = getattr(self, name)(smiles)
                preds[name] = pred.detach().cpu().numpy()
                logvars[name] = torch.exp(logvar).detach().cpu().numpy()

        for i in range(1, 4):
            preds[f'model{i}'] = self.scaler.inverse_transform(preds[f'model{i}']).tolist()
            logvars[f'model{i}'] = self.scaler.inverse_transform_variance(logvars[f'model{i}'])

            preds[f'model{i}_1'] = self.scaler_1.inverse_transform(preds[f'model{i}_1']).tolist()
            logvars[f'model{i}_1'] = self.scaler_1.inverse_transform_variance(logvars[f'model{i}_1'])

            preds[f'model{i}_2'] = self.scaler_2.inverse_transform(preds[f'model{i}_2']).tolist()
            logvars[f'model{i}_2'] = self.scaler_2.inverse_transform_variance(logvars[f'model{i}_2'])

            preds[f'model{i}_3'] = self.scaler_3.inverse_transform(preds[f'model{i}_3']).tolist()
            logvars[f'model{i}_3'] = self.scaler_3.inverse_transform_variance(logvars[f'model{i}_3'])

        def get_avg_stats(prefix="", suffix=""):
            pred_arr = np.array([preds[f'{prefix}{i}{suffix}'] for i in range(1, 4)])  
            logvar_arr = np.array([logvars[f'{prefix}{i}{suffix}'] for i in range(1, 4)])
            all_preds = np.transpose(pred_arr, (1, 2, 0))

            avg_pred = np.mean(pred_arr, axis=0)
            avg_ale = np.mean(logvar_arr, axis=0)
            avg_epi = np.var(all_preds, axis=2)
            return avg_pred[0][0], avg_ale[0][0], avg_epi[0][0]

        
        avg_preds, avg_ale_uncs, avg_epi_uncs = get_avg_stats('model')
        avg_preds_1, avg_ale_uncs_1, avg_epi_uncs_1 = get_avg_stats('model', "_1")
        avg_preds_2, avg_ale_uncs_2, avg_epi_uncs_2 = get_avg_stats('model', "_2")
        avg_preds_3, avg_ale_uncs_3, avg_epi_uncs_3 = get_avg_stats('model', "_3")

        state.score_dict = {
            "ES1": avg_preds,
            "avg_ale_uncs": avg_ale_uncs,
            "avg_epi_uncs": avg_epi_uncs,
            "ET1": avg_preds_1,
            "avg_ale_uncs_1": avg_ale_uncs_1,
            "avg_epi_uncs_1": avg_epi_uncs_1,
            "DS1": avg_preds_2,
            "avg_ale_uncs_2": avg_ale_uncs_2,
            "avg_epi_uncs_2": avg_epi_uncs_2,
            "DT1": avg_preds_3,
            "avg_ale_uncs_3": avg_ale_uncs_3,
            "avg_epi_uncs_3": avg_epi_uncs_3,}

        def compute_reward(pred, ale, epi, config, prefix=""):
            if getattr(config, f'lambda_ale{prefix}', 0) > 0 and getattr(config, f'lambda_epi{prefix}', 0) > 0:
                r_ale = self.reward_exp(pred, ale, getattr(config, f'lambda_ale{prefix}'))
                r_epi = self.reward_exp(pred, epi, getattr(config, f'lambda_epi{prefix}'))
            elif getattr(config, f'lambda_ale{prefix}', 0) > 0 and getattr(config, f'k_epi{prefix}', 0) > 0:
                r_ale = self.reward_exp(pred, ale, getattr(config, f'lambda_ale{prefix}'))
                r_epi = self.reward_k(pred, epi, getattr(config, f'k_epi{prefix}'))
            elif getattr(config, f'k_ale{prefix}', 0) > 0 and getattr(config, f'lambda_epi{prefix}', 0) > 0:
                r_ale = self.reward_k(pred, ale, getattr(config, f'k_ale{prefix}'))
                r_epi = self.reward_exp(pred, epi, getattr(config, f'lambda_epi{prefix}'))
            elif getattr(config, f'k_ale{prefix}', 0) > 0 and getattr(config, f'k_epi{prefix}', 0) > 0:
                r_ale = self.reward_k(pred, ale, getattr(config, f'k_ale{prefix}'))
                r_epi = self.reward_k(pred, epi, getattr(config, f'k_epi{prefix}'))
            else:
                raise ValueError(f"Invalid reward config for prefix: {prefix}")
            return r_ale, r_epi

        assert self.config.ALE > 0 and self.config.EPI > 0
        assert self.config.ALE_1 > 0 and self.config.EPI_1 > 0
        assert self.config.ALE_2 > 0 and self.config.EPI_2 > 0
        assert self.config.ALE_3 > 0 and self.config.EPI_3 > 0

        if self.config.lambda_ale > 0 and self.config.lambda_epi > 0:
            reward_ALE = self.reward_exp(avg_preds, avg_ale_uncs, self.config.lambda_ale)
            reward_EPI = self.reward_exp(avg_preds, avg_epi_uncs, self.config.lambda_epi)
        elif self.config.lambda_ale > 0 and self.config.k_epi > 0:
            reward_ALE = self.reward_exp(avg_preds, avg_ale_uncs, self.config.lambda_ale)
            reward_EPI = self.reward_k(avg_preds, avg_epi_uncs, self.config.k_epi)
        elif self.config.k_ale > 0 and self.config.lambda_epi > 0:
            reward_ALE = self.reward_k(avg_preds, avg_ale_uncs, self.config.k_ale)
            reward_EPI = self.reward_exp(avg_preds, avg_epi_uncs, self.config.lambda_epi)
        elif self.config.k_ale > 0 and self.config.k_epi > 0:
            reward_ALE = self.reward_k(avg_preds, avg_ale_uncs, self.config.k_ale)
            reward_EPI = self.reward_k(avg_preds, avg_epi_uncs, self.config.k_epi)
        else:
            raise ValueError("Invalid reward config for main model")

        reward_ALE_1, reward_EPI_1 = compute_reward(avg_preds_1, avg_ale_uncs_1, avg_epi_uncs_1, self.config, "_1")
        reward_ALE_2, reward_EPI_2 = compute_reward(avg_preds_2, avg_ale_uncs_2, avg_epi_uncs_2, self.config, "_2")
        reward_ALE_3, reward_EPI_3 = compute_reward(avg_preds_3, avg_ale_uncs_3, avg_epi_uncs_3, self.config, "_3")
        # 6. 更新分数并返回最终奖励
        state.score_dict.update({
            "eqexp_ale_preds": reward_ALE,
            "eqexp_epi_preds": reward_EPI,
            "eqexp_ale_preds_1": reward_ALE_1,
            "eqexp_epi_preds_1": reward_EPI_1,
            "eqexp_ale_preds_2": reward_ALE_2,
            "eqexp_epi_preds_2": reward_EPI_2,    
            "eqexp_ale_preds_3": reward_ALE_3,
            "eqexp_epi_preds_3": reward_EPI_3,     })

        atom_num = mol.GetNumAtoms()
        if atom_num > 100 or (reward_ALE_2 + reward_EPI_2)/2 <3 or (reward_ALE_3 + reward_EPI_3)/2 >3:
            final_reward = 0
        elif (reward_ALE + reward_EPI)/2 <3.8 and (reward_ALE_1 + reward_EPI_1)/2 >= 1.5 and (reward_ALE_2 + reward_EPI_2)/2 > 4:
            final_reward = ((reward_ALE + reward_EPI) / 2) - (reward_ALE_1 + reward_EPI_1) + 6       
        # elif (reward_ALE + reward_EPI)/2 <3.8 and (reward_ALE_1 + reward_EPI_1)/2 >= 1.5:
        #     final_reward = ((reward_ALE + reward_EPI) / 2) - (reward_ALE_1 + reward_EPI_1) +4
        # elif (reward_ALE + reward_EPI)/2 <3.8 or (reward_ALE_1 + reward_EPI_1)/2 >= 1.5:
        #     final_reward = ((reward_ALE + reward_EPI) / 2) - (reward_ALE_1 + reward_EPI_1) +2
        else:
            final_reward = 0
        state.score_dict.update({
            "atom_num": atom_num,})

        return final_reward 

    # def calc_score(self, state: State) -> float:
    #     mol = state.mol    
    #     smiles = [Chem.MolToSmiles(mol)] #'c1ccc(-c2nc(-c3ccccc3)nc(-c3ccc(-n4c5ccccc5c5c4c4ccccc4n5-c4ccccc4)cc3)n2)cc1'
    #     # if 'FORMAT' in self.save_model:
    #     #     data = mol_to_graph_data_obj_simple(mol)
    #     # elif 'Cho' in self.save_model:
    #     #     data = mol_to_graph_data_obj_simple_2(mol=mol,sol=Chem.MolFromSmiles('ClCCl'))
    #     # else:
    #     #     raise ValueError(f"penalized_logp_reward.py Unsupported save_model type: {self.save_model}")
    #     # data = data.to(self.device)
    #     with torch.no_grad():
    #         pred1,logvar1 = self.model1(smiles)
    #         pred2,logvar2 = self.model2(smiles)
    #         pred3,logvar3 = self.model3(smiles)
    #         pred1_1,logvar1_1 = self.model1_1(smiles)
    #         pred2_1,logvar2_1 = self.model2_1(smiles)
    #         pred3_1,logvar3_1 = self.model3_1(smiles)

    #         pred1,pred2,pred3= pred1.detach().cpu().numpy(),pred2.detach().cpu().numpy(),pred3.detach().cpu().numpy()
    #         pred1_1,pred2_1,pred3_1= pred1_1.detach().cpu().numpy(),pred2_1.detach().cpu().numpy(),pred3_1.detach().cpu().numpy()

    #         logvar1 = torch.exp(logvar1)
    #         logvar2 = torch.exp(logvar2)
    #         logvar3 = torch.exp(logvar3)
    #         logvar1_1 = torch.exp(logvar1_1)
    #         logvar2_1 = torch.exp(logvar2_1)
    #         logvar3_1 = torch.exp(logvar3_1)

    #         logvar1,logvar2,logvar3= logvar1.detach().cpu().numpy(),logvar2.detach().cpu().numpy(),logvar3.detach().cpu().numpy()
    #         logvar1_1,logvar2_1,logvar3_1= logvar1_1.detach().cpu().numpy(),logvar2_1.detach().cpu().numpy(),logvar3_1.detach().cpu().numpy()

    #         pred1 = self.scaler.inverse_transform(pred1).tolist()
    #         pred2 = self.scaler.inverse_transform(pred2).tolist()
    #         pred3 = self.scaler.inverse_transform(pred3).tolist()
    #         logvar1 = self.scaler.inverse_transform_variance(logvar1)
    #         logvar2 = self.scaler.inverse_transform_variance(logvar2)
    #         logvar3 = self.scaler.inverse_transform_variance(logvar3)
    #         pred1_1 = self.scaler_1.inverse_transform(pred1_1).tolist()
    #         pred2_1 = self.scaler_1.inverse_transform(pred2_1).tolist()
    #         pred3_1 = self.scaler_1.inverse_transform(pred3_1).tolist()
    #         logvar1_1 = self.scaler_1.inverse_transform_variance(logvar1_1)
    #         logvar2_1 = self.scaler_1.inverse_transform_variance(logvar2_1)
    #         logvar3_1 = self.scaler_1.inverse_transform_variance(logvar3_1)

    #         all_preds = np.zeros((1,1,3))
    #         all_preds[:, :, 0] = pred1
    #         all_preds[:, :, 1] = pred2
    #         all_preds[:, :, 2] = pred3
    #         all_preds_1 = np.zeros((1,1,3))
    #         all_preds_1[:, :, 0] = pred1_1
    #         all_preds_1[:, :, 1] = pred2_1
    #         all_preds_1[:, :, 2] = pred3_1

    #         avg_preds = (np.array(pred1) + np.array(pred2) + np.array(pred3)) / 3
    #         avg_ale_uncs = (np.array(logvar1) + np.array(logvar2) + np.array(logvar3)) / 3
    #         avg_epi_uncs = np.var(all_preds, axis=2)
    #         avg_preds_1 = (np.array(pred1_1) + np.array(pred2_1) + np.array(pred3_1)) / 3
    #         avg_ale_uncs_1 = (np.array(logvar1_1) + np.array(logvar2_1) + np.array(logvar3_1)) / 3
    #         avg_epi_uncs_1 = np.var(all_preds_1, axis=2)
    #         state.score_dict = {"avg_preds":avg_preds[0][0],"avg_ale_uncs":avg_ale_uncs[0][0],"avg_epi_uncs":avg_epi_uncs[0][0],"avg_preds_1":avg_preds_1[0][0],"avg_ale_uncs_1":avg_ale_uncs_1[0][0],"avg_epi_uncs_1":avg_epi_uncs_1[0][0],}


    #     assert self.config.EPI > 0 and self.config.ALE > 0      
    #     assert self.config.EPI_1 > 0 and self.config.ALE_1 > 0  
    #     if self.config.lambda_ale>0 and self.config.lambda_epi>0:
    #         reward_ALE=self.reward_exp_fu(avg_preds[0][0],avg_ale_uncs[0][0],self.config.lambda_ale)
    #         reward_EPI=self.reward_exp_fu(avg_preds[0][0],avg_epi_uncs[0][0],self.config.lambda_epi)
    #         if self.config.lambda_ale_1>0 and self.config.lambda_epi_1>0:
    #             reward_ALE_1=self.reward_exp(avg_preds_1[0][0],avg_ale_uncs_1[0][0],self.config.lambda_ale_1)
    #             reward_EPI_1=self.reward_exp(avg_preds_1[0][0],avg_epi_uncs_1[0][0],self.config.lambda_epi_1)
    #         if self.config.lambda_ale_1>0 and self.config.k_epi_1>0:
    #             reward_ALE_1=self.reward_exp(avg_preds_1[0][0],avg_ale_uncs_1[0][0],self.config.lambda_ale_1)
    #             reward_EPI_1=self.reward_k(avg_preds_1[0][0],avg_epi_uncs_1[0][0],self.config.k_epi_1)
    #         if self.config.k_ale_1>0 and self.config.lambda_epi_1>0:
    #             reward_ALE_1=self.reward_k(avg_preds_1[0][0],avg_ale_uncs_1[0][0],self.config.k_ale_1)
    #             reward_EPI_1=self.reward_exp(avg_preds_1[0][0],avg_epi_uncs_1[0][0],self.config.lambda_epi_1)
    #         if self.config.k_ale_1>0 and self.config.k_epi_1>0:
    #             reward_ALE_1=self.reward_k(avg_preds_1[0][0],avg_ale_uncs_1[0][0],self.config.k_ale_1)
    #             reward_EPI_1=self.reward_k(avg_preds_1[0][0],avg_epi_uncs_1[0][0],self.config.k_epi_1)
    #         state.score_dict.update({"eqexp_ale_preds":reward_ALE,"eqexp_epi_preds":reward_EPI,"eqexp_ale_preds_1":reward_ALE_1,"eqexp_epi_preds_1":reward_EPI_1,})
    #         return (reward_ALE+reward_EPI+reward_ALE_1+reward_EPI_1)/4#

    #     elif self.config.lambda_ale>0 and self.config.k_epi>0:
    #         reward_ALE=self.reward_exp_fu(avg_preds[0][0],avg_ale_uncs[0][0],self.config.lambda_ale)
    #         reward_EPI=self.reward_k_fu(avg_preds[0][0],avg_epi_uncs[0][0],self.config.k_epi)
    #         if self.config.lambda_ale_1>0 and self.config.lambda_epi_1>0:
    #             reward_ALE_1=self.reward_exp(avg_preds_1[0][0],avg_ale_uncs_1[0][0],self.config.lambda_ale_1)
    #             reward_EPI_1=self.reward_exp(avg_preds_1[0][0],avg_epi_uncs_1[0][0],self.config.lambda_epi_1)
    #         if self.config.lambda_ale_1>0 and self.config.k_epi_1>0:
    #             reward_ALE_1=self.reward_exp(avg_preds_1[0][0],avg_ale_uncs_1[0][0],self.config.lambda_ale_1)
    #             reward_EPI_1=self.reward_k(avg_preds_1[0][0],avg_epi_uncs_1[0][0],self.config.k_epi_1)
    #         if self.config.k_ale_1>0 and self.config.lambda_epi_1>0:
    #             reward_ALE_1=self.reward_k(avg_preds_1[0][0],avg_ale_uncs_1[0][0],self.config.k_ale_1)
    #             reward_EPI_1=self.reward_exp(avg_preds_1[0][0],avg_epi_uncs_1[0][0],self.config.lambda_epi_1)
    #         if self.config.k_ale_1>0 and self.config.k_epi_1>0:
    #             reward_ALE_1=self.reward_k(avg_preds_1[0][0],avg_ale_uncs_1[0][0],self.config.k_ale_1)
    #             reward_EPI_1=self.reward_k(avg_preds_1[0][0],avg_epi_uncs_1[0][0],self.config.k_epi_1)
    #         state.score_dict.update({"eqexp_ale_preds":reward_ALE,"eqexp_epi_preds":reward_EPI,"eqexp_ale_preds_1":reward_ALE_1,"eqexp_epi_preds_1":reward_EPI_1,})
    #         return (reward_ALE+reward_EPI+reward_ALE_1+reward_EPI_1)/4#

        
    #     elif self.config.k_ale>0 and self.config.lambda_epi>0:
    #         reward_ALE=self.reward_k_fu(avg_preds[0][0],avg_ale_uncs[0][0],self.config.k_ale)
    #         reward_EPI=self.reward_exp_fu(avg_preds[0][0],avg_epi_uncs[0][0],self.config.lambda_epi)
    #         if self.config.lambda_ale_1>0 and self.config.lambda_epi_1>0:
    #             reward_ALE_1=self.reward_exp(avg_preds_1[0][0],avg_ale_uncs_1[0][0],self.config.lambda_ale_1)
    #             reward_EPI_1=self.reward_exp(avg_preds_1[0][0],avg_epi_uncs_1[0][0],self.config.lambda_epi_1)
    #         if self.config.lambda_ale_1>0 and self.config.k_epi_1>0:
    #             reward_ALE_1=self.reward_exp(avg_preds_1[0][0],avg_ale_uncs_1[0][0],self.config.lambda_ale_1)
    #             reward_EPI_1=self.reward_k(avg_preds_1[0][0],avg_epi_uncs_1[0][0],self.config.k_epi_1)
    #         if self.config.k_ale_1>0 and self.config.lambda_epi_1>0:
    #             reward_ALE_1=self.reward_k(avg_preds_1[0][0],avg_ale_uncs_1[0][0],self.config.k_ale_1)
    #             reward_EPI_1=self.reward_exp(avg_preds_1[0][0],avg_epi_uncs_1[0][0],self.config.lambda_epi_1)
    #         if self.config.k_ale_1>0 and self.config.k_epi_1>0:
    #             reward_ALE_1=self.reward_k(avg_preds_1[0][0],avg_ale_uncs_1[0][0],self.config.k_ale_1)
    #             reward_EPI_1=self.reward_k(avg_preds_1[0][0],avg_epi_uncs_1[0][0],self.config.k_epi_1)
    #         state.score_dict.update({"eqexp_ale_preds":reward_ALE,"eqexp_epi_preds":reward_EPI,"eqexp_ale_preds_1":reward_ALE_1,"eqexp_epi_preds_1":reward_EPI_1,})
    #         return (reward_ALE+reward_EPI+reward_ALE_1+reward_EPI_1)/4#

    #     elif self.config.k_ale>0 and self.config.k_epi>0:
    #         reward_ALE=self.reward_k_fu(avg_preds[0][0],avg_ale_uncs[0][0],self.config.k_ale)
    #         reward_EPI=self.reward_k_fu(avg_preds[0][0],avg_epi_uncs[0][0],self.config.k_epi)
    #         if self.config.lambda_ale_1>0 and self.config.lambda_epi_1>0:
    #             reward_ALE_1=self.reward_exp(avg_preds_1[0][0],avg_ale_uncs_1[0][0],self.config.lambda_ale_1)
    #             reward_EPI_1=self.reward_exp(avg_preds_1[0][0],avg_epi_uncs_1[0][0],self.config.lambda_epi_1)
    #         if self.config.lambda_ale_1>0 and self.config.k_epi_1>0:
    #             reward_ALE_1=self.reward_exp(avg_preds_1[0][0],avg_ale_uncs_1[0][0],self.config.lambda_ale_1)
    #             reward_EPI_1=self.reward_k(avg_preds_1[0][0],avg_epi_uncs_1[0][0],self.config.k_epi_1)
    #         if self.config.k_ale_1>0 and self.config.lambda_epi_1>0:
    #             reward_ALE_1=self.reward_k(avg_preds_1[0][0],avg_ale_uncs_1[0][0],self.config.k_ale_1)
    #             reward_EPI_1=self.reward_exp(avg_preds_1[0][0],avg_epi_uncs_1[0][0],self.config.lambda_epi_1)
    #         if self.config.k_ale_1>0 and self.config.k_epi_1>0:
    #             reward_ALE_1=self.reward_k(avg_preds_1[0][0],avg_ale_uncs_1[0][0],self.config.k_ale_1)
    #             reward_EPI_1=self.reward_k(avg_preds_1[0][0],avg_epi_uncs_1[0][0],self.config.k_epi_1)
    #         state.score_dict.update({"eqexp_ale_preds":reward_ALE,"eqexp_epi_preds":reward_EPI,"eqexp_ale_preds_1":reward_ALE_1,"eqexp_epi_preds_1":reward_EPI_1,})
    #         return (reward_ALE+reward_EPI+reward_ALE_1+reward_EPI_1)/4#

    #     else :
    #         raise ValueError("Unexpected condition in config.ALE and config.EPI")

                        