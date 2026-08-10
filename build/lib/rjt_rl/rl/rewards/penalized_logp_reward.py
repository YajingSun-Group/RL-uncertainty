from __future__ import annotations

import networkx as nx
from rdkit import Chem
# from rdkit.Chem.Descriptors import MolLogP
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from rjt_rl.rl.envs.states import State

from .base_mol_reward import BaseMolReward
from .TADFVAE.to_rjt import *
from .pre_model.to_rjt_pre import *
from sklearn.metrics import roc_auc_score
from rdkit.Contrib.SA_Score import sascorer
from rdkit.Chem import rdMolDescriptors

class similarityReward(BaseMolReward):
    def __init__(self, config):
        super().__init__(config)
        self.mol1 = Chem.MolFromSmiles('O=C(C1=CC=C(N2C(C=CC=C3)=C3OC4=C2C=CC=C4)C=C1)C5=CC(C(C=CC=C6)=C6N7C8=CC=CC=C8)=C7C=C5')
        # self.mol1 = Chem.MolFromSmiles('N#CC1=C(N2C(C=CC=C3)=C3C4=C2C=CC=C4)C(N5C(C=CC(C6=CC=CC=C6)=C7)=C7C8=C5C=CC(C9=CC=CC=C9)=C8)=C(N%10C(C=CC=C%11)=C%11C%12=C%10C=CC=C%12)C(N%13C(C=CC(C%14=CC=CC=C%14)=C%15)=C%15C%16=C%13C=CC(C%17=CC=CC=C%17)=C%16)=C1N%18C(C=CC=C%19)=C%19C%20=C%18C=CC=C%20')
        self.morgan_gen = GetMorganGenerator(radius=2, fpSize=128) # 小分子128 大分子518
        self.mol1_fp = self.morgan_gen.GetFingerprint(self.mol1)

    def calc_score(self, state: State) -> float:
        mol = state.mol
        fp2 = self.morgan_gen.GetFingerprint(mol)
        similarity = TanimotoSimilarity(self.mol1_fp, fp2)
        if mol.HasSubstructMatch(self.mol1):
            similarity += 1
        state.score_dict = {
            "similarity_score": similarity,
        }
        return similarity

class TADFscoreReward(BaseMolReward):
    def __init__(self, config):
        super().__init__(config)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        cmd = set_cuda_visible_device(1)
        os.environ["CUDA_VISIBLE_DEVICES"] = cmd
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
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

    def calc_score(self, state: State) -> float:
        mol = state.mol    
        smiles = Chem.MolToSmiles(mol)#'c1ccc(-c2nc(-c3ccccc3)nc(-c3ccc(-n4c5ccccc5c5c4c4ccccc4n5-c4ccccc4)cc3)n2)cc1'
        data = make_npz_file(smiles)
        try:
            _, unlabel_xs, _,_ = get_dataset_dataloader(
                data, batch_size=1, num_workers=1
            )
            unlab = score(self.trainer, unlabel_xs, self.device).cpu().detach().numpy().reshape(-1)
            if unlab>0:
                TADF_score = unlab[0]/10
            else:
                TADF_score = 0
        except:
            TADF_score = 0
        state.score_dict = {
            "TADF_score": TADF_score,}
        return TADF_score

class PropReward(BaseMolReward):
    def __init__(self, config):
        super().__init__(config)
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # cmd = set_cuda_visible_device(0)
        # os.environ["CUDA_VISIBLE_DEVICES"] = cmd
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(self.device,'self.device')
        self.model = GNN_graphpred(num_layer=5, emb_dim=300, num_tasks=1, JK = 'last', drop_ratio = 0.5, graph_pooling = 'mean', gnn_type = 'gin')
        self.save_model = config.prop_model
        # self.model.from_pretrained(save_model)
        self.model.load_state_dict(torch.load(self.save_model))
        self.model.to(self.device)
        self.model.eval()

    def calc_score(self, state: State) -> float:
        mol = state.mol    
        # smiles = Chem.MolToSmiles(mol)#'c1ccc(-c2nc(-c3ccccc3)nc(-c3ccc(-n4c5ccccc5c5c4c4ccccc4n5-c4ccccc4)cc3)n2)cc1'
        if 'FORMAT' in self.save_model:
            data = mol_to_graph_data_obj_simple(mol)
        elif 'Cho' in self.save_model:
            data = mol_to_graph_data_obj_simple_2(mol=mol,sol=Chem.MolFromSmiles('ClCCl'))
        else:
            raise ValueError(f"penalized_logp_reward.py Unsupported save_model type: {self.save_model}")
        data = data.to(self.device)
        with torch.no_grad():
            pred = self.model(data.x, data.edge_index, data.edge_attr, data.batch)
            pred = torch.where(pred < 0, torch.tensor(-1, device=pred.device), pred)
            pred = torch.where(pred >= 0, torch.tensor(1, device=pred.device), pred)
            pre2 = pred.cpu().detach().numpy().reshape(-1)
            # print(pre2,'pre2')
            state.score_dict = {"total_score":pre2,}
        return pre2[0]

class PropReward2(BaseMolReward):
    def __init__(self, config):
        super().__init__(config)
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # cmd = set_cuda_visible_device(0)
        # os.environ["CUDA_VISIBLE_DEVICES"] = cmd
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(self.device,'self.device')
        self.model1 = GNN_graphpred(num_layer=5, emb_dim=300, num_tasks=1, JK = 'last', drop_ratio = 0.5, graph_pooling = 'mean', gnn_type = 'gin')
        self.model2 = GNN_graphpred(num_layer=5, emb_dim=300, num_tasks=1, JK = 'last', drop_ratio = 0.5, graph_pooling = 'mean', gnn_type = 'gin')
        self.save_model1 = config.prop_model1
        self.save_model2 = config.prop_model2
        # self.model.from_pretrained(save_model)
        self.model1.load_state_dict(torch.load(self.save_model1))
        self.model2.load_state_dict(torch.load(self.save_model2))
        self.model1.to(self.device)
        self.model2.to(self.device)
        self.model1.eval()
        self.model2.eval()

    def calc_score(self, state: State) -> float:
        mol = state.mol    
        # smiles = Chem.MolToSmiles(mol)#'c1ccc(-c2nc(-c3ccccc3)nc(-c3ccc(-n4c5ccccc5c5c4c4ccccc4n5-c4ccccc4)cc3)n2)cc1'
        # if 'FORMAT' in self.save_model:
        data = mol_to_graph_data_obj_simple(mol)
        # elif 'Cho' in self.save_model:
        #     data = mol_to_graph_data_obj_simple_2(mol=mol,sol=Chem.MolFromSmiles('ClCCl'))
        # else:
        #     raise ValueError(f"penalized_logp_reward.py Unsupported save_model type: {self.save_model}")
        data = data.to(self.device)
        with torch.no_grad():
            pred1 = self.model1(data.x, data.edge_index, data.edge_attr, data.batch)
            pred2 = self.model2(data.x, data.edge_index, data.edge_attr, data.batch)          
            pred1 = torch.where(pred1 < 0, torch.tensor(-1, device=pred1.device), pred1)
            pred1 = torch.where(pred1 >= 0, torch.tensor(1, device=pred1.device), pred1)
            pred2 = torch.where(pred2 < 0, torch.tensor(-1, device=pred2.device), pred2)
            pred2 = torch.where(pred2 >= 0, torch.tensor(1, device=pred2.device), pred2)
            print(pred1,' pred1', pred2,' pred2')
            pre_all = (pred1+pred2).cpu().detach().numpy().reshape(-1)
            # print(pre2,'pre2')
            state.score_dict = {"total_score": pre_all[0],}
        return  pre_all[0]

class TPSAReward(BaseMolReward):
    def calc_score(self, state: State) -> float:
        mol = state.mol
        # log_p = MolLogP(mol)
        TPSA = rdMolDescriptors.CalcTPSA(mol)
        state.score_dict = {"TPSA": TPSA}
        # if TPSA>0:
        #     return TPSA
        # else:
        return TPSA#700

class SAReward(BaseMolReward):
    def calc_score(self, state: State) -> float:
        mol = state.mol

        SA = sascorer.calculateScore(mol)
        state.score_dict = {"SA": SA}
        return SA   

