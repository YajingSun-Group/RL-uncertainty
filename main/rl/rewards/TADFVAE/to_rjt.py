import os
import pickle
import random
from multiprocessing import Pool
import glob
import sys
import numpy as np
import pandas as pd
import torch
from descriptastorus.descriptors import rdNormalizedDescriptors
from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdMolDescriptors import *
import argparse
from .ae_trainer import AETrainer
from .dataloader import *
from .model import *
from .utils import score, set_cuda_visible_device

def make_results(smiles):
    properties = []
    if "." in smiles:
        return ""
    mol = Chem.MolFromSmiles(smiles)
    try:
        generator = rdNormalizedDescriptors.RDKit2DNormalized()
        properties = generator.process(smiles)[1:]
    except:
        return ""
    a = np.array(properties)
    a[np.isnan(a)] = 0
    properties = list(a)
    return properties

def make_data(smiles, target):
    test_y = []
    test_x = make_results(smiles)
    # print(len(test_x),'smiles???')
    smiles_list = []
    for i in range(len(test_x)):
        test_y.append(target)
    smiles_list.append(smiles)
    # print(len(smiles_list))

    return test_x, test_y, smiles_list

def make_npz_file(smiles):
    data = {"id": [], "prop": [], "feats": [], "smiles": []}
    x_list, _, smiles_list = make_data(smiles, 0)
    # print(smiles_list,'smiles_list')
    if not isinstance(x_list, str):
        y = float(0.0)
        data["prop"].append(y)
        data["feats"].append(x_list)
        data["smiles"].append(smiles_list)
    # print(data,'data')
    return data


def make_npz_file_list(smiles_list):
    data = {"id": [], "prop": [], "feats": [], "smiles": []}
    
    # 确保是列表
    if isinstance(smiles_list, str):
        smiles_list = [smiles_list]

    try:
        x_list, _, smiles_out = make_data(smiles_list, 0)
    except Exception as e:
        print("make_data failed:", e)
        return data
    if isinstance(x_list, str):
        print("make_data returned error string:", x_list)
        return data

    for x, smi in zip(x_list, smiles_out):
        y = float(0.0)  # Dummy label
        data["prop"].append(y)
        data["feats"].append(x)
        data["smiles"].append(smi)

    return data


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    cmd = set_cuda_visible_device(1)
    os.environ["CUDA_VISIBLE_DEVICES"] = cmd
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    random.seed(0)
    np.random.seed(seed=0)
    generator = rdNormalizedDescriptors.RDKit2DNormalized()
    smiles = 'c1ccc(-c2nc(-c3ccccc3)nc(-c3ccc(-n4c5ccccc5c5c4c4ccccc4n5-c4ccccc4)cc3)n2)cc1'
    data = make_npz_file(smiles)

    autoencoder = AutoEncoder(200)
    autoencoder.to(device)
    save_model = "./trained_model/TADF_self_check.pt"
    trainer = AETrainer(
        autoencoder,
        None,
        optimizer_name=None,
        lr=1e-3,
        n_epochs=1,
        lr_milestones=(),
        batch_size=1,
        weight_decay=0.0,
        save_model=save_model,
        device=device,
        lr_decay=1.0,
    )

    _, unlabel_xs, _,_ = get_dataset_dataloader(
        data, batch_size=1, num_workers=1
    )
    save_model = "./trained_model/TADF_self_check.pt"
    trainer.ae_net.load_state_dict(torch.load(save_model,weights_only=True))
    trainer.ae_net.eval()

    unlab = score(trainer, unlabel_xs, device).cpu().detach().numpy().reshape(-1)
    TADF_score = unlab[0]
    print(TADF_score)

