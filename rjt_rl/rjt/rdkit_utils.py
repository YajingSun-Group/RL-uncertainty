from __future__ import annotations

import copy
import logging
from typing import Any, Optional

from rdkit import Chem
from rdkit.Chem.rdchem import BondType
from rdkit.Chem import Draw
logger = logging.getLogger(__name__)


_tmpl_atoms: dict[Any, Any] = {}


class RDKitUtilsBase:
    @staticmethod
    def get_debug_smiles(mol: Chem.Mol) -> str:
        result: str = Chem.MolToSmiles(
            mol, isomericSmiles=False, kekuleSmiles=True, canonical=False
        )
        return result

    @staticmethod
    def dump_mol(mol: Chem.Mol, msg: Optional[str] = None) -> None:
        if msg:
            print(f"=== {msg} ===")
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            bt = bond.GetBondType()
            if str(bt) == "SINGLE":
                sbt = "-"
            elif str(bt) == "DOUBLE":
                sbt = "="
            elif str(bt) == "TRIPLE":
                sbt = "#"
            else:
                sbt = "."

            logger.info(
                f"{a1.GetSymbol()}{a1.GetIdx()}{sbt}{a2.GetSymbol()}{a2.GetIdx()}"
            )

        if msg:
            logger.info("===")

    @staticmethod
    def copy_atom(atom: Chem.Atom) -> Chem.Atom:
        elem_sym = atom.GetSymbol()
        if elem_sym in _tmpl_atoms:
            new_atom = _tmpl_atoms[elem_sym]
        else:
            new_atom = Chem.Atom(elem_sym)
            _tmpl_atoms[elem_sym] = new_atom

        new_atom.SetFormalCharge(atom.GetFormalCharge())
        new_atom.SetAtomMapNum(atom.GetAtomMapNum())
        return new_atom

    @staticmethod
    def get_kekule_mol(smiles: str) -> Chem.Mol:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            Chem.Kekulize(mol)
        return(mol)

    # @staticmethod
    # def get_kekule_mol_0(smiles: str) -> Chem.Mol:
    #     #xxn code
    #     mol = Chem.MolFromSmiles(smiles)
    #     if mol is not None:
    #         ring_info = mol.GetRingInfo()
    #         rings = ring_info.AtomRings()  
    #         bond_in_multiple_rings = {}

    #         for bond_confu in mol.GetBonds():
    #             involved_rings = set()
    #             for ring_idx, ring in enumerate(rings):
    #                 if bond_confu.GetBeginAtomIdx() in ring and bond_confu.GetEndAtomIdx() in ring:
    #                     involved_rings.add(ring_idx)
    #             if len(involved_rings) > 1:
    #                 bond_in_multiple_rings[bond_confu.GetIdx()] = involved_rings

    #         Chem.Kekulize(mol, clearAromaticFlags=True)
    #         suppl = Chem.ResonanceMolSupplier(mol, Chem.KEKULE_ALL)
    #         for resMol in suppl:
    #             # res_smi = Chem.MolToSmiles(resMol, kekuleSmiles=True)
    #             all_double_bonds = True
    #             for bond_idx in bond_in_multiple_rings:
    #                 bond_confu = resMol.GetBondWithIdx(bond_idx)
    #                 if bond_confu.GetBondType() != Chem.BondType.DOUBLE:
    #                     all_double_bonds = False
    #                     break  
    #             if all_double_bonds:
    #                 new_resMol = resMol
    #     return new_resMol

    @staticmethod
    def get_kekule_smiles(mol: Chem.Mol) -> str:
        raise NotImplementedError()

    @classmethod
    def copy_edit_mol(cls, mol: Chem.Mol) -> Chem.Mol:
        new_mol = Chem.RWMol()
        for atom in mol.GetAtoms():
            new_atom = cls.copy_atom(atom)
            new_mol.AddAtom(new_atom)
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            bt = bond.GetBondType()
            new_mol.AddBond(a1, a2, bt)
        return new_mol

    @classmethod
    def check_mol(cls, mol: Chem.Mol) -> None:
        try:
            mol = cls.copy_edit_mol(mol)
            smiles = cls.get_kekule_smiles(mol)
            mol = cls.get_kekule_mol(smiles)
            Chem.SanitizeMol(mol)
        except Exception as e:
            raise RuntimeError(f"mol check failed: {smiles} {e}")

    @staticmethod
    def set_atommap(mol: Chem.Mol, num: int = 0) -> None:
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(num)

    @staticmethod
    def set_index_atommap(mol: Chem.Mol) -> None:
        for i, atom in enumerate(mol.GetAtoms()):
            atom.SetAtomMapNum(i + 1)

    @staticmethod
    def atom_equal(a1: Chem.Atom, a2: Chem.Atom) -> bool:
        result: bool = (
            a1.GetSymbol() == a2.GetSymbol()
            and a1.GetFormalCharge() == a2.GetFormalCharge()
        )
        return result

    @classmethod
    def get_fragment_mol(
        cls, mol: Chem.Mol, atom_ids: list[int]
    ) -> tuple[Chem.Mol, dict[int, int]]:
        mol_copy = Chem.Mol(mol)
        if mol is not None:
            new_resMol=None
            ring_info = mol_copy.GetRingInfo()
            rings = ring_info.AtomRings()  
            bond_in_multiple_rings = {}

            for bond_confu in mol_copy.GetBonds():
                involved_rings = set()
                for ring_idx, ring in enumerate(rings):
                    if bond_confu.GetBeginAtomIdx() in ring and bond_confu.GetEndAtomIdx() in ring:
                        involved_rings.add(ring_idx)
                if len(involved_rings) > 1:
                    bond_in_multiple_rings[bond_confu.GetIdx()] = involved_rings

            Chem.Kekulize(mol_copy, clearAromaticFlags=True)
            suppl = Chem.ResonanceMolSupplier(mol_copy, Chem.KEKULE_ALL)
            for resMol in suppl:
                # res_smi = Chem.MolToSmiles(resMol, kekuleSmiles=True)
                all_double_bonds = True
                for bond_idx in bond_in_multiple_rings:
                    bond_confu = resMol.GetBondWithIdx(bond_idx)
                    if bond_confu.GetBondType() != Chem.BondType.DOUBLE:
                        all_double_bonds = False
                        break  
                if all_double_bonds:
                    new_resMol = resMol
                    # print(Chem.MolToSmiles(new_resMol),'new_resMol')
                    break 
                # else:
                #     print('new_resMol=None')
        else:
            raise ValueError("Input molecule is None.")

        bond_ids = {}
        # print(Chem.MolToSmiles(new_resMol),'new_resMol')
        # print(new_resMol,'new_resMol',Chem.MolToSmiles(mol_copy),'molcopy')
        for i, bond in enumerate(new_resMol.GetBonds()):
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()

            ai1 = a1.GetIdx()
            ai2 = a2.GetIdx()
            # print(bond.GetBondType(),'bond.GetBondType()')
            if ai1 in atom_ids and ai2 in atom_ids:
                bond_ids[i] = bond
                # print(bond.GetBondType(),'bond.GetBondType()',)
        # print(Chem.MolToSmiles(mol),'mol')
        # print(bond_ids,'bond_ids')

        new_mol = Chem.RWMol()
        aid_map = {}
        # print(atom_ids,'atom_ids')
        for ai in atom_ids:
            atom = new_resMol.GetAtomWithIdx(ai) 
            new_atom = cls.copy_atom(atom)
            aid_map[ai] = new_mol.AddAtom(new_atom)

        for _bi, bond in bond_ids.items():
            ai1 = bond.GetBeginAtom().GetIdx()
            ai2 = bond.GetEndAtom().GetIdx()
            bt = bond.GetBondType()
            if not (
                bt == BondType.SINGLE or bt == BondType.DOUBLE or bt == BondType.TRIPLE
            ):
                raise ValueError(f"unsuported bond type: {bt}")
            # print(bt,'bt')
            # if bond.GetIsAromatic() and len(atom_ids)==6 and :
                # print(len(atom_ids),'len(atom_ids)')
            # if len(atom_ids):
            # print(is_aromatic,'is_aromatic',bond.GetBondType())


            new_mol.AddBond(aid_map[ai1], aid_map[ai2], bt)
        return new_mol, aid_map

    @classmethod
    def get_clique_mol(cls, mol: Chem.Mol, atoms: list[int]) -> Chem.Mol:
        raise NotImplementedError()

    @classmethod
    def get_clique_mapping(
        cls, amol: Chem.Mol, atom_ids: list[int], voc_mol: Chem.Mol
    ) -> dict[int, int]:
        # print(atom_ids,'atom_ids',Chem.MolToSmiles(amol))
        new_mol, rmap = cls.get_fragment_mol(amol, atom_ids)
        # print(cls,'cls')
        # print(Chem.MolToSmiles(new_mol),'new_mol')
        # print(Chem.MolToSmiles(voc_mol),'vocal_mol')

        local_map = voc_mol.GetSubstructMatches(new_mol, uniquify=False)
        # print(local_map,'local_map')

        if len(local_map) == 0:
            voc_mol = copy.deepcopy(voc_mol)
            Chem.SanitizeMol(voc_mol)
            Chem.SanitizeMol(new_mol)
            local_map = voc_mol.GetSubstructMatches(new_mol)

        if len(local_map) == 0:
            logger.error(f"ERROR: local_map is empty: {local_map}")
            logger.error(f"voc_mol: {Chem.MolToSmiles(voc_mol)}")
            logger.error(f"amol: {Chem.MolToSmiles(amol)}")
            raise RuntimeError("get_clique_mapping failed")

        local_map = local_map[0]
        rmap2 = {i: local_map[j] for i, j in rmap.items()}
        return rmap2


class RDKitUtils2020(RDKitUtilsBase):
    @classmethod
    def get_clique_mol(cls, mol: Chem.Mol, atoms: list[int]) -> Chem.Mol:
        # try:
        smiles = Chem.MolFragmentToSmiles(mol, atoms, kekuleSmiles=True)
        # except:
            # smiles = 'CC'#Chem.MolFragmentToSmiles(mol, atoms, kekuleSmiles=False)
            # print(Chem.MolFragmentToSmiles(mol, atoms),'smiles'*20)
        new_mol = Chem.MolFromSmiles(smiles, sanitize=False)
        new_mol = cls.copy_edit_mol(new_mol).GetMol()
        new_mol = cls.sanitize(new_mol)
        return new_mol

    @classmethod
    def sanitize(cls, mol: Chem.Mol) -> Chem.Mol:
        try:
            smiles = cls.get_kekule_smiles(mol)
            mol = cls.get_kekule_mol(smiles)
        except Exception:
            logger.error("sanitize failed", exc_info=True)
            return None
        return mol

    @staticmethod
    def get_kekule_smiles(mol: Chem.Mol) -> str:
        result: str = Chem.MolToSmiles(mol, kekuleSmiles=True)
        return result
