import json
import numpy as np
import math
import torch

class Reconstructor():
    
    def __init__(self, path: str):
        """
        Takes path to reconstruction_vectorized.json
        """
        
        with open(path, 'r') as f:
            reconstruction = json.loads(f.read())
        
        # =====Beta carbon virtualization==========
        self.CA_CB_dict = torch.tensor(reconstruction['CA_CB_dict']) # averge length of CA-CB bond length, by AA type
        self.N_C_CA_dict = torch.tensor(reconstruction['N_C_CA_dict']) # average N-C-CA bond angle
        self.N_C_CA_CB_dict = torch.tensor(reconstruction['N_C_CA_CB_dict']) # avereg N-C-CA-CB dihedral angle
        
        # =====Chi Angle reconstruction===========
        self.ideal_bond_lengths = torch.tensor(reconstruction['ideal_bond_lengths']) # average bonds lengths by AA + chi
        self.ideal_bond_angles = torch.tensor(reconstruction['ideal_bond_angles']) # averge bond angle (NOT dihedral) by AA + chi
    
        AAs = ['W', 'N', 'I', 'G', 'H', 'V', 'M', 'T', 'S', 'Y', 'Q', 'F', 'E', 'K', 'P', 'C', 'L', 'A', 'D', 'R']
        self.aa_to_idx = {aa: i for i, aa in enumerate(AAs)}
    
    def reconstruct(self, atoms: List[torch.Tensor], AA: List[str], chi_angles: torch.Tensor) -> Tuple(List[torch.Tensor], list[torch.Tensor]):
        '''
        For backbone atoms, order is C, O, N, CA
        The order is important! In particular N-CA as they are used to compute the first and second chi angles
        
        --- Input ---
        atoms: list of length 4. each element is a torch tensor containing a batch (batch size = B) of atom coordinates for the atoms [C, O, N, CA], in this order
        AA: list of single-char amino-acid identifiers, of length B
        chi_angles: Tensor of shape (B x 4) containing desired chi angles, with NaN values for invalid angles
        
        --- Output ---
        ordered_placed_atoms: list of atom coordinates that have been placed, in the order of placement
        ordered_norms: list of plane norms of the sidechains, 
        '''
        
        AA = torch.tensor([self.aa_to_idx[aa] for aa in AA])
    
        CB_norm = get_normal_vector__torch_batch(atoms[2], atoms[0], atoms[3])
        
        atoms.append(get_atom_place__torch_batch(
             CB_norm, self.N_C_CA_CB_dict[AA], 
             atoms[0], atoms[3], 
             self.CA_CB_dict[AA],
             self.N_C_CA_dict[AA]
        )[0])
        
        ordered_norms = []
        
        #====place side chain atoms===========
        for chi_num in range(4):
            
            p1_norm = get_normal_vector__torch_batch(atoms[-3], atoms[-2], atoms[-1])
            chi_angle = chi_angles[:, chi_num]
            
            bond_length, bond_angle = self.ideal_bond_lengths[AA, chi_num], self.ideal_bond_angles[AA, chi_num]
            
            predicted_place, norm = get_atom_place__torch_batch(p1_norm, chi_angle, atoms[-2], atoms[-1], bond_length, bond_angle)

            # Use predicted place downstream
            atoms.append(predicted_place)
            ordered_norms.append(norm)
           
        ordered_placed_atoms = atoms[4:]

        return ordered_placed_atoms, ordered_norms
        
    