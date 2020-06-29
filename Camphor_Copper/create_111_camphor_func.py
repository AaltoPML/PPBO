# Python3 function for creating camphor/Cu(111) geometry
#
# Reads in origin-centered camphor geometry from xyz file 
#
# Arguments:
# Camphor x and y location from system center (fractional unit-cell coords)
# Height of camphor origin from surface (Angstroms) 
# Camphor rotation angles alpha, beta and gamma (degrees)

import os 
path_from_root_to_files = os.getcwd() + '/Camphor_Copper/'


import numpy as np
from ase.io import read, write
from ase.build import fcc111
from ase import Atom
from ase.constraints import FixAtoms

def create_file(camp_dx, camp_dy, camp_origin_height, alpha, beta, gamma):
    ######################### SET PARAMETERS HERE ##############################
    campfile = path_from_root_to_files + 'camphor_Light_T1.xyz' # Camphor xyz file
    cu12_dist = 2.075665 # Relaxed distance between Cu layers 1 and 2 
    cu23_dist = 2.080891 # Relaxed distance between Cu layers 2 and 3 
    latt_const = 3.631584 # Cu cubic lattice constant 
    z_shift = 2.0 # Slab shift in z direction
    vac = 50.0 # Vacuum (separation between slabs)
    nx = 6 # Cu slab dimensions (no. of unit cells)
    ny = 4 
    nz = 4
    nz_fixed = 2 # Number of fixed atom layers at the bottom of the slab
    ############################################################################

    ny = ny * 2 # Doubled for the orthogonal unit-cell definition

    # Create non-relaxed Cu slab
    slab = fcc111('Cu', size=(nx,ny,nz), a=latt_const, \
            vacuum=vac/2.0, orthogonal=True)
    layers = slab.get_tags() # Array of layer numbers
    layer_sep = latt_const / np.sqrt(3) # Unrelaxed layer separation

    # Shift 2 topmost layers to relaxed layer separation
    ind_1 = np.where(layers == 1) # Indices of layer 1
    ind_2 = np.where(layers == 2) # Indices of layer 2
    dz1 = cu12_dist - layer_sep # Shift of layer 1
    dz2 = cu23_dist - layer_sep # Shift of layer 2
    for i in ind_1[0]: # Move layer 1
        slab[i].z = slab[i].z + dz1 + dz2
    for i in ind_2[0]: # Move layer 2
        slab[i].z = slab[i].z + dz2 


    # Move slab to correct height
    slab_bottom = np.min(slab.get_positions()[:,2]) # Slab bottom z
    slab.translate((0.0, 0.0, - slab_bottom + z_shift))

    # Read camphor (in default orientation in origin)
    camp = read(campfile)

    # Rotate camphor
    camp.rotate(alpha, 'x', center=(0,0,0))
    camp.rotate(beta, 'y', center=(0,0,0))
    camp.rotate(gamma, 'z', center=(0,0,0))

    # Move camphor to correct height in the center of slab
    # and translate according to function arguments
    slab_top = np.max(slab.get_positions()[:,2]) # Slab surface z
    dx = ((nx/2) + camp_dx) * np.sqrt(2) * latt_const / 2.0
    dy = ((ny/4) + camp_dy) * np.sqrt(6) * latt_const / 2.0
    camp.translate((dx, dy, slab_top + camp_origin_height))

    # Set fixed atom layers in FHI-aims output file
    ind_fixed = np.where(layers > (nz - nz_fixed))
    const = FixAtoms(ind_fixed[0])
    slab.set_constraint(const)

    # Combine geometries and write to file
    slab.extend(camp)
    write(path_from_root_to_files+'geometry.in', slab) # FHI-aims format

def create_geometry(camp_dx, camp_dy, camp_origin_height, alpha, beta, gamma):
    
    ######################### SET PARAMETERS HERE ##############################
    campfile = path_from_root_to_files+'camphor_Light_T1.xyz' # Camphor xyz file
    cu12_dist = 2.075665 # Relaxed distance between Cu layers 1 and 2 
    cu23_dist = 2.080891 # Relaxed distance between Cu layers 2 and 3 
    latt_const = 3.631584 # Cu cubic lattice constant 
    z_shift = 2.0 # Slab shift in z direction
    vac = 50.0 # Vacuum (separation between slabs)
    nx = 6 # Cu slab dimensions (no. of unit cells)
    ny = 4 
    nz = 4
    nz_fixed = 2 # Number of fixed atom layers at the bottom of the slab
    ############################################################################

    ny = ny * 2 # Doubled for the orthogonal unit-cell definition

    # Create non-relaxed Cu slab
    slab = fcc111('Cu', size=(nx,ny,nz), a=latt_const, \
            vacuum=vac/2.0, orthogonal=True)
    layers = slab.get_tags() # Array of layer numbers
    layer_sep = latt_const / np.sqrt(3) # Unrelaxed layer separation

    # Shift 2 topmost layers to relaxed layer separation
    ind_1 = np.where(layers == 1) # Indices of layer 1
    ind_2 = np.where(layers == 2) # Indices of layer 2
    dz1 = cu12_dist - layer_sep # Shift of layer 1
    dz2 = cu23_dist - layer_sep # Shift of layer 2
    for i in ind_1[0]: # Move layer 1
        slab[i].z = slab[i].z + dz1 + dz2
    for i in ind_2[0]: # Move layer 2
        slab[i].z = slab[i].z + dz2 


    # Move slab to correct height
    slab_bottom = np.min(slab.get_positions()[:,2]) # Slab bottom z
    slab.translate((0.0, 0.0, - slab_bottom + z_shift))

    # Read camphor (in default orientation in origin)
    camp = read(campfile)

    # Rotate camphor
    camp.rotate(alpha, 'x', center=(0,0,0))
    camp.rotate(beta, 'y', center=(0,0,0))
    camp.rotate(gamma, 'z', center=(0,0,0))

    # Move camphor to correct height in the center of slab
    # and translate according to function arguments
    slab_top = np.max(slab.get_positions()[:,2]) # Slab surface z
    dx = ((nx/2) + camp_dx) * np.sqrt(2) * latt_const / 2.0
    dy = ((ny/4) + camp_dy) * np.sqrt(6) * latt_const / 2.0
    camp.translate((dx, dy, slab_top + camp_origin_height))

    # Set fixed atom layers in FHI-aims output file
    ind_fixed = np.where(layers > (nz - nz_fixed))
    const = FixAtoms(ind_fixed[0])
    slab.set_constraint(const)

    # Combine geometries and write to file
    slab.extend(camp)
    return(slab)