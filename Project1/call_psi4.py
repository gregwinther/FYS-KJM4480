'''
Program that makes a imple call to Psi4, returning:
Energy from computation, nuclear repulsion energy,
atomic orbital Hamiltonian, electron repulsion integral,
overlap matrix, number of spins up / down

Example by Simen Kvaal (2017)
'''

import psi4
import numpy as np

def call_psi4(mol_spec, extra_opts = {}):
    mol = psi4.geometry(mol_spec)

    # Set options to Psi4.
    opts = {'basis': 'cc-pVDZ',
                     'reference' : 'uhf',
                     'scf_type' :'direct',
                     'guess' : 'core',
                     'guess_mix' : 'true',
                     'e_convergence': 1e-7}

    opts.update(extra_opts)

    # If you uncomment this line, Psi4 will *not* write to an output file.
    #psi4.core.set_output_file('output.dat', False)

    # Supress writing to terminal
    psi4.core.be_quiet()

    psi4.set_options(opts)

    # Set up Psi4's wavefunction. Psi4 parses the molecule specification,
    # creates the basis based on the option 'basis' and the geometry,
    # and computes all necessary integrals.
    wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option('BASIS'))

    # Get access to integral objects
    mints = psi4.core.MintsHelper(wfn.basisset())

    # Get the integrals, and store as numpy arrays
    T = np.asarray(mints.ao_kinetic())
    V = np.asarray(mints.ao_potential())
    H = T + V
    W = np.asarray(mints.ao_eri())
    S = np.asarray(mints.ao_overlap())

    # Get number of basis functions and number of occupied orbitals of each type
    nbf = wfn.nso()
    nalpha = wfn.nalpha()
    nbeta = wfn.nbeta()


    # Perform a SCF calculation based on the options.
    SCF_E_psi4 = psi4.energy('SCF')
    Enuc = mol.nuclear_repulsion_energy()

    # We're done, return.
    return SCF_E_psi4, Enuc, H,W,S, nbf, nalpha, nbeta
