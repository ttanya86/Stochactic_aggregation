# Stochactic_aggregation
Kinetic Monte Carlo model for amyloid aggregtaion is presented. The model includes aggregation of kinetically favored off pathway oligomers and thermodynamically stable on pathway fibrils with secondary nucleation.

Necessary on/off aggregation rates are taken from the article "Mechanism of Fibril and Soluble Oligomer Formation in Amyloid Beta and Hen Egg White Lysozyme Proteins". Carlos Perez, Tatiana Miti, Filip Hasecke, Georg Meisl, Wolfgang Hoyer, Martin Muschol, and Ghanim Ullah. The Journal of Physical Chemistry B 2019 123 (27), 5678-5689,DOI: 10.1021/acs.jpcb.9b02338

Necessary diffusion rates are taken from the article "Hydration and Hydrodynamic Interactions of Lysozyme: Effects of Chaotropic versus Kosmotropic Ions", Avanish S. Parmar, Martin Muschol,Biophys. Journal 97 (2), DOI:10.1016/j.bpj.2009.04.045
and from the article “Spatial extent of charge repulsion regulates assembly pathways for lysozyme amyloid fibrils.” Hill, Shannon E et al. PlosOne vol. 6,4 e18171. 5 Apr. 2011, doi:10.1371/journal.pone.0018171

## The system consists of particles modeled as disks with a certain radius and can not overlap (hard core repulsion),\
# and implicit solvent given trough diffusion and charge on particles, aka electrostatic interactions between particles
# L is a list of lists!! each particle is a list with x-coordinate, y-coordinate, particle ID, slope ID
# temperature enters through Arrehnius like reaction rates for each type of possible reaction
# ionic strength  and viscosity of solvent via Debye length giving the interaction length at each salt concentration

## Particles ID for monoers in a state, aka L[position in L][2]
# free monomer :0,  low end of a on pathway dimer :102, high end of a on pathway dimer :101, 
# middle monomer of a pre-nucleus on pathway :12, high end of fibril :11, low end of fibril :12, middle fibril : 1,
# first monomer atached to the side of a fibril,aka secondary nucleus with positive slope :301,
# first monomer atached to the side of a fibril,aka secondary nucleus with negative slope :302,
# end of secondary nucleated dimer and pre-nucleus :303,
# middle of secondary pre-nucleus :300
# monomer attached to fibril passed a dimer, positive slope :3010,
# monomer attached to fibril passed a dimer, negative slope :3020,
# a secondary fibril can form and not detach immediately, and while is still attached, codes change:
# low, high end of fibril if slope is positive :33, 31
# high end of fibril is slope is negative : 33, 32
# middle of a seondary fibril :30
# monomers of a off aggregate : 20 + number of added monomer, startinf with the central monomer being 20

# radius of the particles is the real radius measured for the protein 
# volume of the box is 1uM x 1uM x particle diameter, so it's virtually a 2D system
