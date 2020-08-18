#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#"""
#Created on Tue Aug 28 14:27:30 2018
#
#@author: tatianamiti
#"""
import random, math, os, json
#from matplotlib import pylab
import numpy as np
########################################## system set up ##################################################

## The system consists of particles modeled as disks with a certain radius and can not overlap (hard core repulsion),\
## and implicit solvent given trough diffusion and charge on particles, aka electrostatic interactions between particles
## L is a list of lists!! each particle is a list with x-coordinate, y-coordinate, particle ID, slope ID
## temperature enters through Arrehnius like reaction rates for each type of possible reaction
## ionic strength  and viscosity of solvent via Debye length giving the interaction length at each salt concentration

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

# radius of the particles is the ral radius measured for the protein 
# volume of the box is 1uM x 1uM x particle diameter, so it's virtually a 2D system

############################################ end system set up ######################################################

############################################# sysytem parameters ####################################################
iter_nums = 3000001 # number of moves or reactions to be performed 
nuc_size = 5 # off nucleus size + 1 
temp = 323.15 # temperature in kelvin, K
k_b = 1.38064852e-23 # boltzman constant in J/K
N = 1600 # 0.7 mMol, number of particles in the system
boxSide = 1.0e-06 # length of the box sides
sigma = 1.9e-09 #radius of particles
diameter1 = 2.0*sigma
diamter = round(diameter1,4)
Volume = boxSide*boxSide*2.0*sigma # volume for the simulation box
Volume_uM = Volume/3.8e-18
eta = ((4.0/3.0)*sigma*sigma*sigma*N)/Volume # scaling factor for concentrations

visc = 0.92e-03 # solution viscosity
diff_generic = (k_b*temp)/(6*math.pi*visc) # diffusion coefficient to be scaled by aggregate radius
diff_0 = (k_b*temp)/(6*math.pi*visc*sigma) # monomer diffusion coefficient

kD_50 = 11.4633 # two body interaction parameter from SLS experiments
diff_m = diff_0*(1 + kD_50*eta) # adjusted difusion for the interation in solution

NaCl_concentr = 0.05 # salt concentration of buffer to be used for debye length
ionic_strength = 0.025 + NaCl_concentr #0.025 is of the buffer with no salt

debye_length =  math.sqrt((8.854187817620e-12*70.0*1.38064852e-23*323.15)/\
                          (2000.0*2.56696992e-38*6.0221409e+23*ionic_strength) )
#print("debye length", debye_length)
inter_range_off = round(2.51*sigma,13) #defining th einteracting range in terms of debye length
inter_range_on = round(3.51*sigma,13)

################################################ end system parameters ###############################################

########################################## Reaction energy bariers and rates #####################################################
dimer_on_association_e = 9e-2*1e-03*0.2777#math.exp(-3.1234e-20/ (k_b * temp))           
pre_nucleus_on_association_e = 9e-2*1e-03*0.2777#math.exp(-0.09234e-20/ (k_b * temp))
pre_nucleus_on_dissociation_e = 3.96e-04*1e-03*0.2777#math.exp(-5.1234e-20/ (k_b * temp))     

fibril_association_e = 1.98e6*1e-03*0.2777#math.exp(-1.0e-25/ (k_b * temp))
fibril_dissociation_e = 1.98e-08*1e-03*0.2777#math.exp(-1.0e-18/ (k_b * temp))

dimer_off_association_e = 3.6e04*1e-03*0.2777#math.exp(-3.65e-20/(k_b * temp))
dimer_off_dissociation_e = 3.6e-04*1e-03*0.2777#math.exp(-2.5e-19/(k_b * temp))            
pre_nucleus_off_association_e = 1.8e04*1e-03*0.2777#math.exp(-1.1099e-22/(k_b * temp))
aggregate_off_dissociation_e = 3.6e-04*1e-03*0.2777#math.exp(-5.0e-20/(k_b * temp))        

second_monomer_attach_e = 1.929e2*1e-03*0.2777#math.exp(-0.0101798e-20/(k_b * temp))
second_pre_nucleus_association_e = 1.929e2*1e-03*0.2777#math.exp(-1.0034e-27 /(k_b * temp))
second_fibril_detach_e = 1.9e02#math.exp(-0.10e-26 / (k_b * temp))
second_monomer_detach_e = 3.96e-4*1e-03*0.2777#math.exp(-0.3010e-19/ (k_b * temp)) 

monomer_diffusion_e = 5e0*1e-03*0.2777#math.exp(-2.0e-20/ (k_b * temp))               
mono_inter_diffusion_on_e = 9e0*1e-03*0.2777#math.exp(-2.20e-19/ (k_b * temp)) 
mono_inter_diffusion_off_e = 9e0*1e-03*0.2777#math.exp(-4.20e-19/ (k_b * temp)) 
pre_on_diffusion_e = 5e0*1e-03*0.2777#math.exp(-2.00e-22/ (k_b * temp)) 
pre_off_diffusion_e = 5e0*1e-03*0.2777#math.exp(-2.00e-20/ (k_b * temp))                 

kinetic_rates = {'dimer_on_association':dimer_on_association_e,           'pre_nucleus_on_association':pre_nucleus_on_association_e,\
                     'fibril_association':fibril_association_e,                 'second_monomer_attach':second_monomer_attach_e,\
                     'second_pre_nucleus_association':second_pre_nucleus_association_e, 'dimer_off_association':dimer_off_association_e,\
                     'dimer_off_dissociation':dimer_off_dissociation_e,            'pre_nucleus_off_association':pre_nucleus_off_association_e,\
                     'aggregate_off_dissociation':aggregate_off_dissociation_e,        'second_fibril_detach':second_fibril_detach_e, \
                     'pre_nucleus_on_dissociation':pre_nucleus_on_dissociation_e,     'fibril_dissociation':fibril_dissociation_e,\
                     'monomer_diffusion':monomer_diffusion_e,                'pre_on_diffusion':pre_on_diffusion_e,\
                     'pre_off_diffusion':pre_off_diffusion_e,                'second_monomer_detach':second_monomer_detach_e,\
                     'mono_inter_diffusion_on':mono_inter_diffusion_on_e,
                     'mono_inter_diffusion_off':mono_inter_diffusion_off_e}
print(kinetic_rates)

################################################## end reaction rates bariers and rates ###############################


########################################## Reaction and populations lists ################################################################
dimer_on_association,dimer_off_association = [],[]
dimer_on_dissociation = []
dimer_off_dissociation = []
pre_nucleus_on_association,pre_nucleus_off_association = [],[]
pre_nucleus_on_dissociation,pre_nucleus_off_dissociation = [],[]
fibril_association,fibril_dissociation = [],[]
second_monomer_attach, second_monomer_detach = [],[]
second_dimer_association, mono_inter_diffusion_on = [],[]
second_pre_nucleus_association, second_pre_nucleus_dissociation = [],[]
second_fibril_association,second_fibril_dissociation = [],[]
second_fibril_detach = [] #monomer to be detached is passed as argument

monos_inter_diffusion_on = [] #monomers within on pathway range interaction
monos_inter_diffusion_off = [] # monomers within off pathway interaction

dimer_on_diffusion, dimer_off_diffusion = [],[] #entire aggregate is passed as argument
trimer_on_diffusion, trimer_off_diffusion = [],[]
tetramer_on_diffusion, tetramer_off_diffusion = [],[]
pentamer_on_diffusion, pentamer_off_diffusion = [],[]
nucleus_off_numb = [] # entire nucleus is passed
monomers = [] # the same as monomer_diffusion

############################################# end reaction and populations lists ####################################################


########################################## DISTANCE BETWEEN TWO PARTICLES; takes in 2 lists from L ############
def dist(xm0m,ym0m):
    d_x = xm0m[0] - ym0m[0]  
    d_y = xm0m[1] - ym0m[1]   
    return  math.sqrt(d_x**2 + d_y**2)
########################################### end function ##############################################
    
####################################### markov chain algorithm to creadte a random initial distribution, L ######
# takes in the paricles on a grid and moves them randomly from their position by delta_n ########################
# fast way of generating distribution for initial L, has to go through a lot of iterations for a true randomized system
    
def markov_disks_box(LL,sigma, delta,n_steps):
    for steps in range(n_steps):
        a2q = random.choice(LL)
        b0 = [a2q[0] + random.uniform(-delta,delta), a2q[1] + random.uniform(-delta,delta),0,0]
        min_dist = min(dist(b0,c0) for c0 in LL if c0 != a2q)
#        print(min_dist)
        box_cond = min(b0[0], b0[1]) < sigma or max(b0[0], b0[1]) > 1.0e-06 - sigma
#        print(box_cond)
        if box_cond == True:
            a2q[:] = a2q
        elif min_dist > 2.0 * sigma:
           a2q[:] = b0
#    print(L)
    return (LL)

output_dir = 'aggregation_movie_50_450mM_tst3_diffusion'
if not os.path.exists(output_dir): os.makedirs(output_dir)
##############################################


########################################## snapshot of the populations at a given time ################################
# GIVES A SNAPSHOT OF THE PARTICLE DISTRIBUTION AT A GIVEN TIME; 
#TAKES IN THE LIST OF PARTICLE AND THE COLORS ASSIGNED TO EACH PARTICLE 
#img = 0
#def snapshot(pos, colors):
#
#    global img
#
#    pylab.subplots_adjust(left=0.10, right=0.90, top=0.90, bottom=0.10)
#
#    pylab.gcf().set_size_inches(20, 20)
#
#    pylab.axis([-0.99e-06, 1.99e-06, -0.99e-06, 1.99e-06])
#
#    pylab.setp(pylab.gca(), xticks=[0, 1.0e-06], yticks=[0, 1.0e-06])
#    pylab.xticks(np.linspace(0, 1.0e-06, 9, endpoint=True))
#    pylab.yticks(np.linspace(0, 1.0e-06, 9, endpoint=True))
#    for (x, y, z, v), c in zip(pos, colors):
#
#        circle = pylab.Circle((x, y), radius=sigma, fc=c)
#
#        pylab.gca().add_patch(circle)
#
##    pylab.savefig(os.path.join(output_dir, '%d.png' % img), transparent=True)
#    pylab.grid(True)
#    pylab.show()
#
#
#    pylab.close()
#
#    img += 1
############################################################# end snapshot ####################################
#
#
########################## modified snapshot to zoom out ########################################################
#def snapshot2(pos, colors):
#
#    global img
#
#    pylab.subplots_adjust(left=0.10, right=0.90, top=0.90, bottom=0.10)
#
#    pylab.gcf().set_size_inches(20, 20)
#
#    pylab.axis([-0.0e-06, 1.0e-06, -0.0e-06, 1.0e-06])
#
#    pylab.setp(pylab.gca(), xticks=[0, 1.0e-06], yticks=[0, 1.0e-06])
#    pylab.xticks(np.linspace(0, 1.0e-06, 9, endpoint=True))
#    pylab.yticks(np.linspace(0, 1.0e-06, 9, endpoint=True))
#
#    for (x, y, z, v), c in zip(pos, colors):
#
#        circle = pylab.Circle((x, y), radius=sigma, fc=c)
#
#        pylab.gca().add_patch(circle)
#
#    pylab.savefig(os.path.join(output_dir, '%d.png' % img), transparent=True)
#    pylab.grid(True)
#    pylab.show()
#
#
#    pylab.close()
#    img += 1
################################################### end modified snapshot ##############################################


#####################Create a list L from scratch or load a pre-exoisting one from a file #######################
colors = [] # list of colors for creating snapshots
sqrt_N = math.sqrt(N)
sqrt_N1 = int(sqrt_N) + 1
x_coord = [(i*0.98e-06)/sqrt_N for i in range (1,sqrt_N1)] # x coordinate on the grid
y_coord = [(i*0.98e-06)/sqrt_N for i in range (1,sqrt_N1)] # y coordinate on the grid

L2 = []
for jqa1 in x_coord:
    for lqa1 in y_coord:
        L2.append([jqa1,lqa1,0,0])

#sigma_sq = sigma ** 2
delta = 2.0e-08 # shift distance allowed by the particle in a step in markov_disks_box
n_steps = 15000 # number of attempts to move particles ofrom fixed positions on the grid
if os.path.isfile('test_L.txt'):
    with open('test_L.txt', 'r') as f:
        L = json.loads(f.read())
else:
    L = markov_disks_box(L2,sigma,delta,n_steps)   
#snapshot2(L,colors)
###################################### end creating L #########################################################
    

## STARTING CODING THE EVENTS
######################################## Making the list of all the monomers in the system #####################
def avail_monos(G):
    list_monos =[]
    for gw1 in range(0,len(G)):
        if G[gw1][2] == 0:
            list_monos.append(gw1)
            
    return list_monos
######################################## end making monomers list #############################################


####################################### monomer diffusion reaction ############################################
# MOVES  A MONOMER A DISTANCE in accordance to the random time chosen;    
# takes in a randomely chosen monomer, radius of monomer and the current list of particles   
# once the free monomer is moved, all reaction lists it had been in have to be modofied
# find new interation monomers and update the interaction lists 
def markov_mono_box(M,index0,sigma,diff_coeff,monomers1,monos_inter_diffusion_on1, monos_inter_diffusion_off1,
                    dimer_on_association1, dimer_off_association1, pre_nucleus_on_association1,
                    fibril_association1, second_monomer_attach1, second_fibril_association1, 
                    second_pre_nucleus_association1, second_dimer_association1, pre_nucleus_off_association1,STOP_TEST1):  
# index0 is the position number in the list of the monomer to be diffused                                       
#    print('monomer diffusion', index0)
    boxSide = 1.0e-06
    a1a = []
    a1a[:] = M[index0][:]
    atmp_dimer_on_association1 = [] # find free monomers  within interaction range in pathway
    atmp_mono_off = [] # find free monomers within interaction range off pathway
    atmp_mono_pre_on_nucleus0 = [] # find pre nucleus on ends monomers within interaction range
    atmp_mono_fibrils = [] # find fibril ends monomers within interaction range
    atmp_mono_second_mono_2 = [] # find fibril monomer within interaction range for secondary nucleation
    atmp_mono_second_fibril_2 = [] #find second fibril end within interaction range for secondary fibirl elongation
    atmp_mono_second_dimer_2 = [] #find second pre nucleus end within interaction range 
    atmp_mono_off_2 = [] # find off pre nucleus within interaction range
    atmp_mono_second_pre_2 = [] 
    atmp_mono_inter_on = []
    atmp_mono_inter_off = []
    atmp_pre_nucleus_on_association1 = []
    atmp_fibril_association1 = []
    atmp1_dimer_on_association1 = []
    atmp_dimer_off_association1 = []
    atmp_second_monomer_attach1 = []
    atmp_second_fibril_association1 = []
    atmp_second_pre_nucleus_association1 = []
    atmp_second_dimer_association1 = []
    atmp_pre_nucleus_off_association1 = []
    atest_dist_mono_test = False
#    print(dimer_on_association1)
#    finding new positio for the diffusing monomer
    atheta = random.uniform(-math.pi,math.pi)
    ab = [a1a[0] + diff_coeff*math.cos(atheta), a1a[1] + diff_coeff*math.sin(atheta),0,0]
#    print('new monomer coordinates are', ab)
# check periodic boundary conditions
    if ab[0] > 0:            
        ab[0] = ab[0]%boxSide
    if ab[0] < 0:
        ab[0] = boxSide - abs(ab[0])%boxSide
    if ab[1] > 0:
        ab[1] = ab[1]%boxSide
    if ab[1] < 0:
        ab[1] = boxSide - abs(ab[1])%boxSide      
#    print('monomer coordinates after boundary conditions', ab)
    
    for arqw in range(len(M)):     
        if arqw != index0: #checking that it's not the same position and it chooses possible interation partners and \
            # check for hard core overlaps
            test_dist_mono = round(dist(ab,M[arqw]),13)
#            print(test_dist_mono)
            if test_dist_mono < round(2.0*sigma,13):#3.8e-09:
#                print(test_dist_mono)
                atest_dist_mono_test = True
                STOP_TEST1.append(0)
#                print("NOT PASSED")
#                print('didnt pass test in momo')
#                break
                return (M)
            
#            elif M[arqw][2] in [12, 21, 22, 23, 24, 3010, 3020, 300,31, 32, 30]:
#                pass
            elif  (M[arqw][2] == 0) and (test_dist_mono >= 2.0*sigma) and (test_dist_mono < inter_range_on):                
                atmp_dimer_on_association1.append([index0, arqw]) # adding a potential interacting pair to form a dimer to a tmp list         
#                print('dimer on found',M[index0], M[irq])
                atmp_mono_inter_on.append(arqw)
                atmp_mono_inter_on.append(index0)
            elif (M[arqw][2] == 0) and (test_dist_mono >= 2.0 * sigma) and (test_dist_mono < inter_range_off):
                atmp_mono_off.append([index0, arqw])
#                print('formed dimer off in monomer routine', M[index0], M[arqw])
                atmp_mono_inter_off.append(arqw)
                atmp_mono_inter_off.append(index0)
            elif (M[arqw][2] in [101, 102]) and (test_dist_mono >= 2.0*sigma) and (test_dist_mono < inter_range_on):
                atmp_mono_pre_on_nucleus0.append([index0, arqw])
#                print('formed pre on in monomer routine', M[index0], M[arqw])
            elif (M[arqw][2] in [121, 122]) and (test_dist_mono >= 2.0*sigma) and (test_dist_mono < inter_range_on):
                atmp_mono_fibrils.append([index0, arqw])
            elif (M[arqw][2] == 1) and (test_dist_mono >= 2.0*sigma) and (test_dist_mono < inter_range_on):
                atmp_mono_second_mono_2.append([index0, arqw])            
#            elif (M[arqw][2] == 33) and (test_dist_mono >= 2.0*sigma) and (test_dist_mono < inter_range_on):
#                atmp_mono_second_fibril_2.append([index0, arqw])
            elif (M[arqw][2] in [301, 302]) and (test_dist_mono >= 2.0*sigma) and (test_dist_mono < inter_range_on):
                atmp_mono_second_dimer_2.append([index0, arqw])
            elif (M[arqw][2] == 303) and (test_dist_mono >= 2.0*sigma) and (test_dist_mono < inter_range_on):
                atmp_mono_second_pre_2.append([index0, arqw])   
            elif (M[arqw][2] == 20) and (test_dist_mono >= 2.0*sigma) and (test_dist_mono < inter_range_off):
                atmp_mono_off_2.append([index0, arqw])            
# changing old coordinates for new ones and updating relevant population/reaction lists
#    print('test_dist_mono', test_dist_mono, atest_dist_mono_test)
    if atest_dist_mono_test == False:
        STOP_TEST1.append(1)
#        print("PASSED")
        M[index0][:] = ab[:]
#        print("true")
        if index0 in monos_inter_diffusion_on1: # removing the free monomer from previous reaction lists
            monos_inter_diffusion_on1.remove(index0)
            
        if index0 in monos_inter_diffusion_off1:
            monos_inter_diffusion_off1.remove(index0)
            
        for axc1 in dimer_on_association1:
            if index0 in axc1:
                atmp1_dimer_on_association1.append(axc1)
        for axc11 in atmp1_dimer_on_association1:
            dimer_on_association1.remove(axc11)
                
        for axc3 in pre_nucleus_on_association1:
            if index0 in axc3:
                atmp_pre_nucleus_on_association1.append(axc3)
        for axc33 in atmp_pre_nucleus_on_association1:
            pre_nucleus_on_association1.remove(axc33)
            
        for axc4 in fibril_association1:
            if index0 in axc4:
                atmp_fibril_association1.append(axc4)
        for axc44 in atmp_fibril_association1:
            fibril_association1.remove(axc44)

        for axc2 in dimer_off_association1:
            if index0 in axc2:
                atmp_dimer_off_association1.append(axc2)
        for axc22 in atmp_dimer_off_association1:
            dimer_off_association1.remove(axc22)
        
        for axc9 in pre_nucleus_off_association1:
            if index0 in axc9:
                atmp_pre_nucleus_off_association1.append(axc9)
        for axc99 in atmp_pre_nucleus_off_association1:
            pre_nucleus_off_association1.remove(axc99)
            
        for axc5 in second_monomer_attach1:
            if index0 in axc5:
                atmp_second_monomer_attach1.append(axc5)
        for axc55 in atmp_second_monomer_attach1:
            second_monomer_attach1.remove(axc55)
#        second_fibril_association1 = [axc6 for axc6 in second_fibril_association1 if not index0 in axc6]
        for axc6 in second_fibril_association1:
            if index0 in axc6:
                atmp_second_fibril_association1.append(axc6)
        for axc66 in atmp_second_fibril_association1:
            second_fibril_association1.remove(axc66)
#        second_pre_nucleus_association1 = [axc7 for axc7 in second_pre_nucleus_association1 if not index0 in axc7]
        for axc7 in second_pre_nucleus_association1:
            if index0 in axc7:
                atmp_second_pre_nucleus_association1.append(axc7)
        for axc77 in atmp_second_pre_nucleus_association1:
            second_pre_nucleus_association1.remove(axc77)
#        second_dimer_association1 = [axc8 for axc8 in second_dimer_association1 if not index0 in axc8]
        for axc8 in second_dimer_association1:
            if index0 in axc8:
                atmp_second_dimer_association1.append(axc8)
        for axc88 in atmp_second_dimer_association1:
            second_dimer_association1.remove(axc88)
            
        for akl1 in atmp_mono_inter_on:            
            monos_inter_diffusion_on1.append(akl1) # adding to the list of monomers within interaction range for difusion 

        for akb1 in atmp_mono_inter_off:            
            monos_inter_diffusion_off1.append(akb1) # adding to the list of monomers within interaction range for difusion   

        for axxw0 in atmp_dimer_on_association1:
            dimer_on_association1.append(axxw0) # adding to the dimer_on_association list 
            
        for aatx8 in atmp_mono_off:
            dimer_off_association1.append(aatx8)

        for aqq1r in atmp_mono_pre_on_nucleus0:            
            pre_nucleus_on_association1.append(aqq1r) # adding to pre nucleus on association list

        for aww1 in atmp_mono_fibrils:
            fibril_association1.append(aww1) # = fibril_association1 + tmp_mono_fibrils # adding to the fibril association list 

        for aee1 in atmp_mono_second_mono_2:
            second_monomer_attach1.append(aee1) # = second_monomer_attach1 + tmp_mono_second_mono_2# adding to the  secondary monomer attach list

        for arr7 in  atmp_mono_second_fibril_2:
            second_fibril_association1.append(arr7)# = second_fibril_association1 + tmp_mono_second_fibril_2 # adding to the secondary fibril elongation list

        for att7 in atmp_mono_second_pre_2:
            second_pre_nucleus_association1.append(att7) # = second_pre_nucleus_association1 + tmp_mono_second_pre_2 # adding  to the secondary pre nucleus association list

        for ayy7 in atmp_mono_second_dimer_2:
            second_dimer_association1.append(ayy7) # = second_dimer_association1 + tmp_mono_second_pre_2 # adding to the secondary dimer association list                             

        for auu7 in atmp_mono_off_2:
            pre_nucleus_off_association1.append(auu7) # = pre_nucleus_off_association1 + tmp_mono_off_2 # adding to the pre nucleus off association list
    atmp_dimer_on_association1 = [] # find free monomers  within interaction range in pathway
    atmp_mono_off = [] # find free monomers within interaction range off pathway
    atmp_mono_pre_on_nucleus0 = [] # find pre nucleus on ends monomers within interaction range
    atmp_mono_fibrils = [] # find fibril ends monomers within interaction range
    atmp_mono_second_mono_2 = [] # find fibril monomer within interaction range for secondary nucleation
    atmp_mono_second_fibril_2 = [] #find second fibril end within interaction range for secondary fibirl elongation
    atmp_mono_second_dimer_2 = [] #find second pre nucleus end within interaction range 
    atmp_mono_off_2 = [] # find off pre nucleus within interaction range
    atmp_mono_second_pre_2 = [] 
    atmp_mono_inter_on = []
    atmp_mono_inter_off = []
    atmp_pre_nucleus_on_association1 = []
    atmp_fibril_association1 = []
    atmp1_dimer_on_association1 = []
    atmp_dimer_off_association1 = []
    atmp_second_monomer_attach1 = []
    atmp_second_fibril_association1 = []
    atmp_second_pre_nucleus_association1 = []
    atmp_second_dimer_association1 = []
    atmp_pre_nucleus_off_association1 = []
    return (M)
######################################### monomer diffusion function ###################################################          


############################### CREATES A ON DIMER WITH LOW AND HIGH ENDS AS id AND SLOPE ID ################################    
# two monomers form a on pathway dimer
# both need to be taken out of all previous intercation lists
# both need to be verified for new interaction partners
# distance from other particles has to be more than 2*sigma for both of them
# inedx is the monomer that doesn't change position, bb is the monomer that changes position and goes next to index forming a dimer with slope calculated from the posistions of the two monomers
def dimer_on(H,bindex,sigma,bb, monomers2,monos_inter_diffusion_on2, monos_inter_diffusion_off2,
                    dimer_on_association2, dimer_off_association2, pre_nucleus_on_association2,
                    fibril_association2, second_monomer_attach2, second_fibril_association2, 
                    second_pre_nucleus_association2, second_dimer_association2, pre_nucleus_off_association2,
                    dimer_on_dissociation2, dimer_on_diffusion2,STOP_TEST2):     
# index is the monoers that doesn't move
    b11 = H[bindex] # the monomer to stay in the fiexd position
#    print('dimer on',[bindex,H[bindex],bb,H[bb]])
    btmp_dimer_on_1 = []
    
    bslope_on1 = (b11[1]-H[bb][1])/(b11[0]-H[bb][0])
    bslope_on = round(bslope_on1,8)
    bangle = math.atan(bslope_on)
    boxSide = 1.0e-06
    btest_dist_dimer_on1 = False
    btest_dist_dimer_on2 = False
    
# tmp_dimer_on are the coordinates of the moving monomer given that it passes the distance test
    if H[bindex][1] < H[bb][1]:
        btmp_dimer_on_1.append(b11[0] + 2.0*sigma*math.cos(bangle))
        btmp_dimer_on_1.append(b11[1] + 2.0*sigma*math.sin(bangle))
    elif H[bindex][1] > H[bb][1]:
        btmp_dimer_on_1.append(b11[0] - 2.0*sigma*math.cos(bangle))
        btmp_dimer_on_1.append(b11[1] - 2.0*sigma*math.sin(bangle))
    btmp_dimer_on = btmp_dimer_on_1[:]
    btmp_on11 = []
    btmp_inter_range_on_mono11 = []
    btmp_dimer_on_remove = []
    btmp_prenuc_on_remove = []
    btmp_fibirl_association = []
    btmp_dimer_off_association2 = []
    btmp_second_monomer_attach2 = []
    btmp_second_fibril_association2 = []
    btmp_second_pre_nucleus_association2 = []
    btmp_second_dimer_association2 = []
    btmp_pre_nucleus_off_association2 = []
#finding new coordinate for the moving free monomer, b    
    if btmp_dimer_on_1[0] > 0:            
        btmp_dimer_on[0] = btmp_dimer_on_1[0]%boxSide
    if btmp_dimer_on_1[0] < 0:
        btmp_dimer_on[0] = boxSide - abs(btmp_dimer_on_1[0])%boxSide
    if btmp_dimer_on_1[1] > 0:
        btmp_dimer_on[1] = btmp_dimer_on_1[1]%boxSide
    if btmp_dimer_on_1[1] < 0:
        btmp_dimer_on[1] = boxSide - abs(btmp_dimer_on_1[1])%boxSide
# iterating through the entire list of particles and cjecking for overalp between the new dimer and other particles
# careful in pre_nucleus on routine, position 0 is the monomer, position 1 is the aggregate
    for bxwj in range(len(H)):        
        if bxwj != bb:
            if bxwj != bindex:
                bdist_tmp11 = round(dist(btmp_dimer_on,H[bxwj]),13)   # distance between the moved to a new place monomer and other particles, bb
                bdist_tmp21 = round(dist(H[bindex],H[bxwj]),13)       # distance between the fixed monomer and other particles, bindex
                if bdist_tmp11 < round(2.0*sigma,13):
                    btest_dist_dimer_on1 = True
                    STOP_TEST2.append(0)
#                    print("NOT PASSED")
#                    print('false in dimer first', btest_dist_dimer_on1)
#                    break
                    return (H)
                elif bdist_tmp21 < round(2.0*sigma,13):
                    btest_dist_dimer_on2 = True
                    STOP_TEST2.append(0)
#                    print("NOT PASSED")
#                    print('false in dimer second', btest_dist_dimer_on2)
#                    break
                    return (H)
                elif H[bxwj][2] == 0:
                    if (bdist_tmp11 >= round(2.0*sigma,13)) and (bdist_tmp11 < inter_range_on):
# finding potential monomers within interaction range for the newly moved monomer, it would be to for an on pathway trimer,
#so it would be a potential pre on nucleus intercating pair, and the monomer found would be incorporated in interacting monomers list
                        btmp_on11.append([bxwj,bb]) # interacting pre on nucleus pair
#                        print('new pre nucleus formed in dimer on functio is', [bxwj,bb], H[bxwj], H[bb])
                        btmp_inter_range_on_mono11.append(bxwj) # monomer witin interating range
                
                    if (bdist_tmp21 >= round(2.0*sigma,13)) and (bdist_tmp21 < inter_range_on):
# finding potential monomers within interaction range for the newly moved monomer, it would be to for an on pathway trimer,
#so it would be a potential pre on nucleus intercating pair, and the monomer found would be incorporated in interacting monomers list
                        btmp_on11.append([bxwj,bindex]) # interacting pre on nucleus pair
#                        print('new pre nucleus formed in dimer on functio is', [bxwj,bindex], H[bxwj], H[bindex])
                        btmp_inter_range_on_mono11.append(bxwj) # monomer witin interating range
#    print('dimer distance test', btest_dist_dimer_on1,btest_dist_dimer_on2)
#changing old coordinates with the new ones and updating relevant populations/reaction lists
    if (btest_dist_dimer_on1 == False) and (btest_dist_dimer_on2 == False):
        STOP_TEST2.append(1)
#        print("PASSED")
#        print('passed distance in dimer')

        if H[bindex][1] > btmp_dimer_on_1[1]:
            H[bindex][2] = 101             #hiher y coordinate, giving a direction to the fibril end
            btmp_dimer_on.append(102)          #lower y coordinate, giving a direction to the fibril end
        elif btmp_dimer_on_1[1] > H[bindex][1]:
                H[bindex][2] = 102
                btmp_dimer_on.append(101)

        H[bindex][3] = bslope_on
        btmp_dimer_on.append(bslope_on)
        H[bb][:] = btmp_dimer_on[:]

        monomers2[:] = [bx1 for bx1 in monomers2 if bx1 != bindex]#

        monomers2[:] = [bbx1 for bbx1 in monomers2 if bbx1 != bb]#

# first, remove both monomers from their potential existing interacting lists
        if bindex in monos_inter_diffusion_on2: 
            monos_inter_diffusion_on2.remove(bindex)# removing the free monomer from previous interation reaction lists
            
        for bttx in dimer_on_association2:
            if bindex in bttx:
#                print('remoing a dimer on', ittx)
                btmp_dimer_on_remove.append(bttx)   
            elif bb in bttx:
                btmp_dimer_on_remove.append(bttx)                   
        for bicv in btmp_dimer_on_remove:       
            dimer_on_association2.remove(bicv)
            
        for bttx0 in pre_nucleus_on_association2:
            if bindex in bttx0: 
#                print('remoing a dimer on', ittx)
                btmp_prenuc_on_remove.append(bttx0)                
            elif bb in bttx0:
                btmp_prenuc_on_remove.append(bttx0)
        for bicv0 in btmp_prenuc_on_remove:       
            pre_nucleus_on_association2.remove(bicv0)
        
        for bxq19 in dimer_off_association2:
            if bindex in bxq19:
                btmp_dimer_off_association2.append(bxq19)
            elif bb in bxq19:
                btmp_dimer_off_association2.append(bxq19)
        for bxq11 in btmp_dimer_off_association2:
            dimer_off_association2.remove(bxq11)

        for bxq17 in fibril_association2:
            if bindex in bxq17:
                btmp_fibirl_association.append(bxq17)
            elif bb in bxq17:
                btmp_fibirl_association.append(bxq17)
        for bxq9 in btmp_fibirl_association:
            fibril_association2.remove(bxq9)
            
        for bxq16 in second_monomer_attach2:
            if bindex in bxq16:
                btmp_second_monomer_attach2.append(bxq16)
            elif bb in bxq16:
                btmp_second_monomer_attach2.append(bxq16)
        for bxq8 in btmp_second_monomer_attach2:
            second_monomer_attach2.remove(bxq8)
#        second_fibril_association2 = [bxq15 for bxq15 in second_fibril_association2 if not bindex in bxq15]
        for bxq15 in second_fibril_association2:
            if bindex in bxq15:
                btmp_second_fibril_association2.append(bxq15)
            elif bb in bxq15:
                btmp_second_fibril_association2.append(bxq15)
        for bxq7 in btmp_second_fibril_association2:
            second_fibril_association2.remove(bxq7)
#        second_pre_nucleus_association2 = [bxq14 for bxq14 in second_pre_nucleus_association2 if not bindex in bxq14]           
        for bxq14 in second_pre_nucleus_association2:
            if bindex in bxq14:
                btmp_second_pre_nucleus_association2.append(bxq14)
            elif bb in bxq14:
                btmp_second_pre_nucleus_association2.append(bxq14)                
        for bxq6 in btmp_second_pre_nucleus_association2:
            second_pre_nucleus_association2.remove(bxq6)
#        second_dimer_association2 = [bxq13 for bxq13 in second_dimer_association2 if not bindex in bxq13]            
        for bxq13 in second_dimer_association2:
            if bindex in bxq13:
                btmp_second_dimer_association2.append(bxq13)
            elif bb in bxq13:
                btmp_second_dimer_association2.append(bxq13)
        for bxq5 in btmp_second_dimer_association2:
            second_dimer_association2.remove(bxq5)
                                                                                                               
            
        if bindex in monos_inter_diffusion_off2:            
            monos_inter_diffusion_off2.remove(bindex)# = [x for x in mono_inter_diffusion_on1 if not index in x]
#        pre_nucleus_off_association2 = [bxq12 for bxq12 in pre_nucleus_off_association2 if not bindex in bxq12]    
        for bxq12 in pre_nucleus_off_association2:
            if bindex in bxq12:
                btmp_pre_nucleus_off_association2.append(bxq12)
            elif bb in bxq12:
                btmp_pre_nucleus_off_association2.append(bxq12)
        for bxq3 in btmp_pre_nucleus_off_association2:
            pre_nucleus_off_association2.remove(bxq3)
        
        if bb in monos_inter_diffusion_on2: # removing the second free monomer from previous reaction lists
            monos_inter_diffusion_on2.remove(bb)
   
        if bb in monos_inter_diffusion_off2:
            monos_inter_diffusion_off2.remove(bb)# = [bxq4 for bxq4 in monos_inter_diffusion_off2 if  bxq4 != bb in monos_inter_diffusion_off2]

# append the new possible interaction lists
        dimer_on_dissociation2.append(bindex) # creating the dimer_on_dissociation list
        dimer_on_dissociation2.append(bb)
        dimer_on_diffusion2.append([bindex,bb]) # used for diffusion and dissociation

        for bixi in btmp_on11:
            pre_nucleus_on_association2.append(bixi) # = pre_nucleus_on_association1 + tmp_on # update pre nucleus association list
#            print('adding a pre nuc from dimer end',pre_nucleus_on_association2)
        for bjxi in btmp_inter_range_on_mono11:
            monos_inter_diffusion_on2.append(bjxi) # = mono_inter_diffusion_on1 + tmp_inter_range_on_mono
#            print('test inter monomers range liist',monos_inter_diffusion_on2)
#        print('end result dimer on:', bindex, bb, H[bindex], H[bb])
            
    btmp_on11 = []
    btmp_inter_range_on_mono11 = []
    btmp_dimer_on_remove = []
    btmp_prenuc_on_remove = []
    btmp_fibirl_association = []
    btmp_dimer_off_association2 = []
    btmp_second_monomer_attach2 = []
    btmp_second_fibril_association2 = []
    btmp_second_pre_nucleus_association2 = []
    btmp_second_dimer_association2 = []
    btmp_pre_nucleus_off_association2 = []
#                
                
    return (H)
############################################ end dimer on asssoctaion ##########################################


################################# CREATES A off DIMER WITH PARTICLE ID AS order number and SLOPE ###############
def dimer_off(K,index_01,sigma,c, monomers4,mono_inter_diffusion_on4, mono_inter_diffusion_off4,
                    dimer_on_association4, dimer_off_association4, pre_nucleus_on_association4,
                    fibril_association4, second_monomer_attach4, second_fibril_association4, 
                    second_pre_nucleus_association4, second_dimer_association4, pre_nucleus_off_association4,
                    dimer_on_dissociation4, dimer_on_diffusion4,dimer_off_dissociation4,dimer_off_diffusion4,STOP_TEST4): # index_01 is the stationary monomer, c is the one moving to form the dimer
#    print("dimer off", index_01,c,K[index_01],K[c])
#    print("dimer off dissociation", dimer_off_dissociation4)
    c0 = K[index_01]
    ctmp_dimer_off = []
    ctmp_off_off = []
    ctest_dist_pass_4 = False
    ctmp_inter_range_off_mono = []
    
    ctmp_dimer_off_association4 = []
    ctmp_dimer_on_association4 = []
    ctmp_pre_nucleus_off_association4 = []
    ctmp_pre_nucleus_on_association4 = []
    ctmp_fibril_association4 = []
    
    ctmp_second_monomer_attach4 = []
    ctmp_second_fibril_association4 = []
    ctmp_second_pre_nucleus_association4 = []
    ctmp_second_dimer_association4 = []
    
# finding new coordinate for the moving monomer
    cslope_off = (c0[1]-K[c][1])/(c0[0]-K[c][0])
    cangle = math.atan(cslope_off)
    ctmp_dimer_off.append(c0[0] + 2.0*sigma*math.cos(cangle))
    ctmp_dimer_off.append(c0[1] + 2.0*sigma*math.sin(cangle))
# checking periodic boundary conditions
    if ctmp_dimer_off[0] > 0:            
        ctmp_dimer_off[0] = ctmp_dimer_off[0]%boxSide
    if ctmp_dimer_off[0] < 0:
        ctmp_dimer_off[0] = boxSide - abs(ctmp_dimer_off[0])%boxSide
    if ctmp_dimer_off[1] > 0:
        ctmp_dimer_off[1] = ctmp_dimer_off[1]%boxSide
    if ctmp_dimer_off[1] < 0:
        ctmp_dimer_off[1] = boxSide - abs(ctmp_dimer_off[1])%boxSide
# checking for overlap with over particles and monomer interaction partners for the new position
    for cai in range(len(K)):
        if (cai != c) and (cai != index_01):
            cdist_tmp_off = dist(ctmp_dimer_off,K[cai])
            cdist_tmp_off_1 = dist(c0,K[cai])           # looking for new monoes to add to the dimer with respect to the center
            if cdist_tmp_off < round(2.0*sigma,13):
                ctest_dist_pass_4 = True
                STOP_TEST4.append(0)
#                print("NOT PASSED")
#                break          
                return (K)
            elif (K[cai][2] == 0) and (cdist_tmp_off_1 >= 2.0*sigma) and (cdist_tmp_off_1 < inter_range_off):
                ctmp_off_off.append([cai,index_01])
                ctmp_inter_range_off_mono.append(cai)
# changing old coordinate for new coordinaties and update relevant population/reaction lists
    if ctest_dist_pass_4 == False:
        STOP_TEST4.append(1)
#        print("PASSED")
# remove from interaction lists and population lists        
        monomers4.remove(index_01) # = [x for x in monomers1 if not index_01 in x]
        monomers4.remove(c)# = [x for x in monomers1 if not c in x]
        if index_01 in mono_inter_diffusion_on4: # removing the first free monomer from previous reaction lists
#            mono_inter_diffusion_on1.remove(index_01)
            mono_inter_diffusion_on4.remove(index_01)
        
        # change coordinates with new ones
        c0[2] = 20                   # center of the off aggregate
        ctmp_dimer_off.append(21)
        K[index_01][3] = cslope_off
        ctmp_dimer_off.append(cslope_off)
        K[c][:] = ctmp_dimer_off[:]
        
        for cxac in dimer_on_association4:
            if index_01 in cxac:
                ctmp_dimer_on_association4.append(cxac)
            elif c in cxac:
                ctmp_dimer_on_association4.append(cxac)
        for cxad in ctmp_dimer_on_association4:
            dimer_on_association4.remove(cxad)
        
        for cxa19 in dimer_off_association4:
            if index_01 in cxa19:
                ctmp_dimer_off_association4.append(cxa19)
            elif c in cxa19:
                ctmp_dimer_off_association4.append(cxa19)
        for cxa10 in ctmp_dimer_off_association4:
            dimer_off_association4.remove(cxa10)
            
        
        for cxa18 in pre_nucleus_on_association4:
            if index_01 in cxa18:
                ctmp_pre_nucleus_on_association4.append(cxa18)
            elif c in cxa18:
                ctmp_pre_nucleus_on_association4.append(cxa18)
        for cxa9 in ctmp_pre_nucleus_on_association4:
            pre_nucleus_on_association4.remove(cxa9)
        
        for cxa17 in fibril_association4:
            if index_01 in cxa17:
                ctmp_fibril_association4.append(cxa17)
            elif c in cxa17:
                ctmp_fibril_association4.append(cxa17)
        for cxa8 in ctmp_fibril_association4:
            fibril_association4.remove(cxa8)
        
        for cxa16 in second_monomer_attach4:
            if index_01 in cxa16:
                ctmp_second_monomer_attach4.append(cxa16)
            elif c in cxa16:
                ctmp_second_monomer_attach4.append(cxa16)
        for cxa7 in ctmp_second_monomer_attach4:
            second_monomer_attach4.remove(cxa7)
            
        for cxa15 in second_fibril_association4:
            if index_01 in cxa15:
                ctmp_second_fibril_association4.append(cxa15)
            elif c in cxa15:
                ctmp_second_fibril_association4.append(cxa15)
        for cxa6 in ctmp_second_fibril_association4:
            second_fibril_association4.remove(cxa6)
            
        for cxa14 in second_pre_nucleus_association4:
            if index_01 in cxa14:
                ctmp_second_pre_nucleus_association4.append(cxa14)
            elif c in cxa14:
                ctmp_second_pre_nucleus_association4.append(cxa14)
        for cxa5 in ctmp_second_pre_nucleus_association4:
            second_pre_nucleus_association4.remove(cxa5)
        
        for cxa13 in second_dimer_association4:
            if index_01 in cxa13:
                ctmp_second_dimer_association4.append(cxa13)
            elif c in cxa13:
                ctmp_second_dimer_association4.append(cxa13)
        for cxa4 in ctmp_second_dimer_association4:
            second_dimer_association4.remove(cxa4)
        
#            mono_inter_diffusion_on1 = [x for x in mono_inter_diffusion_on1 if not index_01 in x]
        if index_01 in mono_inter_diffusion_off4:
            mono_inter_diffusion_off4.remove(index_01)
#            mono_inter_diffusion_off1 = [x for x in mono_inter_diffusion_on1 if not index_01 in x]
        
        for cxa12 in pre_nucleus_off_association4:
            if index_01 in cxa12:
                ctmp_pre_nucleus_off_association4.append(cxa12)
            elif c in cxa12:
                ctmp_pre_nucleus_off_association4.append(cxa12)
        for cxa3 in ctmp_pre_nucleus_off_association4:
            pre_nucleus_off_association4.remove(cxa3)
#        pre_nucleus_off_association4 = [cxa12 for cxa12 in pre_nucleus_off_association4 if not index_01 in cxa12]

        if c in mono_inter_diffusion_on4: # removing the second free monomer from previous reaction lists
            mono_inter_diffusion_on4.remove(c)

#            mono_inter_diffusion_on1 = [x for x in mono_inter_diffusion_on1 if not c in x]
        if c in mono_inter_diffusion_off4:            
            mono_inter_diffusion_off4.remove(c)
            
        dimer_off_dissociation4.append(index_01)
        dimer_off_diffusion4.append([index_01, c]) # creating/modifying dimer off diffusion list


# add to relevant reaction and population lists
            
        for coo in ctmp_off_off:
            pre_nucleus_off_association4.append(coo) # = pre_nucleus_off_association1 + tmp_off_off# creating modifying pre_nucleaus_off_association list
        for cpp in ctmp_inter_range_off_mono:
            mono_inter_diffusion_off4.append(cpp)# = tmp_inter_range_off_mono + mono_inter_diffusion_off1    
    
    ctmp_dimer_off_association4 = []
    ctmp_dimer_on_association4 = []
    ctmp_pre_nucleus_off_association4 = []
    ctmp_pre_nucleus_on_association4 = []
    ctmp_fibril_association4 = []
    ctmp_second_monomer_attach4 = []
    ctmp_second_fibril_association4 = []
    ctmp_second_pre_nucleus_association4 = []
    ctmp_second_dimer_association4 = []
    
    return (K)
############################################ end dimer off function #################################################
    

############################################# Pre nuceus on elongation low y end #########################################
#elongates an on pre-nucleus by adding a monomer to the end with lowest y coordinate 
def nucleus_on_low_end(J,index_02,sigma,d,monomers5,monos_inter_diffusion_on5,
                    monos_inter_diffusion_off5,pre_nucleus_off_association5, second_dimer_association5,
                    dimer_on_association5, dimer_off_association5, pre_nucleus_on_association5,
                    fibril_association5, second_monomer_attach5, second_fibril_association5, 
                    second_pre_nucleus_association5, fibril_dissociation5,trimer_on_diffusion5,
                    dimer_on_dissociation5, dimer_on_diffusion5,pre_nucleus_on_dissociation5,
                    tetramer_on_diffusion5,pentamer_on_diffusion5,STOP_TEST5):   # h is the end monomer of dimer, index_02 is the monomer to be added
    dlength = []    
#    a = J[index_02]  
#    print('aggregate is low end', J[d], J[index_02])
    dslope = J[d][3]  
    dtest_dist_low1 = False
    dtest_dist_low2 = False
    dtmp23  = []  
    dtmp_low = []
    dtmp_inter_range_on_mono1 = []
    dtmp_dimer_on1_remove = []
    dtmp_pre_on3_remove = []
    ddimer_diff_remove_low = []
    dtmp_dimer_on_dissociation = []
    dtmp_pre_nucleus_on_dissociation = []
    dtmp_fibril_association = []
    dtmp_dimer_off_association5 = []
    dtmp_trimer_on_diffusion5 = []
    dtmp_tetramer_on_diffusion5 = []
    dtmp_second_monomer_attach5 = []
    dtmp_second_fibril_association5 = []
    dtmp_second_pre_nucleus_association5 = []
    dtmp_second_dimer_association5 = []
    dtmp_pre_nucleus_off_association5 = []
    dtmp_pre_nucleus_on_dissociation5 = []
    dtmp_second_mono_attach5 = []
#find new position for the added monomer
    if dslope < 0:        
        dtmp23.append(J[d][0] - 2.0*sigma*math.cos(math.pi - math.atan(dslope))) # d is the end of the pre nucleus
        dtmp23.append(J[d][1] + 2.0*sigma*math.sin(math.pi - math.atan(dslope))) # index_02 is the moving monomer

    elif dslope > 0:
        dtmp23.append(J[d][0] + 2.0*sigma*math.cos(math.pi - math.atan(dslope)))
        dtmp23.append(J[d][1] - 2.0*sigma*math.sin(math.pi - math.atan(dslope))) 

# check for periodic boundary conditions
    if dtmp23[0] > 0:            
        dtmp23[0] = dtmp23[0]%boxSide
    if dtmp23[0] < 0:
        dtmp23[0] = boxSide - abs(dtmp23[0])%boxSide
    if dtmp23[1] > 0:
        dtmp23[1] = dtmp23[1]%boxSide
    if dtmp23[1] < 0:
        dtmp23[1] = boxSide - abs(dtmp23[1])%boxSide     
#check for overlap and interaction partners of monomer in new position
    for dfrt3 in range(len(J)):
        if J[dfrt3][3] == dslope:
#            print(J[dfrt3])
            dlength.append(dfrt3)
#    dpro_length = [(ind, J[ind].index(dslope)) for ind in range(len(J)) if dslope in J[ind]]   
#    print('the other end in low1', dpro_length)
#    for drr1 in dpro_length:
#        dlength.append(drr1[0]) # the entire aggregate
#    length = [list(element) for element in pro_length]
    dthe_other_end_low1 = [elem for elem in dlength if 101 in J[elem]]
#    print('the other_end_low_list', dthe_other_end_low1)
    dthe_other_end_low = dthe_other_end_low1[0]
#    print('the other end low', dthe_other_end_low)
#    print('the_other_end_low', dthe_other_end_low, J[dthe_other_end_low])
    for dids in range(len(J)):
        if dids != index_02:
            if J[dids][3]  != dslope:
                if dids != dthe_other_end_low:
                    dtest_dist_nucleus_on_low1 = round(dist(dtmp23, J[dids]),13) #distance bewteen the new position of index_02 and other particles
                    dtest_dist_nucleus_on_low2 = round(dist(J[dthe_other_end_low], J[dids]),13) # distanc ebetween the other end and other particles
                    if dtest_dist_nucleus_on_low1 < round(2.0*sigma,13):
                        dtest_dist_low1 = True
                        STOP_TEST5.append(0)
#                        print("NOT PASSED")
 #                       print('didn not pass dist test inlow', dtest_dist_nucleus_on_low1,dtest_dist_nucleus_on_low2)
#                        break
                        return (J)
                    elif dtest_dist_nucleus_on_low2 <= round(2.0 * sigma,13):
                        dtest_dist_low1 = True
                        STOP_TEST5.append(0)
#                        break
                        return (J)
                    elif (J[dids][2] == 0):
                        if dtest_dist_nucleus_on_low1 >= round(2.0*sigma,13):
                            if dtest_dist_nucleus_on_low1 <= round(inter_range_on):
                                dtmp_low.append([dids,index_02]) # monomer position 0, aggregate end, position 1, pre nucleus on potential pair
#                                print('added pre nucl from low', dtmp_low, J[dids], dtmp23)
                                dtmp_inter_range_on_mono1.append(dids)
                   
                        if dtest_dist_nucleus_on_low2 >= round(2.0*sigma,13):
                            if dtest_dist_nucleus_on_low2 <= round(inter_range_on):
                                dtmp_low.append([dids,dthe_other_end_low]) # monomer position 0, aggregate end, position 1, pre nucleus on potential pair
#                                print('added pre nucl from low', dtmp_low, J[dids], J[dthe_other_end_low])
                                dtmp_inter_range_on_mono1.append(dids)
                        

# changing old coordinates for the new ones and updating population/reaction lists
    if (dtest_dist_low1 == False) and (dtest_dist_low2 == False):  # checking for overlap   
        STOP_TEST5.append(1)
#        print("PASSED")
        J[d][2] = 12   
        dtmp23.append(102)
        dtmp23.append(dslope)
        J[index_02][:] = dtmp23[:]
#        dlength_old = dlength[:]
        dlength.append(index_02)

        monomers5[:] = [ddx for ddx in monomers5 if ddx != index_02] #monomers5.remove(index_02) # = [x for x in monomers1 if not index_02 in x]

        if index_02 in monos_inter_diffusion_on5: # removing the first free monomer from previous reaction lists
            monos_inter_diffusion_on5 = list(filter(lambda d101: (d101 != index_02) , monos_inter_diffusion_on5))
            
        if index_02 in monos_inter_diffusion_off5: 
            monos_inter_diffusion_off5.remove(index_02)# removing the free monomer from previous interation reaction lists
            
        for dttx1 in dimer_on_association5:
            if index_02 in dttx1: 
                dtmp_dimer_on1_remove.append(dttx1)            
        for dcv1 in dtmp_dimer_on1_remove:       
            dimer_on_association5.remove(dcv1)
            
        for dttx5 in pre_nucleus_on_association5:
            if (index_02 in dttx5):
                dtmp_pre_on3_remove.append(dttx5)
            elif (d in dttx5):
                dtmp_pre_on3_remove.append(dttx5) 
            elif dthe_other_end_low in dttx5:
                dtmp_pre_on3_remove.append(dttx5)                
        for dcv5 in dtmp_pre_on3_remove:       
            pre_nucleus_on_association5.remove(dcv5)

#        fibril_association5 = [dxf9 for dxf9 in fibril_association5 if not index_02 in dxf9]
        for dxf9 in fibril_association5:
            if index_02 in dxf9:
                dtmp_fibril_association.append(dxf9)
        for dxf99 in dtmp_fibril_association:
            fibril_association5.remove(dxf99)
            
        for dxf7 in second_monomer_attach5:
            if index_02 in dxf7:
                dtmp_second_monomer_attach5.append(dxf7)
        for dxf77 in dtmp_second_monomer_attach5:
            second_monomer_attach5.remove(dxf77)
#        second_fibril_association5 = [dxf6 for dxf6 in second_fibril_association5 if not index_02 in dxf6]
        for dxf6 in second_fibril_association5:
            if index_02 in dxf6:
                dtmp_second_fibril_association5.append(dxf6)
        for dxf66 in dtmp_second_fibril_association5:
            second_fibril_association5.remove(dxf66)
#        second_pre_nucleus_association5 = [dxf5 for dxf5 in second_pre_nucleus_association5 if not index_02 in dxf5]           
        for dxf5 in second_pre_nucleus_association5:
            if index_02 in dxf5:
                dtmp_second_pre_nucleus_association5.append(dxf5)
        for dxf55 in dtmp_second_pre_nucleus_association5:
            second_pre_nucleus_association5.remove(dxf55)
#        second_dimer_association5 = [dxf4 for dxf4 in second_dimer_association5 if not index_02 in dxf4]            
        for dxf4 in second_dimer_association5:
            if index_02 in dxf4:
                dtmp_second_dimer_association5.append(dxf4)
        for dxf44 in dtmp_second_dimer_association5:
            second_dimer_association5.remove(dxf44)
             
        for dxf13 in dimer_off_association5:
            if index_02 in dxf13: 
                dtmp_dimer_off_association5.append(dxf13)
            elif d in dxf13:
                dtmp_dimer_off_association5.append(dxf13)
        for dxf133 in dtmp_dimer_off_association5:
            dimer_off_association5.remove(dxf133)            
            
#        pre_nucleus_off_association5 = [dxf3 for dxf3 in pre_nucleus_off_association5 if not index_02 in dxf3]
        for dxf3 in pre_nucleus_off_association5:
            if index_02 in dxf3:
                dtmp_pre_nucleus_off_association5.append(dxf3)
            elif d in dxf3:
                dtmp_pre_nucleus_off_association5.append(dxf3)
        for dxf33 in dtmp_pre_nucleus_off_association5:
            pre_nucleus_off_association5.remove(dxf33)


        if len(dlength) == 3:
#            print(dlength)
#            print('dimer on diffusion', dimer_on_diffusion)
            for dd12 in dimer_on_diffusion5:
                if  (d in dd12) or (dthe_other_end_low in dd12):
                    ddimer_diff_remove_low.append(dd12)
            for dh15 in ddimer_diff_remove_low:
                dimer_on_diffusion5.remove(dh15)
#                print('removed dimer diffusion from low', dh15)

            for ddg1 in dimer_on_dissociation5:
                if (ddg1 == d): 
                    dtmp_dimer_on_dissociation.append(ddg1)
                elif (ddg1 == dthe_other_end_low):
                    dtmp_dimer_on_dissociation.append(ddg1)
            for ddg11 in dtmp_dimer_on_dissociation:
                    dimer_on_dissociation5.remove(ddg11)
#                    print('removed dimer_on_dissociation in low', ddg1)
            trimer_on_diffusion5.append(dlength) # creating/modifying trimer on diffusion
#            dimer_on_dissociation5.remove(dthe_other_end_low)     
#            dimer_on_dissociation5.remove(d)
            for daa2 in dtmp_low:
                pre_nucleus_on_association5.append(daa2) # = pre_nucleus_on_association1 + tmp_low # adding it with new partners to the pre_nucleus_on list     
#                print('pre nuc added low', daa2)
            pre_nucleus_on_dissociation5.append(index_02) # creating/modifying the pre_nucleus_on_dissociation list   
            pre_nucleus_on_dissociation5.append(dthe_other_end_low)
            for dss2 in dtmp_inter_range_on_mono1:
                monos_inter_diffusion_on5.append(dss2) # = monos_inter_diffusion_on1 + tmp_inter_range_on_mono
                
        elif len(dlength) == 4:
            for dxf24 in trimer_on_diffusion5:
                if (d in dxf24) or (dthe_other_end_low in dxf24):
                    dtmp_trimer_on_diffusion5.append(dxf24)
            for dxf244 in dtmp_trimer_on_diffusion5:
                trimer_on_diffusion5.remove(dxf244)
#            trimer_on_diffusion5 = [dxf24 for dxf24 in trimer_on_diffusion5 if not d in dxf24] 
#            trimer_on_diffusion5 = [dxf27 for dxf27 in trimer_on_diffusion5 if not dthe_other_end_low in dxf27] 
            tetramer_on_diffusion5.append(dlength) #creating/modifying tetramer on diffusion list
            for dvty1 in pre_nucleus_on_dissociation5:
                if (dvty1 == d):
                    dtmp_pre_nucleus_on_dissociation.append(dvty1)      
                elif (dvty1 == dthe_other_end_low):
                    dtmp_pre_nucleus_on_dissociation.append(dvty1)      
            for dvty11 in dtmp_pre_nucleus_on_dissociation:
                pre_nucleus_on_dissociation5.remove(dvty11)                
#                print('removing a pre on dissociation from low', dvty11)
                    
            pre_nucleus_on_dissociation5.append(index_02) # creating/modifying the pre_nucleus_on_dissociation list          
            pre_nucleus_on_dissociation5.append(dthe_other_end_low)
            for daa1 in dtmp_low:
                pre_nucleus_on_association5.append(daa1) # = pre_nucleus_on_association1 + tmp_low # adding it with new partners to the pre_nucleus_on list    
#                print('pre nuc added, low', aa1)
                
            for dss1 in dtmp_inter_range_on_mono1:
                monos_inter_diffusion_on5.append(dss1) # = monos_inter_diffusion_on1 + tmp_inter_range_on_mono
        elif len(dlength) == 5:
            for ds in dlength:
                if J[ds][2] == 102:
                    J[ds][2] = 122    # changing the particle ID as lower y end of fibril, not pre-nucleus
                elif J[ds][2] == 12:
                    J[ds][2] = 1
                elif J[ds][2] == 101:
                    J[ds][2] = 121   # changing the particle ID as higher y end of fibril, not pre-nucleus
            for dxf1 in tetramer_on_diffusion5:
                if (d in dxf1) or (dthe_other_end_low in dxf1):
                    dtmp_tetramer_on_diffusion5.append(dxf1)
            for dxf11 in dtmp_tetramer_on_diffusion5:
                tetramer_on_diffusion5.remove(dxf11)
                
            for dloq1 in pre_nucleus_on_dissociation5:
                if (dloq1 == d): 
                    dtmp_pre_nucleus_on_dissociation5.append(dloq1)
                elif (dloq1 == dthe_other_end_low):
                    dtmp_pre_nucleus_on_dissociation5.append(dloq1)
            for dloq11 in dtmp_pre_nucleus_on_dissociation5:
                pre_nucleus_on_dissociation5.remove(dloq11)
                
            for ddtr in range(len(J)):
                if J[ddtr][2] == 0:
                    for ddtw in dlength:
                        if J[ddtw][2] == 1:
                            dtest_dist_sec = round(dist(J[ddtw], J[ddtr]),13)
                            if dtest_dist_sec > round(2.0*sigma,13) and dtest_dist_sec <= round(inter_range_on):
                                dtmp_second_mono_attach5.append([ddtr,ddtw])
                                
                        
#                print('removing a pre on  dissociation from low', dloq11)
            for dnju in dtmp_low:
                fibril_association5.append(dnju) #creating/modifying fibirl association list
            pentamer_on_diffusion5.append(dlength) # creating/modifying fibril nucleus diffusion list
            fibril_dissociation5.append(index_02) # cfeating/modifying fibril dissociation list
            fibril_dissociation5.append(dthe_other_end_low)
            for dd1 in dtmp_inter_range_on_mono1:
                monos_inter_diffusion_on5.append(dd1) # = monos_inter_diffusion_on1 + tmp_inter_range_on_mono
            
            for ddfu in dtmp_second_mono_attach5:
                second_monomer_attach5.append(ddfu)
#            pre_nucleus_on_dissociation5.remove(dthe_other_end_low)
#        for dfri in dlength:            
#            print('end resul pre on low:',dfri, J[dfri] )
                
    dtmp23  = []  
    dtmp_low = []
    dtmp_inter_range_on_mono1 = []
    dtmp_dimer_on1_remove = []
    dtmp_pre_on3_remove = []
    ddimer_diff_remove_low = []
    dtmp_dimer_on_dissociation = []
    dtmp_pre_nucleus_on_dissociation = []
    dtmp_pre_nucleus_on_dissociation5 = []
    dtmp_fibril_association = []
    dtmp_dimer_off_association5 = []
    dtmp_trimer_on_diffusion5 = []
    dtmp_tetramer_on_diffusion5 = []
    dtmp_second_monomer_attach5 = []
    dtmp_second_fibril_association5 = []
    dtmp_second_pre_nucleus_association5 = []
    dtmp_second_dimer_association5 = []
    dtmp_pre_nucleus_off_association5 = []
    dtmp_second_mono_attach5 = []
    return (J)
############################################ end pre nucleus on elongation #########################################
    

#############################################  pre nucleus elongation high end #######################################
#elongates an on pre-nucleus (with PARTICLE ID AS END and FIBRIL ID AS SLOPE) by adding a monomer\
# to the end with highest y coordinate 
def nucleus_on_high_end(D,index_03,sigma,e, monomers6,monos_inter_diffusion_on6,
                    monos_inter_diffusion_off6,pre_nucleus_off_association6, second_dimer_association6,
                    dimer_on_association6, dimer_off_association6, pre_nucleus_on_association6,
                    fibril_association6, second_monomer_attach6, second_fibril_association6, 
                    second_pre_nucleus_association6, fibril_dissociation6,trimer_on_diffusion6,
                    dimer_on_dissociation6, dimer_on_diffusion6,pre_nucleus_on_dissociation6,
                    tetramer_on_diffusion6,pentamer_on_diffusion6,STOP_TEST6):     
    elength_1 = []    
    eslope_1 = D[e][3]     # is the aggregate end, stationary monomer, index_03 is moving
#    print('aggregate is high end', D[e], D[index_03])
    etest_dist_high1 = False
    etest_dist_high2 = False
    etmp_1  = []  
    etmp_high = []
    etmp_inter_range_on_mono = []
    etmp_dimer_on2_remove = []
    etmp_pre_on2_remove = []
    edimer_diff_remove_high = []
    etmp_fibril_association = []
    etmp_dimer_off_association6 =[]
    etmp_trimer_on_diffusion6 = []
    ettmp_dimer_on_dissociation = []
    etmp_pre_nucleus_on_dissociation6 = []
    etmp1_pre_nucleus_on_dissociation6 = []
    etmp_tetramer_on_diffusion6 = []
    etmp_second_monomer_attach6 = []
    etmp_second_fibril_association6 = []
    etmp_second_pre_nucleus_association6 = []
    etmp_second_dimer_association6 = []
    etmp_pre_nucleus_off_association6 = []
    etmp_second_mono_attach6_1 = []


#finding new coordinates of index_03
    if eslope_1 < 0:        
        etmp_1.append(D[e][0] + 2.0*sigma*math.cos(math.pi - math.atan(eslope_1)))
        etmp_1.append(D[e][1] + 2.0*sigma*math.sin(math.pi + math.atan(eslope_1)))
    elif eslope_1 > 0:
        etmp_1.append(D[e][0] - 2.0*sigma*math.cos(math.pi + math.atan(eslope_1)))
        etmp_1.append(D[e][1] + 2.0*sigma*math.sin(math.pi - math.atan(eslope_1))) 
# check for periodic boundary conditions        
    if etmp_1[0] > 0:            
        etmp_1[0] = etmp_1[0]%boxSide
    if etmp_1[0] < 0:
        etmp_1[0] = boxSide - abs(etmp_1[0])%boxSide
    if etmp_1[1] > 0:
        etmp_1[1] = etmp_1[1]%boxSide
    if etmp_1[1] < 0:
        etmp_1[1] = boxSide - abs(etmp_1[1])%boxSide       
     
    for ert5 in range(len(D)):
        if D[ert5][3] == eslope_1:
            elength_1.append(ert5)
#    length_1 = [element1 for element1 in pro_length_1]
    ethe_other_end_high1 = [elemn for elemn in elength_1 if 102 in D[elemn]]
#    print('the other_end_low_list', ethe_other_end_high1)
    ethe_other_end_high = ethe_other_end_high1[0]
#    print('the other end low', ethe_other_end_high)
#    print('the_other_end_low', ethe_other_end_high, D[ethe_other_end_high]) 
        
#check for overlap and interaction partners of monomer in new position
    for eixi in range(len(D)):
        if eixi != index_03:
            if D[eixi][3] != eslope_1: 
                if eixi != ethe_other_end_high:            
                    etest_dist_nucleus_on_high1 = round(dist(etmp_1, D[eixi]),13)
                    etest_dist_nucleus_on_high2 = round(dist(D[ethe_other_end_high], D[eixi]),13)
                    if (etest_dist_nucleus_on_high1 <= round(2.0*sigma,13)): 
                        etest_dist_high1 = True
                        STOP_TEST6.append(0)
#                        print("NOT PASSED")
#                        break
                        return (D)            
                    elif (etest_dist_nucleus_on_high2 <= round(2.0*sigma,13)):
                        etest_dist_high2 = True
                        STOP_TEST6.append(0)
#                        print("NOT PASSED")
#                        break
                        return (D) 
                                                                    
                    elif (D[eixi][2] == 0) and (eixi != index_03):
                            if etest_dist_nucleus_on_high1 >= round(2.0*sigma,13):
                                if etest_dist_nucleus_on_high1 <= round(inter_range_on):
                                    etmp_high.append([eixi,index_03])
                                    etmp_inter_range_on_mono.append(eixi)
                           
                                if etest_dist_nucleus_on_high2 >= round(2.0*sigma,13):
                                    if etest_dist_nucleus_on_high2 <= round(inter_range_on):
                                        etmp_high.append([eixi,ethe_other_end_high])
                                        etmp_inter_range_on_mono.append(eixi)

# changing old coordinates for the new ones and updating population/reaction lists
    if (etest_dist_high1 == False) and (etest_dist_high2 == False):  # checking for overlap   
        STOP_TEST6.append(1)
#        print("PASSED")
        D[e][2] = 12   
        etmp_1.append(101)
        etmp_1.append(eslope_1)
        D[index_03][:] = etmp_1[:]
        elength_1.append(index_03)
        
        monomers6[:] = [ex for ex in monomers6 if ex != index_03] #monomers6.remove(index_03)# = [x for x in monomers1 if not index_03 in x]
        
        if index_03 in monos_inter_diffusion_on6: 
            monos_inter_diffusion_on6.remove(index_03)# removing the free monomer from previous interation reaction lists
            
        if index_03 in monos_inter_diffusion_off6: 
            monos_inter_diffusion_off6.remove(index_03)# removing the free monomer from previous interation reaction lists
            
        for ettx2 in dimer_on_association6:
            if index_03 in ettx2:
#                print('remoing a dimer on', ittx2)
                etmp_dimer_on2_remove.append(ettx2)               
        for ecv2 in etmp_dimer_on2_remove:       
            dimer_on_association6.remove(ecv2) # removing from dimer on association
#            print(icv2)
        

        for ettx4 in pre_nucleus_on_association6:
            if (index_03 in ettx4):
#                print('remoing a prenuc on', ittx4)
                etmp_pre_on2_remove.append(ettx4) 
            elif e in ettx4:
#                print('remoing a prenuc on', ittx4)
                etmp_pre_on2_remove.append(ettx4) 
            elif ethe_other_end_high in ettx4:
#                print('remoing a prenuc on', ittx4)
                etmp_pre_on2_remove.append(ettx4)                
        for ecv4 in etmp_pre_on2_remove:       
            pre_nucleus_on_association6.remove(ecv4)
#            print('removing a pre on association high', ecv4)
            
        for ettx5 in dimer_off_association6:
            if index_03 in ettx5:
                etmp_dimer_off_association6.append(ettx5)
            elif e in ettx5:
                etmp_dimer_off_association6.append(ettx5)
        for ettx55 in etmp_dimer_off_association6:
            dimer_off_association6.remove(ettx55)
#
#        fibril_association6 = [xe10 for xe10 in fibril_association6 if not index_03 in xe10]
        for exe10 in fibril_association6:
            if index_03 in exe10:
                etmp_fibril_association.append(exe10)
        for exe100 in etmp_fibril_association:
            fibril_association6.remove(exe100)
            
#        second_monomer_attach6 = [exe9 for exe9 in second_monomer_attach6 if not index_03 in exe9]
        for exe9 in second_monomer_attach6:
            if index_03 in exe9:
                etmp_second_monomer_attach6.append(exe9)
        for exe99 in etmp_second_monomer_attach6:
            second_monomer_attach6.remove(exe99)
#        second_fibril_association6 = [exe8 for exe8 in second_fibril_association6 if not index_03 in exe8]
        for exe8 in second_fibril_association6:
            if index_03 in exe8:
                etmp_second_fibril_association6.append(exe8)
        for exe88 in etmp_second_fibril_association6:
            second_fibril_association6.remove(exe88)
#        second_pre_nucleus_association6 = [exe6 for exe6 in second_pre_nucleus_association6 if not index_03 in exe6]           
        for exe6 in second_pre_nucleus_association6:
            if index_03 in exe6:
                etmp_second_pre_nucleus_association6.append(exe6)
        for exe66 in etmp_second_pre_nucleus_association6:
            second_pre_nucleus_association6.remove(exe66)
#        second_dimer_association6 = [exe5 for exe5 in second_dimer_association6 if not index_03 in exe5]            
        for exe5 in second_dimer_association6:
            if index_03 in exe5:
                etmp_second_dimer_association6.append(exe5)
        for exe55 in etmp_second_dimer_association6:
            second_dimer_association6.remove(exe55)
#        pre_nucleus_off_association6 = [exe4 for exe4 in pre_nucleus_off_association6 if not index_03 in exe4]            
        for exe4 in pre_nucleus_off_association6:
            if index_03 in exe4:
                etmp_pre_nucleus_off_association6.append(exe4)
            elif e in exe4:
                etmp_pre_nucleus_off_association6.append(exe4)
        for exe44 in etmp_pre_nucleus_off_association6:
            pre_nucleus_off_association6.remove(exe44)

 # removing/adding intercting partners from relevant lists
        if len(elength_1) == 3:
#            print(elength_1)
#            print('dimer on diffusion', dimer_on_diffusion)
            for efd1 in dimer_on_diffusion6:
                if  (e in efd1) or (ethe_other_end_high in efd1):
                    edimer_diff_remove_high.append(efd1)                    
            for egh1 in edimer_diff_remove_high:
                dimer_on_diffusion6.remove(egh1)
#                print('removing a dime diffusion in high', egh1)

            for eas1 in dimer_on_dissociation6:
                if (eas1 == e): 
                    ettmp_dimer_on_dissociation.append(eas1)
                elif (eas1 == ethe_other_end_high):
                    ettmp_dimer_on_dissociation.append(eas1)
            for eas11 in ettmp_dimer_on_dissociation:
                dimer_on_dissociation6.remove(eas11)
#                print('removing a  dimer dissociation in high', eas11)
#            dimer_on_dissociation6.remove(ethe_other_end_high)     
#            dimer_on_dissociation6.remove(e)
            pre_nucleus_on_dissociation6.append(index_03) # creating/modifying the pre_nucleus_on_dissociation list  
            pre_nucleus_on_dissociation6.append(ethe_other_end_high)
            for egg1 in etmp_inter_range_on_mono:
                monos_inter_diffusion_on6.append(egg1)# = monos_inter_diffusion_on1 + tmp_inter_range_on_mono
            for eff in etmp_high:
                pre_nucleus_on_association6.append(eff) # = pre_nucleus_on_association1 + tmp_high # adding it with new partners to the pre_nucleus_on list      
            trimer_on_diffusion6.append(elength_1) # creating/modifying trimer on diffusion
        elif len(elength_1) == 4:
            for exe2 in trimer_on_diffusion6:
                if (e in exe2) or (ethe_other_end_high in exe2):
                    etmp_trimer_on_diffusion6.append(exe2)
            for exe22 in etmp_trimer_on_diffusion6:
                trimer_on_diffusion6.remove(exe22)
#            trimer_on_diffusion6 = [exe2 for exe2 in dimer_on_diffusion6 if not e in exe2] 
#            trimer_on_diffusion6 = [exe22 for exe22 in dimer_on_diffusion6 if not ethe_other_end_high in exe22]
            for euyt1 in pre_nucleus_on_dissociation6:
                if (euyt1 == e): 
                    etmp1_pre_nucleus_on_dissociation6.append(euyt1)
                elif (euyt1 == ethe_other_end_high):
                    etmp1_pre_nucleus_on_dissociation6.append(euyt1)
            for euyt11 in etmp1_pre_nucleus_on_dissociation6:
                pre_nucleus_on_dissociation6.remove(euyt11)
#            pre_nucleus_on_dissociation6.remove(e)
            for egg2 in etmp_inter_range_on_mono:
                monos_inter_diffusion_on6.append(egg2)# = monos_inter_diffusion_on1 + tmp_inter_range_on_mono
            for eff1 in etmp_high:
                pre_nucleus_on_association6.append(eff1) # = pre_nucleus_on_association1 + tmp_high # adding it with new partners to the pre_nucleus_on list
            tetramer_on_diffusion6.append(elength_1) #creating/modifying tetramer on diffusion list                
            pre_nucleus_on_dissociation6.append(index_03) # creating/modifying the pre_nucleus_on_dissociation list  
            pre_nucleus_on_dissociation6.append(ethe_other_end_high) # creating/modifying the pre_nucleus_on_dissociation list  
                
        elif len(elength_1) == 5:
            for es in elength_1:
                if D[es][2] == 102:
                    D[es][2] = 122    # changing the particle ID as lower y end of fibril, not pre-nucleus
                elif D[es][2] == 12:
                    D[es][2] = 1
                elif D[es][2] == 101:
                    D[es][2] = 121   # changing the particle ID as higher y end of fibril, not pre-nucleus
            for exe1 in tetramer_on_diffusion6:
                if (e in exe1) or (ethe_other_end_high in exe1):
                    etmp_tetramer_on_diffusion6.append(exe1)
            for exe11 in etmp_tetramer_on_diffusion6:
                tetramer_on_diffusion6.remove(exe11)
#            tetramer_on_diffusion6 = [exe1 for exe1 in tetramer_on_diffusion6 if not e in exe1] #creating/modifying tetramer on diffusion list, add entire list
#            tetramer_on_diffusion6 = [exe11 for exe11 in tetramer_on_diffusion6 if not ethe_other_end_high in exe11] #creating/modifying tetramer on diffusion list, add entire list

            for elo21 in pre_nucleus_on_dissociation6:
                if (elo21 == e): 
                    etmp_pre_nucleus_on_dissociation6.append(elo21)
                elif (elo21 == ethe_other_end_high):
                    etmp_pre_nucleus_on_dissociation6.append(elo21)
            for elo211 in etmp_pre_nucleus_on_dissociation6:
                pre_nucleus_on_dissociation6.remove(elo211)
                
            for edtr in range(len(D)):
                if D[edtr][2] == 0:
                    for edtw in elength_1:
                        if D[edtw][2] == 1:
                            etest_dist_sec = round(dist(D[edtw], D[edtr]),13)
                            if etest_dist_sec > round(2.0*sigma,13) and etest_dist_sec <= round(inter_range_on):
                                etmp_second_mono_attach6_1.append([edtr,edtw])
                
            for enju in etmp_high:
                fibril_association6.append(enju) #creating/modifying fibirl association list
                
            pentamer_on_diffusion6.append(elength_1) # creating/modifying fibril nucleus diffusion list
            fibril_dissociation6.append(index_03) # cfeating/modifying fibril dissociation list
            fibril_dissociation6.append(ethe_other_end_high)
            for ed1 in etmp_inter_range_on_mono:
                monos_inter_diffusion_on6.append(ed1) # = monos_inter_diffusion_on1 + tmp_inter_range_on_mono

            for ehh in etmp_inter_range_on_mono:
                monos_inter_diffusion_on6.append(ehh) # = monos_inter_diffusion_on1 + tmp_inter_range_on_mono
                
            for effr in etmp_second_mono_attach6_1:
                second_monomer_attach6.append(effr)
            
                            
#        for ebhy in elength_1:
#            print('end resul pre on high end:', ebhy, D[ebhy])
            
    etmp_1  = []  
    etmp_high = []
    etmp_inter_range_on_mono = []
    etmp_dimer_on2_remove = []
    etmp_pre_on2_remove = []
    edimer_diff_remove_high = []
    etmp_fibril_association = []
    etmp_dimer_off_association6 =[]
    etmp_trimer_on_diffusion6 = []
    ettmp_dimer_on_dissociation = []
    etmp_pre_nucleus_on_dissociation6 = []
    etmp1_pre_nucleus_on_dissociation6 = []
    etmp_tetramer_on_diffusion6 = []
    etmp_second_monomer_attach6 = []
    etmp_second_fibril_association6 = []
    etmp_second_pre_nucleus_association6 = []
    etmp_second_dimer_association6 = []
    etmp_pre_nucleus_off_association6 = []
    etmp_second_mono_attach6_1 = []
            
                
    return (D)
############################################ end prenucleus high end elongation ##########################################
    
    
############################################# nucleus off growth ######################################################
#elongates an off pre-nucleus (with PARTICLE ID AS END and aggr. ID AS SLOPE) by adding a monomer around it
def nucleus_off(E,index_04,sigma,f,m, monomers7,monos_inter_diffusion_on7, monos_inter_diffusion_off7,
                    dimer_on_association7, dimer_off_association7, pre_nucleus_on_association7,
                    fibril_association7, second_monomer_attach7, second_fibril_association7, 
                    second_pre_nucleus_association7, second_dimer_association7, pre_nucleus_off_association7,
                    dimer_on_dissociation7, dimer_off_diffusion7,pre_nucleus_off_dissociation7,
                    tetramer_off_diffusion7,pentamer_off_diffusion7,fibril_dissociation7,
                    pre_nucleus_on_dissociation7, trimer_off_diffusion7,dimer_off_dissociation7,nucleus_off_numb7,STOP_TEST7):     # cc is the aggregate end, index_04 is the moving monomer
#    print("off association routine", index_04, E[index_04],f, E[f])
#    print("dimer_off_dissociation7",dimer_off_dissociation7)
    n_size = m
    fslope_2 = E[f][3]
    fangle_2 = math.atan(fslope_2)
    flength_2 = [] # list of indices of all monomers in the aggregtae before adding the new one
#    ftest_dist_2 = [] # coordinates of monomers form the same aggregate
    ftmp_nucleus_off = [] # new coordinates of moving monomer
    ftest_dist_off = False # check the overlap
    ftmp_off_2 = [] # to hold found inetractiong monomers
    ftmp_inter_range_off_mono = []
    ftmp_dimer_on_association7 = []
    ftmp_dimer_off_association7 = []
    ftmp_pre_nucleus_on_association7 = []
    ftmp_pre_nucleus_off_association7 = []
    ftmp_fibril_association7 = []
    ftmp_second_monomer_attach7 = []
    ftmp_second_pre_nucleus_association7 = []
    ftmp_second_dimer_association7 = []
    ftmp_dimer_off_diffusion7 = []
    ftmp_trimer_off_diffusion7 = []
    ftmp_tetramer_off_diffusion7 = []
    ftmp_pentamer_off_diffusion7 = []
    fpre_nucleus_off_dissociation7 = []
#    ftmp_pre_nucleus_off_dissociation7 = []
    ftmp_dimer_off_dissociation7 = []
# find all monomers in the aggregate, by position number in L
    for fi in range(N):
        if E[fi][3] == fslope_2:
            flength_2.append(fi) # the entire pre nucleus off
#            if E[fi][2] == 20:            
#                center = fi
#            ftest_dist_2.append(fi) # list of actual particle information, not just position number in L
#            print("pre nucleus off in routine",fi,E[fi])
#    print("")
    

# find new coordinates of the moving monomer
    if len(flength_2) < n_size:
        ftmp_nucleus_off.append(E[f][0] + 2.0*sigma*math.cos(fangle_2 + (2.0*math.pi)/(n_size -1 )*(len(flength_2)-1)))
        ftmp_nucleus_off.append(E[f][1] + 2.0*sigma*math.sin(fangle_2 + (2.0*math.pi)/(n_size -1 )*(len(flength_2)-1)))
# check periodic boundary conditions
        if ftmp_nucleus_off[0] > 0:            
            ftmp_nucleus_off[0] = ftmp_nucleus_off[0]%boxSide
        if ftmp_nucleus_off[0] < 0:
            ftmp_nucleus_off[0] = boxSide - abs(ftmp_nucleus_off[0])%boxSide
        if ftmp_nucleus_off[1] > 0:
            ftmp_nucleus_off[1] = ftmp_nucleus_off[1]%boxSide
        if ftmp_nucleus_off[1] < 0:
            ftmp_nucleus_off[1] = boxSide - abs(ftmp_nucleus_off[1])%boxSide 
# checking for overalp and interacting monomers for the new position            
        for fij in range(len(E)):
            if (fij not in flength_2) and (fij != index_04):
                ftest_dist_nucleus_off = round(dist(ftmp_nucleus_off, E[fij]),13)
                ftest_dist_nucleus_off_1 = round(dist(E[f], E[fij]),13)
                if ftest_dist_nucleus_off < round(2.0*sigma,13):
                    ftest_dist_off = True
                    STOP_TEST7.append(0)
#                    print("NOT PASSED")
                    return (E)
                elif (E[fij][2] == 0) and (ftest_dist_nucleus_off_1 >= round(2.0*sigma,13)) and (ftest_dist_nucleus_off_1 <= inter_range_off):
                    ftmp_off_2.append([fij,f])
                    ftmp_inter_range_off_mono.append(fij)
# changing old coordinates to the new ones and updating relevant reaction/population lists          
        if ftest_dist_off == False:
            STOP_TEST7.append(1)
#            print("PASSED")
            monomers7.remove(index_04) # = [x for x in monomers1 if not index_04 in x]
            if index_04 in monos_inter_diffusion_on7: # removing the first free monomer from previous reaction lists
                monos_inter_diffusion_on7.remove(index_04)
                
#            dimer_on_association7 = [fw13 for fw13 in dimer_on_association7 if not index_04 in fw13]
            for fw13 in dimer_on_association7:
                if index_04 in fw13:
                    ftmp_dimer_on_association7.append(fw13)
                if f in fw13:
                    ftmp_dimer_on_association7.append(fw13)
            for fw133 in ftmp_dimer_on_association7:
                dimer_on_association7.remove(fw133)
            

            for fw12 in dimer_off_association7:
                if index_04 in fw12:
                    ftmp_dimer_off_association7.append(fw12)
                elif f in fw12:
                    ftmp_dimer_off_association7.append(fw12)
            for fw122 in ftmp_dimer_off_association7:
                dimer_off_association7.remove(fw122)

            
            for fw11 in pre_nucleus_on_association7:
                if index_04 in fw11:
                    ftmp_pre_nucleus_on_association7.append(fw11)
            for fw111 in ftmp_pre_nucleus_on_association7:
                pre_nucleus_on_association7.remove(fw111)
            
            for fw10 in pre_nucleus_off_association7:
                if f in fw10:
                    ftmp_pre_nucleus_off_association7.append(fw10)
                elif index_04 in fw10:
                    ftmp_pre_nucleus_off_association7.append(fw10)
#                elif flength_2[0] in fw10:
#                    ftmp_pre_nucleus_off_association7.append(fw10)
#                elif flength_2[1] in fw10:
#                    ftmp_pre_nucleus_off_association7.append(fw10)
            for fw4 in ftmp_pre_nucleus_off_association7:
                pre_nucleus_off_association7.remove(fw4)
            
            
            for fw9 in fibril_association7:
                if index_04 in fw9:
                    ftmp_fibril_association7.append(fw9)
            for fw99 in ftmp_fibril_association7:
                fibril_association7.remove(fw99)
            
            
            for fw8 in second_monomer_attach7:
                if index_04 in fw8:
                    ftmp_second_monomer_attach7.append(fw8)
            for fw88 in ftmp_second_monomer_attach7:
                second_monomer_attach7.remove(fw88)
            
            for fw6 in second_pre_nucleus_association7:
                if index_04 in fw6:
                    ftmp_second_pre_nucleus_association7.append(fw6)
            for fw66 in ftmp_second_pre_nucleus_association7:
                second_pre_nucleus_association7.remove(fw66)
            
            
            for fw5 in second_dimer_association7:
                if index_04 in fw5:
                    ftmp_second_dimer_association7.append(fw5)
            for fw55 in ftmp_second_dimer_association7:
                second_dimer_association7.remove(fw55)
            
    #            monos_inter_diffusion_on1 = [x for x in monos_inter_diffusion_on1 if not index_04 in x]
            if index_04 in monos_inter_diffusion_off7:            
                monos_inter_diffusion_off7.remove(index_04) 
                
            ftmp_nucleus_off.append(20 + len(flength_2))        # giving the number ID 
            ftmp_nucleus_off.append(fslope_2)                  # giving slope ID
            E[index_04][:] = ftmp_nucleus_off[:]
#            flength_2_old = flength_2[:]                 # transfering info about the aggregte before new addition
            flength_2.append(index_04)                  # adding new monomer to the aggregate
            
#            print("flength_2",flength_2, len(flength_2))
            

            if len(flength_2) == 3:
#                print("in trimer off association routine", index_04,f)                
                for fw3 in dimer_off_diffusion7:
                    if f in fw3:
                        ftmp_dimer_off_diffusion7.append(fw3)
                    elif flength_2[0] in fw3:
                        ftmp_dimer_off_diffusion7.append(fw3)
                    elif flength_2[1] in fw3:
                        ftmp_dimer_off_diffusion7.append(fw3)
                for fw33 in ftmp_dimer_off_diffusion7:
                    dimer_off_diffusion7.remove(fw33)
                    
#                dimer_off_dissociation7.remove(f)
                for ftr4 in dimer_off_dissociation7:
#                    print("ftr4 in dimer dissociation", ftr4)
                    if ftr4 == index_04:
                        ftmp_dimer_off_dissociation7.append(ftr4)
#                        print("the formation of trimer off, delete dimer off", ftr4)
                    elif ftr4 == f:
                        ftmp_dimer_off_dissociation7.append(ftr4)
#                        print("the formation of trimer off, delete dimer off", ftr4)
                for ftr44 in ftmp_dimer_off_dissociation7:
                    dimer_off_dissociation7.remove(ftr44)
                
                for fjj1 in ftmp_off_2:
                    pre_nucleus_off_association7.append(fjj1) # = pre_nucleus_off_association1 + tmp_off_2
                for fkk in ftmp_inter_range_off_mono:
                    monos_inter_diffusion_off7.append(fkk) # = monos_inter_diffusion_off1 + tmp_inter_range_off_mono
                
                pre_nucleus_off_dissociation7.append(flength_2)
                trimer_off_diffusion7.append(flength_2)                
                
            elif len(flength_2) == 4:
#                trimer_off_diffusion7 = [fw2 for fw2 in trimer_off_diffusion7 if not flength_2_old[0] in fw2]
#                print("in tetramer off association routine")
                for fw2 in trimer_off_diffusion7:
                    if f in fw2:
                        ftmp_trimer_off_diffusion7.append(fw2)
                    elif flength_2[0] in fw2:
                        ftmp_trimer_off_diffusion7.append(fw2)
                    elif flength_2[1] in fw2:
                        ftmp_trimer_off_diffusion7.append(fw2)
                for fw22 in ftmp_trimer_off_diffusion7:
                    trimer_off_diffusion7.remove(fw22)
                
                for fvaq in pre_nucleus_off_dissociation7:
                    if f in fvaq:
                        fpre_nucleus_off_dissociation7.append(fvaq)
                    elif index_04 in fvaq:
                        fpre_nucleus_off_dissociation7.append(fvaq)
                for fvaq1 in fpre_nucleus_off_dissociation7:
                    pre_nucleus_off_dissociation7.remove(fvaq1)
                    
                tetramer_off_diffusion7.append(flength_2)
                    
                pre_nucleus_off_dissociation7.append(flength_2)
                
                for ffj in ftmp_off_2:
                    pre_nucleus_off_association7.append(ffj) # = pre_nucleus_off_association1 + tmp_off_2
                for ffk in ftmp_inter_range_off_mono:
                    monos_inter_diffusion_off7.append(ffk) # = monos_inter_diffusion_off1 + tmp_inter_range_off_mono

            elif len(flength_2) == 5:
#                tetramer_off_diffusion7 = [fw1 for fw1 in tetramer_off_diffusion7 if not flength_2_old[0] in fw1]
#                print("in pentamer off association routine")
                
                for fw1 in tetramer_off_diffusion7:
                    if f in fw1:
                        ftmp_tetramer_off_diffusion7.append(fw1)
                    elif flength_2[0] in fw1:
                        ftmp_tetramer_off_diffusion7.append(fw1)
                    elif flength_2[1] in fw1:
                        ftmp_tetramer_off_diffusion7.append(fw1)
                for fw11 in ftmp_tetramer_off_diffusion7:
                    tetramer_off_diffusion7.remove(fw11)
                    
                for fgtaq in pre_nucleus_off_dissociation7:
                    if f in fgtaq:
                        fpre_nucleus_off_dissociation7.append(fgtaq)
                    elif index_04 in fgtaq:
                        fpre_nucleus_off_dissociation7.append(fgtaq)
                for fgtaq1 in fpre_nucleus_off_dissociation7:
                    pre_nucleus_off_dissociation7.remove(fgtaq1)
                
                for ffj in ftmp_off_2:
                    pre_nucleus_off_association7.append(ffj) # = pre_nucleus_off_association1 + tmp_off_2
                for ffk in ftmp_inter_range_off_mono:
                    monos_inter_diffusion_off7.append(ffk) # = monos_inter_diffusion_off1 + tmp_inter_range_off_mono
                    
                pre_nucleus_off_dissociation7.append(flength_2)
                
                pentamer_off_diffusion7.append(flength_2)
#                nucleus_off_numb7.append(flength_2)
                
            elif len(flength_2) == 6:
#                tetramer_off_diffusion7 = [fw1 for fw1 in tetramer_off_diffusion7 if not flength_2_old[0] in fw1]
#                print("in hexamer off association routine")
                
                for fw51 in pentamer_off_diffusion7:
                    if f in fw51:
                        ftmp_pentamer_off_diffusion7.append(fw51)
                    elif flength_2[0] in fw51:
                        ftmp_pentamer_off_diffusion7.append(fw51)
                    elif flength_2[1] in fw51:
                        ftmp_pentamer_off_diffusion7.append(fw51)
                for fw551 in ftmp_pentamer_off_diffusion7:
                    pentamer_off_diffusion7.remove(fw551)
                    
                for frad in pre_nucleus_off_dissociation7:
                    if index_04 in frad:
                        fpre_nucleus_off_dissociation7.append(frad)
                    elif f in frad:
                        fpre_nucleus_off_dissociation7.append(frad)
                for frad1 in fpre_nucleus_off_dissociation7:
                    pre_nucleus_off_dissociation7.remove(frad1)
                    
                pre_nucleus_off_dissociation7.append(flength_2)
                
#                pentamer_off_diffusion7.append(flength_2)
                nucleus_off_numb7.append(flength_2)
    elif len(flength_2) == nuc_size:
        STOP_TEST7.append(0)
        for fkhs2 in pre_nucleus_off_association7:
            if f in fkhs2:
                ftmp_pre_nucleus_off_association7.append(fkhs2)
        for fkhs3 in ftmp_pre_nucleus_off_association7:
            pre_nucleus_off_association7.remove(fkhs3)
        
            
    ftmp_dimer_off_association7 = []
    ftmp_inter_range_off_mono = []
    ftmp_dimer_on_association7 = []
    ftmp_pre_nucleus_on_association7 = []
    ftmp_pre_nucleus_off_association7 = []
    ftmp_fibril_association7 = []
    ftmp_second_monomer_attach7 = []
    ftmp_second_pre_nucleus_association7 = []
    ftmp_second_dimer_association7 = []
    ftmp_trimer_off_diffusion7 = []
    ftmp_tetramer_off_diffusion7 = []
    ftmp_pentamer_off_diffusion7 = []
    fpre_nucleus_off_dissociation7 = []
#                        
    return (E)
############################################ end pre nucleus off association function ###################################
    
    
#############################################  attach a secondary monomer to the side fibril ############################
#adds a monomer on the side of a fibril and starts a secondary nucleated fibril WITH PARTICLE ID AS END \
#and FIBRIL ID AS SLOPE
def mono_second(V,index_05,sigma,g,monomers8,monos_inter_diffusion_on8, monos_inter_diffusion_off8,
                    dimer_on_association8, dimer_off_association8, pre_nucleus_on_association8,
                    fibril_association8, second_monomer_attach8, second_fibril_association8, 
                    second_pre_nucleus_association8, second_dimer_association8, pre_nucleus_off_association8,
                    pre_nucleus_on_dissociation8,second_monomer_detach8,pentamer_on_diffusion8,STOP_TEST8):     # index_05 is the free monomer, bb is the monomer within the fibril
#    print("mono second attaching!!!!!!!!!!!!!!",index_05,V[index_05],g,V[g])
    gslope_second = ((V[index_05][1] - V[g][1])/(V[index_05][0] - V[g][0]))
    gangle_05= math.atan(gslope_second)
    gtest_dist_second_mono = False  # check overlap for new position    
    gslope_old = V[g][3]
    gtmp_mono_sec = []  # new coordinates
    gtmp_mono_sec_mono = [] # find free monomers within interaction range
    gtmp_inter_range_on_mono = []
    gtmp_dimer_on_association8 = []
    gtmp_pre_nucleus_on_association8 = []
    gtmp_dimer_off_association8 = []
    gtmp_fibril_association8 = []
    gtmp_second_monomer_attach8 = []
    gtmp_second_fibril_association8 = []
    gtmp_second_dimer_association8 = []
    gtmp_second_pre_nucleus_association8 = []
    gtmp_pre_nucleus_off_association8 = []
    gtmp_pentamer_on_diffusion8 = []

# find new coordinates for the monomer
    gtmp_mono_sec.append(V[g][0] + 2.0*sigma*math.cos(gangle_05))
    gtmp_mono_sec.append(V[g][1] + 2.0*sigma*math.sin(gangle_05))
#    print("mono second attach", g, V[g], gtmp_mono_sec[0], gtmp_mono_sec[1])
#    print("seond dimer attach", second_monomer_attach8)
# check for periodic boundary conditions
    if gtmp_mono_sec[0] > 0:            
        gtmp_mono_sec[0] = gtmp_mono_sec[0]%boxSide
    if gtmp_mono_sec[0] < 0:
        gtmp_mono_sec[0] = boxSide - abs(gtmp_mono_sec[0])%boxSide
    if gtmp_mono_sec[1] > 0:
        gtmp_mono_sec[1] = gtmp_mono_sec[1]%boxSide
    if gtmp_mono_sec[1] < 0:
        gtmp_mono_sec[1] = boxSide - abs(gtmp_mono_sec[1])%boxSide 
        
    for gi in range(len(V)):
        if gi != index_05:
            if V[gi][3] != gslope_old:
                gdist_mono_sec = round(dist(gtmp_mono_sec,V[gi]),13)
                if gdist_mono_sec < round(2.0*sigma,13):
                    gtest_dist_second_mono = True
#                    print("NOT PASSED",gtmp_mono_sec,V[gi],gdist_mono_sec,V[g])
                    STOP_TEST8.append(0)
                    return (V)
                elif (V[gi][2] == 0):
                    if (gdist_mono_sec > round(2.0*sigma,13)) and (gdist_mono_sec <= inter_range_on):
                        gtmp_mono_sec_mono.append([gi,index_05])
                        gtmp_inter_range_on_mono.append(gi)
    if gtest_dist_second_mono == False:     
        STOP_TEST8.append(1)
#        print("PASSED")
        monomers8[:] = [gex for gex in monomers8 if gex != index_05]
        if index_05 in monos_inter_diffusion_on8: # removing the first free monomer from previous reaction lists
            monos_inter_diffusion_on8.remove(index_05)
            
        for gy10 in dimer_on_association8:
            if index_05 in gy10:
                gtmp_dimer_on_association8.append(gy10)
        for gy100 in gtmp_dimer_on_association8:
            dimer_on_association8.remove(gy100)
#        dimer_on_association8 = [gy10 for gy10 in dimer_on_association8 if not index_05 in gy10]
        for gy8 in pre_nucleus_on_association8:
            if index_05 in gy8:
                gtmp_pre_nucleus_on_association8.append(gy8)
        for gy88 in gtmp_pre_nucleus_on_association8:
            pre_nucleus_on_association8.remove(gy88)
            
        for gy9 in dimer_off_association8:
            if index_05 in gy9:
                gtmp_dimer_off_association8.append(gy9)
        for gy99 in gtmp_dimer_off_association8:
            dimer_off_association8.remove(gy99)
#        dimer_off_association8 = [gy9 for gy9 in dimer_off_association8 if not index_05 in gy9]
#        pre_nucleus_on_association8 = [gy8 for gy8 in pre_nucleus_on_association8 if not index_05 in gy8]
        for gy7 in fibril_association8:
            if index_05 in gy7:
                gtmp_fibril_association8.append(gy7)
        for gy77 in gtmp_fibril_association8:
            fibril_association8.remove(gy77)
#        fibril_association8 = [gy7 for gy7 in fibril_association8 if not index_05 in gy7]
#        second_monomer_attach8 = [gy6 for gy6 in second_monomer_attach8 if not index_05 in gy6]
        for gy5 in second_monomer_attach8:
            if index_05 in gy5:
                gtmp_second_monomer_attach8.append(gy5)
            elif g in gy5:
                gtmp_second_monomer_attach8.append(gy5)
        for gy6 in gtmp_second_monomer_attach8:
            second_monomer_attach8.remove(gy6)
            
#        second_monomer_attach8 = [gy5 for gy5 in second_monomer_attach8 if not g in gy5]            
#        second_fibril_association8 = [gy4 for gy4 in second_fibril_association8 if not index_05 in gy4]
        for gy4 in second_fibril_association8:
            if index_05 in gy4:
                gtmp_second_fibril_association8.append(gy4)
        for gy44 in gtmp_second_fibril_association8:
            second_fibril_association8.remove(gy44)
            
#        second_pre_nucleus_association8 = [gy3 for gy3 in second_pre_nucleus_association8 if not index_05 in gy3]           
        for gy3 in second_pre_nucleus_association8:
            if index_05 in gy3:
                gtmp_second_pre_nucleus_association8.append(gy3)
        for gy33 in gtmp_second_pre_nucleus_association8:
            second_pre_nucleus_association8.remove(gy33)
            
        for gy2 in second_dimer_association8:
            if index_05 in gy2:
                gtmp_second_dimer_association8.append(gy2)
        for gy22 in gtmp_second_dimer_association8:
            second_dimer_association8.remove(gy22)
#        second_dimer_association8 = [gy2 for gy2 in second_dimer_association8 if not index_05 in gy2]            

        if index_05 in monos_inter_diffusion_off8:            
            monos_inter_diffusion_off8.remove(index_05)# = [x for x in monos_inter_diffusion_off1 if not index_05 in x]
#        pre_nucleus_off_association8 = [gy1 for gy1 in pre_nucleus_off_association8 if not index_05 in gy1]
        for gy1 in pre_nucleus_off_association8:
            if index_05 in gy1:
                gtmp_pre_nucleus_off_association8.append(gy1)
        for gy11 in gtmp_pre_nucleus_off_association8:
            pre_nucleus_off_association8.remove(gy11)
            
        for gyy1 in pentamer_on_diffusion8:
            if g in gyy1:
                gtmp_pentamer_on_diffusion8.append(gyy1)
        for gyy11 in gtmp_pentamer_on_diffusion8:
            pentamer_on_diffusion8.remove(gyy11)
        if gtmp_mono_sec[1] > V[g][1]:
            gtmp_mono_sec.append(301)
        elif gtmp_mono_sec[1] < V[g][1]:
            gtmp_mono_sec.append(302)
        gtmp_mono_sec.append(gslope_second)
        V[index_05][:] = gtmp_mono_sec[:]
        
        second_monomer_detach8.append(index_05) # create/modify secondary monomer detach list
        
        for glk in gtmp_mono_sec_mono:
            second_dimer_association8.append(glk) #= second_dimer_association1 + tmp_mono_sec_mono # create/modify secondary dimer association list
        for gll in gtmp_inter_range_on_mono:
            monos_inter_diffusion_on8.append(gll) # = monos_inter_diffusion_on1 + tmp_inter_range_on_mono
    gtmp_mono_sec = []  # new coordinates
    gtmp_mono_sec_mono = [] # find free monomers within interaction range
    gtmp_inter_range_on_mono = []
    gtmp_dimer_on_association8 = []
    gtmp_pre_nucleus_on_association8 = []
    gtmp_dimer_off_association8 = []
    gtmp_fibril_association8 = []
    gtmp_second_monomer_attach8 = []
    gtmp_second_fibril_association8 = []
    gtmp_second_dimer_association8 = []
    gtmp_second_pre_nucleus_association8 = []
    gtmp_pre_nucleus_off_association8 = []
    gtmp_pentamer_on_diffusion8 = []

    return (V)
############################################ end secondary monomer attach function ######################################
    
    
############################################# secondary dimer formation #############################################
# CREATES A secondary dimer WITH PARTICLE ID AS END and FIBRIL ID AS SLOPE
def dimer_second(JJ,index_06,sigma,h,monomers9,monos_inter_diffusion_on9, monos_inter_diffusion_off9,
                    dimer_on_association9, dimer_off_association9, pre_nucleus_on_association9,
                    fibril_association9, second_monomer_attach9, second_fibril_association9, 
                    second_pre_nucleus_association9, second_dimer_association9, pre_nucleus_off_association9,
                    pre_nucleus_on_dissociation9,second_monomer_detach9, second_pre_nucleus_dissociation9,STOP_TEST9): # index_06 is the free monomer, hh is the aggregate monmomer
    hslope_sec = JJ[h][3]  
    htest_dist_sec_dimer = False
    htmp_sec  = []  
    htmp_mono_sec_dimer = []
    htmp_inter_range_on_mono = []
    htmp_dimer_on_association9 = []
    htmp_dimer_off_association9 = []
    htmp_pre_nucleus_on_association9 = []
    htmp_fibril_association9 = []
    htmp_second_monomer_attach9 = []
    htmp_second_pre_nucleus_association9 = []
    htmp_second_fibril_association9 = []
    htmp_second_dimer_association9 = []
    htmp_pre_nucleus_off_association9 = []
    if hslope_sec < 0:          
# find new coordinates for the free monomer
        if JJ[h][2] == 301:
            htmp_sec.append(JJ[h][0] + 2.0*sigma*math.cos(math.pi - math.atan(hslope_sec)))
            htmp_sec.append(JJ[h][1] + 2.0*sigma*math.sin(math.pi - math.atan(hslope_sec)))
        elif JJ[h][2] == 302:
            htmp_sec.append(JJ[h][0] + 2.0*sigma*math.cos(math.atan(hslope_sec)))
            htmp_sec.append(JJ[h][1] + 2.0*sigma*math.sin(math.atan(hslope_sec))) 
    elif hslope_sec > 0:
        if JJ[h][2] == 301:
            htmp_sec.append(JJ[h][0] - 2.0*sigma*math.cos(math.pi + math.atan(hslope_sec)))
            htmp_sec.append(JJ[h][1] + 2.0*sigma*math.sin(math.pi - math.atan(hslope_sec)))
        elif JJ[h][2] == 302:
            htmp_sec.append(JJ[h][0] + 2.0*sigma*math.cos(math.pi - math.atan(hslope_sec)))
            htmp_sec.append(JJ[h][1] - 2.0*sigma*math.sin(math.pi - math.atan(hslope_sec))) 
# checking boundary conditions
    if htmp_sec[0] > 0:            
        htmp_sec[0] = htmp_sec[0]%boxSide
    if htmp_sec[0] < 0:
        htmp_sec[0] = boxSide - abs(htmp_sec[0])%boxSide
    if htmp_sec[1] > 0:
        htmp_sec[1] = htmp_sec[1]%boxSide
    if htmp_sec[1] < 0:
        htmp_sec[1] = boxSide - abs(htmp_sec[1])%boxSide        
# check for overlaps and free monomers within interaction range        
    for hyr in range(len(JJ)):      
        if hyr != index_06:
            if hyr != h:
                htest_dist_dimer_sec = round(dist(htmp_sec,JJ[hyr]),13)
                if htest_dist_dimer_sec < round(2.0*sigma,13):
                    htest_dist_sec_dimer = True
                    STOP_TEST9.append(0)
#                    print("NOT PASSED")
                    return (JJ)
                elif (JJ[hyr][2] == 0) and (htest_dist_dimer_sec >= round(2.0*sigma,13)) and (htest_dist_dimer_sec <= inter_range_on):
                    htmp_mono_sec_dimer.append([hyr,index_06])
                    htmp_inter_range_on_mono.append(hyr)
# change old coordinates with new ones and update relevant reaction/population lists
    if htest_dist_sec_dimer == False:   
        STOP_TEST9.append(1)
#        print("PASSED")
        monomers9[:] = [hhex for hhex in monomers9 if hhex != index_06]
        if index_06 in monos_inter_diffusion_on9: # removing the first free monomer from previous reaction lists
            monos_inter_diffusion_on9.remove(index_06)
#        dimer_on_association9 = [hx10 for hx10 in dimer_on_association9 if not index_06 in hx10]
        for hx10 in dimer_on_association9:
            if index_06 in hx10:
                htmp_dimer_on_association9.append(hx10)
        for hx100 in htmp_dimer_on_association9:
            dimer_on_association9.remove(hx100)
#        dimer_off_association9 = [hx9 for hx9 in dimer_off_association9 if not index_06 in hx9]
        for hx9 in dimer_off_association9:
            if index_06 in hx9:
                htmp_dimer_off_association9.append(hx9)
        for hx99 in htmp_dimer_off_association9:
            dimer_off_association9.remove(hx99)
#        pre_nucleus_on_association9 = [hx8 for hx8 in pre_nucleus_on_association9 if not index_06 in hx8]
        for hx7 in pre_nucleus_on_association9:
            if index_06 in hx7:
                htmp_pre_nucleus_on_association9.append(hx7)
        for hx77 in htmp_pre_nucleus_on_association9:
            pre_nucleus_on_association9.remove(hx77)
#        fibril_association9 = [hx7 for hx7 in fibril_association9 if not index_06 in hx7]
        for hx8 in fibril_association9:
            if index_06 in hx8:
                htmp_fibril_association9.append(hx8)
        for hx88 in htmp_fibril_association9:
            fibril_association9.remove(hx88)
#        second_monomer_attach9 = [hx6 for hx6 in second_monomer_attach9 if not index_06 in hx6]
        for hx6 in second_monomer_attach9:
            if index_06 in hx6:
                htmp_second_monomer_attach9.append(hx6)
        for hx66 in htmp_second_monomer_attach9:
            second_monomer_attach9.remove(hx66)
#        second_fibril_association9 = [hx5 for hx5 in second_fibril_association9 if not index_06 in hx5]
        for hx5 in second_fibril_association9:
            if index_06 in hx5:
                htmp_second_fibril_association9.append(hx5)
        for hx55 in htmp_second_fibril_association9:
            second_fibril_association9.remove(hx55)
#        second_pre_nucleus_association9 = [hx4 for hx4 in second_pre_nucleus_association9 if not index_06 in hx4]           
        for hx4 in second_pre_nucleus_association9:
            if index_06 in hx4:
                htmp_second_pre_nucleus_association9.append(hx4)
            elif h in hx4:
                htmp_second_pre_nucleus_association9
        for hx44 in htmp_second_pre_nucleus_association9:
            second_pre_nucleus_association9.remove(hx44)
#        second_dimer_association9 = [hx3 for hx3 in second_dimer_association9 if not index_06 in hx3]   
        for hx3 in second_dimer_association9:
            if index_06 in hx3:
                htmp_second_dimer_association9.append(hx3)
            elif h in hx3:
                htmp_second_dimer_association9.append(hx3)
        for hx33 in htmp_second_dimer_association9:
            second_dimer_association9.remove(hx33)
#        second_dimer_association9 = [hx2 for hx2 in second_dimer_association9 if not h in hx2]            
#            monos_inter_diffusion_on1 = [x for x in monos_inter_diffusion_on1 if not index_06 in x]
        if index_06 in monos_inter_diffusion_off9:
            monos_inter_diffusion_off9.remove(index_06) # = [x for x in monos_inter_diffusion_on1 if not index_06 in x]
#        pre_nucleus_off_association9 = [hx1 for hx1 in pre_nucleus_off_association9 if not index_06 in hx1]
        for hx1 in pre_nucleus_off_association9:
            if index_06 in hx1:
                htmp_pre_nucleus_off_association9.append(hx1)
        for hx11 in htmp_pre_nucleus_off_association9:
            pre_nucleus_off_association9.remove(hx11)
        htmp_sec.append(303)
        htmp_sec.append(hslope_sec)
        JJ[index_06][:] = htmp_sec[:]
        JJ[h][2] = JJ[h][2]*10

        second_monomer_detach9.remove(h)
        
        for hlh in htmp_mono_sec_dimer:
            second_pre_nucleus_association9.append(hlh) # = second_pre_nucleus_association1 + tmp_mono_sec_dimer#modify the secondary prenucleus association reaction list        

        second_pre_nucleus_dissociation9.append(index_06)#modify the secondary prenucleus dissociation reaction list

        for hlj in htmp_inter_range_on_mono:
            monos_inter_diffusion_on9.append(hlj) # = monos_inter_diffusion_on1 + tmp_inter_range_on_mono  # add monomer to the monomers within range of interaction 
    htmp_sec  = []  
    htmp_mono_sec_dimer = []
    htmp_inter_range_on_mono = []
    htmp_dimer_on_association9 = []
    htmp_dimer_off_association9 = []
    htmp_pre_nucleus_on_association9 = []
    htmp_fibril_association9 = []
    htmp_second_monomer_attach9 = []
    htmp_second_pre_nucleus_association9 = []
    htmp_second_fibril_association9 = []
    htmp_second_dimer_association9 = []
    htmp_pre_nucleus_off_association9 = []

    return (JJ)
############################################ end secondary dimer formation ##############################################

    
############################################# elongation of secondary pre nucleus ####################################
#  elongates a secondary dimer to the size of a Nucleus WITH PARTICLE ID AS END and FIBRIL ID AS SLOPE
def nucleus_second(JJJ,index_07,sigma,j,monomers10,monos_inter_diffusion_on10, monos_inter_diffusion_off10,
                    dimer_on_association10, dimer_off_association10, pre_nucleus_on_association10,
                    fibril_association10, second_monomer_attach10, second_fibril_association10, second_fibril_dissociation10,
                    second_pre_nucleus_association10, second_dimer_association10, pre_nucleus_off_association10,
                    pre_nucleus_on_dissociation10,second_monomer_detach10, second_pre_nucleus_dissociation10,
                    second_fibril_detach10,STOP_TEST10): # index_07 is the free monomer, hhh is the aggregate monomer
#    print("end of pre nucl second", j, JJJ[j])
    jlength_sec = []    
    jslope_sec_1 = JJJ[j][3]  
    jtmp_sec_1  = []  
    jtest_dist_sec_nucleus = False
    jtmp_mono_sec_nucleus = []
    jtmp_mono_sec_inter = []
    jtmp_dimer_on_association10 = []
    jtmp_dimer_off_association10 = []
    jtmp_pre_nucleus_on_association10 = []
    jtmp_fibril_association10 = []
    jtmp_second_monomer_attach10 = []
    jtmp_second_fibril_association10 = []
    jtmp_second_pre_nucleus_association10 = []
    jtmp_second_dimer_association10 = []
    jtmp_pre_nucleus_off_association10 = []
    jtmp_sec_1.append(JJJ[j][0] + 2.0*sigma*math.cos(math.atan(jslope_sec_1)))
    jtmp_sec_1.append(JJJ[j][1] + 2.0*sigma*math.sin(math.atan(jslope_sec_1)))
    if jtmp_sec_1[0] > 0:            
        jtmp_sec_1[0] = jtmp_sec_1[0]%boxSide
    if jtmp_sec_1[0] < 0:
        jtmp_sec_1[0] = boxSide - abs(jtmp_sec_1[0])%boxSide
    if jtmp_sec_1[1] > 0:
        jtmp_sec_1[1] = jtmp_sec_1[1]%boxSide
    if jtmp_sec_1[1] < 0:
        jtmp_sec_1[1] = boxSide - abs(jtmp_sec_1[1])%boxSide      
    for jw in range(len(JJJ)):
        if jw != index_07:
#            if jw != j:
                jtest_dist_nucleus_sec = round(dist(jtmp_sec_1, JJJ[jw]),13)
                if jtest_dist_nucleus_sec < round(2.0*sigma,13):
                    jtest_dist_sec_nucleus = True
                    STOP_TEST10.append(0)
#                    print("NOT PASSED")
                    return(JJJ)
                elif (JJJ[jw][2] == 0) and (jtest_dist_nucleus_sec >= round(2.0*sigma,13)) and (jtest_dist_nucleus_sec <= inter_range_on):
                    jtmp_mono_sec_nucleus.append([jw,index_07])
                    jtmp_mono_sec_inter.append(jw)
                elif JJJ[jw][3] == jslope_sec_1:
                    jlength_sec.append(jw)
    if jtest_dist_sec_nucleus == False: 
        STOP_TEST10.append(1)
#        print("PASSED")
        monomers10[:] = [jex for jex in monomers10 if jex != index_07]
        if index_07 in monos_inter_diffusion_on10: # removing the first free monomer from previous reaction lists
            monos_inter_diffusion_on10.remove(index_07)
#        dimer_on_association10 = [jx10 for jx10 in dimer_on_association10 if not index_07 in jx10]
        for jx10 in dimer_on_association10:
            if index_07 in jx10:
                jtmp_dimer_on_association10.append(jx10)
        for jx100 in jtmp_dimer_on_association10:
            dimer_on_association10.remove(jx100)
#        dimer_off_association10 = [jx09 for jx09 in dimer_off_association10 if not index_07 in jx09]
        for jx09 in dimer_off_association10:
            if index_07 in jx09:
                jtmp_dimer_off_association10.append(jx09)
        for jx099 in jtmp_dimer_off_association10:
            dimer_off_association10.remove(jx099)
#        pre_nucleus_on_association10 = [jx08 for jx08 in pre_nucleus_on_association10 if not index_07 in jx08]
        for jx08 in pre_nucleus_on_association10:
            if index_07 in jx08:
                jtmp_pre_nucleus_on_association10.append(jx08)
        for jx088 in jtmp_pre_nucleus_on_association10:
            pre_nucleus_on_association10.remove(jx088)
#        fibril_association10 = [jx07 for jx07 in fibril_association10 if not index_07 in jx07]
        for jx07 in fibril_association10:
            if index_07 in jx07:
                jtmp_fibril_association10.append(jx07)
        for jx077 in jtmp_fibril_association10:
            fibril_association10.remove(jx077)
#        second_monomer_attach10 = [jx06 for jx06 in second_monomer_attach10 if not index_07 in jx06]
        for jx06 in second_monomer_attach10:
            if index_07 in jx06:
                jtmp_second_monomer_attach10.append(jx06)
        for jx066 in jtmp_second_monomer_attach10:
            second_monomer_attach10.remove(jx066)
#        second_fibril_association10 = [jx05 for jx05 in second_fibril_association10 if not index_07 in jx05]
        for jx05 in second_fibril_association10:
            if index_07 in jx05:
                jtmp_second_fibril_association10.append(jx05)
        for jx055 in jtmp_second_fibril_association10:
            second_fibril_association10.remove(jx055)
#        second_pre_nucleus_association10 = [jx04 for jx04 in second_pre_nucleus_association10 if not index_07 in jx04]  
        for jx04 in second_pre_nucleus_association10:
            if index_07 in jx04:
                jtmp_second_pre_nucleus_association10.append(jx04)
            elif j in jx04:
                jtmp_second_pre_nucleus_association10.append(jx04)
        for jx03 in jtmp_second_pre_nucleus_association10:
            second_pre_nucleus_association10.remove(jx03)
#        second_dimer_association10 = [jx02 for jx02 in second_dimer_association10 if not index_07 in jx02]   
        for jx02 in second_dimer_association10:
            if index_07 in jx02:
                jtmp_second_dimer_association10.append(jx02)
        for jx022 in jtmp_second_dimer_association10:
            second_dimer_association10.remove(jx022)
#            monos_inter_diffusion_on1 = [x for x in monos_inter_diffusion_on1 if not index_07 in x]
        if index_07 in monos_inter_diffusion_off10:
            monos_inter_diffusion_off10.remove(index_07)# = [x for x in monos_inter_diffusion_on1 if not index_07 in x]
#        pre_nucleus_off_association10 = [jx01 for jx01 in pre_nucleus_off_association10 if not index_07 in jx01]
        for jx01 in pre_nucleus_off_association10:
            if index_07 in jx01:
                jtmp_pre_nucleus_off_association10.append(jx01)
        for jx011 in jtmp_pre_nucleus_off_association10:
            pre_nucleus_off_association10.remove(jx011)
        JJJ[j][2] = 300   
        jtmp_sec_1.append(303)
        jtmp_sec_1.append(jslope_sec_1)
        JJJ[index_07][:] = jtmp_sec_1[:]
#        jlength_sec_old = jlength_sec[:]
        jlength_sec.append(index_07)
        
        if len(jlength_sec) <= 4:
            second_pre_nucleus_dissociation10.remove(j)
            for jhg in jtmp_mono_sec_nucleus:
                second_pre_nucleus_association10.append(jhg) # = second_pre_nucleus_association1 + tmp_mono_sec_nucleus
            second_pre_nucleus_dissociation10.append(index_07)
           
            for jfd in jtmp_mono_sec_inter:
                monos_inter_diffusion_on10.append(jfd) ## = monos_inter_diffusion_on1 + tmp_mono_sec_inter    
    
        elif len(jlength_sec) == 5:
            for js in jlength_sec:
                if JJJ[js][2] == 303:
                    JJJ[js][2] = 33    # changing the particle ID as lower y end of fibril, not pre-nucleus
                elif JJJ[js][2] == 300:
                    JJJ[js][2] = 30
                elif JJJ[js][2] == 3010:
                    JJJ[js][2] = 31       # changing the particle ID as higher y end of fibril, not pre-nucleus
                elif JJJ[js][2] == 3020:
                    JJJ[js][2] = 32       # changing the particle ID as higher y end of fibril, not pre-nucleus  
            second_pre_nucleus_dissociation10.remove(j)
#            for jhj in jtmp_mono_sec_nucleus:
#                second_fibril_association10.append(jhj)
            second_fibril_dissociation10.append(index_07)
            second_fibril_detach10.append(jlength_sec)
    
            for jmn in jtmp_mono_sec_inter:
                monos_inter_diffusion_on10 .append(jmn) #= monos_inter_diffusion_on1 + tmp_mono_sec_inter

    jtmp_mono_sec_nucleus = []
    jtmp_mono_sec_inter = []
    jtmp_dimer_on_association10 = []
    jtmp_dimer_off_association10 = []
    jtmp_pre_nucleus_on_association10 = []
    jtmp_fibril_association10 = []
    jtmp_second_monomer_attach10 = []
    jtmp_second_fibril_association10 = []
    jtmp_second_pre_nucleus_association10 = []
    jtmp_second_dimer_association10 = []
    jtmp_pre_nucleus_off_association10 = []

    return (JJJ)
############################################ end dsecondary nucleus function #############################################


#############################################  Elongates a fibril WITH PARTICLE ID AS END and FIBRIL ID AS SLOPE
def elongation_on_low_end(JEL,index_08,sigma,k,monomers11,monos_inter_diffusion_on11, monos_inter_diffusion_off11,
                    dimer_on_association11, dimer_off_association11, pre_nucleus_on_association11,
                    fibril_association11, second_monomer_attach11, second_fibril_association11, second_fibril_dissociation11,
                    second_pre_nucleus_association11, second_dimer_association11, pre_nucleus_off_association11,
                    pre_nucleus_on_dissociation11,second_monomer_detach11, second_pre_nucleus_dissociation11,
                    fibril_dissociation11,pentamer_on_diffusion11,STOP_TEST11):     # if the closest particle (h) is a pre-nucleus end (101) and it is within the interaction radius, the chosen to move is a monomer (a)   
#    print('elongation on low end')
    kslope_fib_low = JEL[k][3]  
#    print("slope elong low",kslope_fib_low)
    ktmp_low1  = []  
    ktest_dist_elong_low_test = False
    ktmp_low_mono = []
    ktmp_low_mono_inter = []
    ktmp_fibril_association11 = []
    ktmp_dimer_on_association = []
    ktmp_pre_nucleus_on_association11 = []
    ktmp_fibril_dissociation11 = []
    ktmp_dimer_off_association11 = []
    ktmp_second_monomer_attach11 = []
    ktmp_second_fibril_association11 = []
    ktmp_second_pre_nucleus_association11 = []
    ktmp_second_dimer_association11 = []
    ktmp_pre_nucleus_off_association11 = []
    ktmp_low_second = []
    ktmp_pentamer_on_diffusion11 = []
    
    if kslope_fib_low < 0:        
        ktmp_low1.append(JEL[k][0] - 2.0*sigma*math.cos(math.pi - math.atan(kslope_fib_low)))
        ktmp_low1.append(JEL[k][1] + 2.0*sigma*math.sin(math.pi - math.atan(kslope_fib_low)))
    elif kslope_fib_low > 0:
        ktmp_low1.append(JEL[k][0] + 2.0*sigma*math.cos(math.pi - math.atan(kslope_fib_low)))
        ktmp_low1.append(JEL[k][1] - 2.0*sigma*math.sin(math.pi - math.atan(kslope_fib_low))) 
    if ktmp_low1[0] > 0:            
        ktmp_low1[0] = ktmp_low1[0]%boxSide
    if ktmp_low1[0] < 0:
#            print("go one lower than box!", djj[0])
        ktmp_low1[0] = boxSide - abs(ktmp_low1[0])%boxSide
    if ktmp_low1[1] > 0:
        ktmp_low1[1] = ktmp_low1[1]%boxSide
    if ktmp_low1[1] < 0:
#            print("go one higher than box!", djj[1])
        ktmp_low1[1] = boxSide - abs(ktmp_low1[1])%boxSide  
    for kc in range(len(JEL)):
        if (JEL[kc][3] != kslope_fib_low) and (kc != index_08):
            ktest_dist_elong_low = round(dist(ktmp_low1, JEL[kc]),13)
            ktest_dist_second_low = round(dist(JEL[k],JEL[kc]),13)
            if ktest_dist_elong_low < round(2.0*sigma,13):
#                print("not passing dist test elong low", kc,JEL[kc],ktest_dist_elong_low)
                ktest_dist_elong_low_test = True
                STOP_TEST11.append(0)
#                print("NOT PASSED")
                return (JEL)
            elif JEL[kc][2] ==0:
                if (ktest_dist_elong_low >= round(2.0*sigma,13)) and (ktest_dist_elong_low <= inter_range_on):
                    ktmp_low_mono.append([kc,index_08])
                    ktmp_low_mono_inter.append(kc)
                if (ktest_dist_second_low >= round(2.0*sigma,13)) and (ktest_dist_second_low <= inter_range_on):
                    ktmp_low_second.append([kc,k])
                    ktmp_low_mono_inter.append(kc)
#    print("dist test in elong low",ktest_dist_elong_low)
    if ktest_dist_elong_low_test == False:  # checking for overlap    
        STOP_TEST11.append(1)
#        print("PASSED")
#        print('elongation on low', index_08, JEL[index_08], k, JEL[k])
        monomers11[:] = [kex for kex in monomers11 if kex != index_08]
        if index_08 in monos_inter_diffusion_on11: # removing the first free monomer from previous reaction lists
            monos_inter_diffusion_on11.remove(index_08)
            
        for kx12 in dimer_on_association11:
            if index_08 in kx12:
                ktmp_dimer_on_association.append(kx12)
        for kx122 in ktmp_dimer_on_association:
            dimer_on_association11.remove(kx122)
                
        for kx10 in pre_nucleus_on_association11:
            if index_08 in kx10:
                ktmp_pre_nucleus_on_association11.append(kx10)
        for kx100 in ktmp_pre_nucleus_on_association11:
            pre_nucleus_on_association11.remove(kx100)
                
        for kx8 in fibril_association11:
            if (index_08 in kx8): 
                ktmp_fibril_association11.append(kx8)                
            elif (k in kx8):
                ktmp_fibril_association11.append(kx8)                
        for kx88 in ktmp_fibril_association11:
            fibril_association11.remove(kx88)
            
#        dimer_on_association11 = [kx12 for kx12 in dimer_on_association11 if not index_08 in kx12]
#        dimer_off_association11 = [kx11 for kx11 in dimer_off_association11 if not index_08 in kx11]
        for kx11 in dimer_off_association11:
            if index_08 in kx11:
                ktmp_dimer_off_association11.append(kx11)
        for kx111 in ktmp_dimer_off_association11:
            dimer_off_association11.remove(kx111)
#        pre_nucleus_on_association11 = [kx10 for kx10 in pre_nucleus_on_association11 if not index_08 in kx10]
 #       fibril_association11 = [kx8 for kx8 in fibril_association11 if not index_08 in kx8]
  #      fibril_association11 = [kx7 for kx7 in fibril_association11 if not k in kx7]
#        second_monomer_attach11 = [kx6 for kx6 in second_monomer_attach11 if not index_08 in kx6]
        for kx6 in second_monomer_attach11:
            if index_08 in kx6:
                ktmp_second_monomer_attach11.append(kx6)
        for kx66 in ktmp_second_monomer_attach11:
            second_monomer_attach11.remove(kx66)
#        second_fibril_association11 = [kx5 for kx5 in second_fibril_association11 if not index_08 in kx5]
        for kx5 in second_fibril_association11:
            if index_08 in kx5:
                ktmp_second_fibril_association11.append(kx5)
        for kx55 in ktmp_second_fibril_association11:
            second_fibril_association11.remove(kx55)
#        second_pre_nucleus_association11 = [kx4 for kx4 in second_pre_nucleus_association11 if not index_08 in kx4]  
        for kx4 in second_pre_nucleus_association11:
            if index_08 in kx4:
                ktmp_second_pre_nucleus_association11.append(kx4)
        for kx44 in ktmp_second_pre_nucleus_association11:
            second_pre_nucleus_association11.remove(kx44)
#        second_dimer_association11 = [kx3 for kx3 in second_dimer_association11 if not index_08 in kx3]   
        for kx3 in second_dimer_association11:
            if index_08 in kx3:
                ktmp_second_dimer_association11.append(kx3)
        for kx33 in ktmp_second_dimer_association11:
            second_dimer_association11.remove(kx33)
#        monos_inter_diffusion_on1 = [x for x in monos_inter_diffusion_on1 if not index_08 in x]
        if index_08 in monos_inter_diffusion_off11:            
            monos_inter_diffusion_off11.remove(index_08)
#        pre_nucleus_off_association11 = [kx2 for kx2 in pre_nucleus_off_association11 if not index_08 in kx2]
        for kx2 in pre_nucleus_off_association11:
            if index_08 in kx2:
                ktmp_pre_nucleus_off_association11.append(kx2)
        for kx22 in ktmp_pre_nucleus_off_association11:
            pre_nucleus_off_association11.remove(kx22)
            
        for kxx65 in pentamer_on_diffusion11:
            if k in kxx65:
                ktmp_pentamer_on_diffusion11.append(kxx65)
        for kxx66 in ktmp_pentamer_on_diffusion11:
            pentamer_on_diffusion11.remove(kxx66)
            
        JEL[k][2] = 1   
        ktmp_low1.append(122)
        ktmp_low1.append(kslope_fib_low)
        JEL[index_08][:] = ktmp_low1[:]
        
        for kbht11 in fibril_dissociation11:
            if kbht11 == k:
                ktmp_fibril_dissociation11.append(kbht11)
        for kbht12 in ktmp_fibril_dissociation11:
            fibril_dissociation11.remove(kbht12)
            
            
        for kla in ktmp_low_mono:
            fibril_association11.append(kla) # = fibril_association1 + tmp_low_mono
#        fibril_dissociation11.remove(k)
        fibril_dissociation11.append(index_08)
        for kb in ktmp_low_mono_inter:
            monos_inter_diffusion_on11.append(kb) # = monos_inter_diffusion_on1 + tmp_low_mono_inter
        for kbb1 in ktmp_low_second:
            second_monomer_attach11.append(kbb1)
            
        
            
    ktmp_low_mono = []
    ktmp_low_mono_inter = []
    ktmp_fibril_association11 = []
    ktmp_dimer_on_association = []
    ktmp_pre_nucleus_on_association11 = []
    ktmp_fibril_dissociation11 = []
    ktmp_dimer_off_association11 = []
    ktmp_second_monomer_attach11 = []
    ktmp_second_fibril_association11 = []
    ktmp_second_pre_nucleus_association11 = []
    ktmp_second_dimer_association11 = []
    ktmp_pre_nucleus_off_association11 = []
    ktmp_low_second = []
    ktmp_low1  = []
    ktmp_pentamer_on_diffusion11 = []

    return (JEL)
############################################

#############################################  Elongates a fibril WITH PARTICLE ID AS END and FIBRIL ID AS SLOPE
def elongation_on_high_end(DEL,index_09,sigma,lEL, monomers12,mono_inter_diffusion_on12, mono_inter_diffusion_off12,
                    dimer_on_association12, dimer_off_association12, pre_nucleus_on_association12,
                    fibril_association12, second_monomer_attach12, second_fibril_association12, second_fibril_dissociation12,
                    second_pre_nucleus_association12, second_dimer_association12, pre_nucleus_off_association12,
                    pre_nucleus_on_dissociation12,second_monomer_detach12, second_pre_nucleus_dissociation12,
                    fibril_dissociation12,pentamer_on_diffusion12,STOP_TEST12): # index_09 is the free monomer, lEL the fibril end
#    print('elongation on high end')
    lslope_fib_high = DEL[lEL][3]     
#    print("slope in elong high",lslope_fib_high)
    ltmp_high  = []  
    ltmp_high_mono = []
    ltmp_high_mono_inter = []
    ltmp_fibril_association12 = []
    ltmp_dimer_on_association12 = []
    ltmp_pre_nucleus_on_association12 = []
    ltmp_fibril_dissociation12 = []
    ltmp_dimer_off_association12 = []
    ltmp_second_monomer_attach12 = []
    ltmp_second_fibril_association12 = []
    ltmp_second_pre_nucleus_association12 = []
    ltmp_pre_nucleus_off_association12 = []
    ltmp_second_high = []
    ltmp_second_dimer_association12 = []
    ltmp_pentamer_on_diffusion12 = []
    ltest_dist_elong_high_test = False
    if lslope_fib_high < 0:        
        ltmp_high.append(DEL[lEL][0] + 2.0*sigma*math.cos(math.pi - math.atan(lslope_fib_high)))
        ltmp_high.append(DEL[lEL][1] + 2.0*sigma*math.sin(math.pi + math.atan(lslope_fib_high)))
    elif lslope_fib_high > 0:
        ltmp_high.append(DEL[lEL][0] - 2.0*sigma*math.cos(math.pi + math.atan(lslope_fib_high)))
        ltmp_high.append(DEL[lEL][1] + 2.0*sigma*math.sin(math.pi - math.atan(lslope_fib_high))) 
    if ltmp_high[0] > 0:            
        ltmp_high[0] = ltmp_high[0]%boxSide
    if ltmp_high[0] < 0:
        ltmp_high[0] = boxSide - abs(ltmp_high[0])%boxSide
    if ltmp_high[1] > 0:
        ltmp_high[1] = ltmp_high[1]%boxSide
    if ltmp_high[1] < 0:
        ltmp_high[1] = boxSide - abs(ltmp_high[1])%boxSide    
    for li in range(len(DEL)):
        if (DEL[li][3] != lslope_fib_high) and (li != index_09):
            ltest_dist_elong_high = round(dist(ltmp_high, DEL[li]),13)
            ltest_dist_second_high = round(dist(DEL[lEL],DEL[li]),13)
            if ltest_dist_elong_high < round(2.0*sigma,13):
#                print("not passing dist test nucl high", li,DEL[li],ltest_dist_elong_high)
                ltest_dist_elong_high_test = True
                STOP_TEST12.append(0)
#                print("NOT PASSED")
#                break
                return (DEL)
            elif DEL[li][2] == 0:
                if ltest_dist_elong_high >= round(2.0*sigma,13) and ltest_dist_elong_high <= inter_range_on:
                    ltmp_high_mono.append([li,index_09])
                    ltmp_high_mono_inter.append(li)
                if ltest_dist_second_high >= round(2.0*sigma,13) and ltest_dist_second_high <= inter_range_on:
                    ltmp_second_high.append([li,lEL])
                    ltmp_high_mono_inter.append(li)
#    print("dist test elong high",ltest_dist_elong_high)
    if ltest_dist_elong_high_test == False:  # checking for overlap    
        STOP_TEST12.append(1)
#        print("PASSED")
#        print('elongationhigh on', index_09, DEL[index_09],lEL, DEL[lEL])
        monomers12[:] = [lx for lx in monomers12 if lx != index_09] #monomers12.remove(index_09) # = [lxx for lxx in monomers12 if not index_09 in lxx]
        if index_09 in mono_inter_diffusion_on12: # removing the first free monomer from previous reaction lists
            mono_inter_diffusion_on12.remove(index_09)
            
        for lxx1 in dimer_on_association12:
            if index_09 in lxx1:
                ltmp_dimer_on_association12.append(lxx1)
        for lxx11 in ltmp_dimer_on_association12:
            dimer_on_association12.remove(lxx11)
            
        for lsfd1 in pre_nucleus_on_association12:
            if index_09 in lsfd1:
                ltmp_pre_nucleus_on_association12.append(lsfd1)
        for lsfd2 in ltmp_pre_nucleus_on_association12:
            pre_nucleus_on_association12.remove(lsfd2)
            
        for lxse in fibril_association12:
            if (index_09 in lxse):
                ltmp_fibril_association12.append(lxse)
            elif (lEL in lxse):
                ltmp_fibril_association12.append(lxse)
        for lxsee in ltmp_fibril_association12:
            fibril_association12.remove(lxsee)
#        dimer_on_association12 = [lxx1 for lxx1 in dimer_on_association12 if not index_09 in lxx1]
#        dimer_off_association12 = [lxx2 for lxx2 in dimer_off_association12 if not index_09 in lxx2]
        for lxx2 in dimer_off_association12:
            if index_09 in lxx2:
                ltmp_dimer_off_association12.append(lxx2)
        for lxx22 in ltmp_dimer_off_association12:
            dimer_off_association12.remove(lxx22)
#        pre_nucleus_on_association12 = [lsdf for lsdf in pre_nucleus_on_association12 if not index_09 in lsdf]
#        fibril_association12 = [lxse for lxse in fibril_association12 if not index_09 in lxse]
#        fibril_association12 = [lxfd for lxfd in fibril_association12 if not lEL in lxfd]
#        second_monomer_attach12 = [lxvv for lxvv in second_monomer_attach12 if not index_09 in lxvv]
        for lxvv in second_monomer_attach12:
            if index_09 in lxvv:
                ltmp_second_monomer_attach12.append(lxvv)
        for lxvv1 in ltmp_second_monomer_attach12:
            second_monomer_attach12.remove(lxvv1)
#        second_fibril_association12 = [lxbv for lxbv in second_fibril_association12 if not index_09 in lxbv]
        for lxbv in second_fibril_association12:
            if index_09 in lxbv:
                ltmp_second_fibril_association12.append(lxbv)
        for lxbv1 in ltmp_second_fibril_association12:
            second_fibril_association12.remove(lxbv1)
#        second_pre_nucleus_association12 = [lxhn for lxhn in second_pre_nucleus_association12 if not index_09 in lxhn]  
        for lxhn in second_pre_nucleus_association12:
            if index_09 in lxhn:
                ltmp_second_pre_nucleus_association12.append(lxhn)
        for lxhn1 in ltmp_second_pre_nucleus_association12:
            second_pre_nucleus_association12.remove(lxhn1)
#        second_dimer_association12 = [lxk for lxk in second_dimer_association12 if not index_09 in lxk]   
        for lxhn21 in second_dimer_association12:
            if index_09 in lxhn21:
                ltmp_second_dimer_association12.append(lxhn21)
        for lxhn211 in ltmp_second_dimer_association12:
            second_dimer_association12.remove(lxhn211)
#            mono_inter_diffusion_on1 = [x for x in mono_inter_diffusion_on1 if not index_09 in x]
        if index_09 in mono_inter_diffusion_off12:
            mono_inter_diffusion_off12.remove(index_09) 
#        pre_nucleus_off_association12 = [lxmk for lxmk in pre_nucleus_off_association12 if not index_09 in lxmk]
        for lxmk in pre_nucleus_off_association12:
            if index_09 in lxmk:
                ltmp_pre_nucleus_off_association12.append(lxmk)
        for lxmk1 in ltmp_pre_nucleus_off_association12:
            pre_nucleus_off_association12.remove(lxmk1)
            
        for lxx56 in pentamer_on_diffusion12:
            if lEL in lxx56:
                ltmp_pentamer_on_diffusion12.append(lxx56)
        for lxx66 in ltmp_pentamer_on_diffusion12:
            pentamer_on_diffusion12.remove(lxx66)
            
        DEL[lEL][2] = 1   
        ltmp_high.append(121)
        ltmp_high.append(lslope_fib_high)
        DEL[index_09][:] = ltmp_high[:]

        for lgtr1 in fibril_dissociation12:
            if lgtr1 == lEL:
                ltmp_fibril_dissociation12.append(lgtr1)
        for lkut2 in ltmp_fibril_dissociation12:
            fibril_dissociation12.remove(lkut2)
            
        for lsf in ltmp_high_mono:
            fibril_association12.append(lsf) # = fibril_association1 + tmp_high_mono
#        fibril_dissociation12.remove(lEL)
        fibril_dissociation12.append(index_09)
        for ldg in ltmp_high_mono_inter:
            mono_inter_diffusion_on12.append(ldg) # = mono_inter_diffusion_on1 + tmp_high_mono_inter
        for ldgg in ltmp_second_high:
            second_monomer_attach12.append(ldgg)

    ltmp_high  = []  
    ltmp_high_mono = []
    ltmp_high_mono_inter = []
    ltmp_fibril_association12 = []
    ltmp_dimer_on_association12 = []
    ltmp_pre_nucleus_on_association12 = []
    ltmp_fibril_dissociation12 = []
    ltmp_dimer_off_association12 = []
    ltmp_second_monomer_attach12 = []
    ltmp_second_fibril_association12 = []
    ltmp_second_pre_nucleus_association12 = []
    ltmp_pre_nucleus_off_association12 = []
    ltmp_second_high = []
    ltmp_second_dimer_association12 = []
    ltmp_pentamer_on_diffusion12 = []
    return (DEL)
############################################
    
#############################################  elongates a second nucleation fibril #####################################
#WITH PARTICLE ID AS END and FIBRIL ID AS SLOPE
def elongation_second(JSEC,index_10,sigma,msec,monomers13,monos_inter_diffusion_on13, monos_inter_diffusion_off13,
                    dimer_on_association13, dimer_off_association13, pre_nucleus_on_association13,
                    fibril_association13, second_monomer_attach13, second_fibril_association13, second_fibril_dissociation13,
                    second_pre_nucleus_association13, second_dimer_association13, pre_nucleus_off_association13,
                    pre_nucleus_on_dissociation13,second_monomer_detach13, second_pre_nucleus_dissociation13,
                    fibril_dissociation13,STOP_TEST13):
    mslope_sec_el = JSEC[msec][3]  
    mtmp_sec_el  = []  
    mtmp_sec_mono = []
    mtmp_sec_mono_inter = []
    ntmp_dimer_on_association13 = []
    ntmp_dimer_off_association13 = []
    ntmp_pre_nucleus_on_association13 = []
    ntmp_fibril_association13 = []
    ntmp_second_monomer_attach13 = []
    ntmp_second_fibril_association13 = []
    ntmp_second_pre_nucleus_association13 = []
    ntmp_second_dimer_association13 = []
    ntmp_pre_nucleus_off_association13 = []
    mtest_dist_elong_sec = False
    mtmp_sec_el.append(JSEC[msec][0] + 2.0*sigma*math.cos(math.atan(mslope_sec_el)))
    mtmp_sec_el.append(JSEC[msec][1] + 2.0*sigma*math.sin(math.atan(mslope_sec_el)))
    if mtmp_sec_el[0] > 0:            
        mtmp_sec_el[0] = mtmp_sec_el[0]%boxSide
    if mtmp_sec_el[0] < 0:
        mtmp_sec_el[0] = boxSide - abs(mtmp_sec_el[0])%boxSide
    if mtmp_sec_el[1] > 0:
        mtmp_sec_el[1] = mtmp_sec_el[1]%boxSide
    if mtmp_sec_el[1] < 0:
        mtmp_sec_el[1] = boxSide - abs(mtmp_sec_el[1])%boxSide      
#    mtest_dist_elong_sec = [dist(mtmp_sec_el, mi) for mi in JSEC]
    for mmi in range(len(JSEC)):
        if (mmi != index_10) and (mmi != msec):
            mtest_dist_elong_low = round(dist(mtmp_sec_el, JSEC[mmi]),13)
            if mtest_dist_elong_low < round(2.0*sigma,13):
                mtest_dist_elong_sec = True
                STOP_TEST13.append(0)
#                print("NOT PASSED")
                return (JSEC)
            elif mtest_dist_elong_low >= round(2.0*sigma,13) and mtest_dist_elong_low <= inter_range_on:
                mtmp_sec_mono.append([mmi,index_10])
                mtmp_sec_mono_inter.append(mmi)
    if mtest_dist_elong_sec == False:  # checking for overlap    
        STOP_TEST13.append(1)
#        print("PASSED")
        monomers13.remove(index_10) # = [mxjj for mxjj in monomers13 if not index_10 in mxjj]
        if index_10 in monos_inter_diffusion_on13: # removing the first free monomer from previous reaction lists
            monos_inter_diffusion_on13.remove(index_10)
            
#        dimer_on_association13 = [mxew for mxew in dimer_on_association13 if not index_10 in mxew]
        for mxew in dimer_on_association13:
            if index_10 in mxew:
                ntmp_dimer_on_association13.append(mxew)
        for mxeww in ntmp_dimer_on_association13:
            dimer_on_association13.remove(mxeww)
#        dimer_off_association13 = [mxqw for mxqw in dimer_off_association13 if not index_10 in mxqw]
        for mxqw in dimer_off_association13:
            if index_10:
                ntmp_dimer_off_association13.append(mxqw)
        for mxqww in ntmp_dimer_off_association13:
            dimer_off_association13.remove(mxqww)
            
        for mxer in pre_nucleus_on_association13:
            if index_10 in mxer:
                ntmp_pre_nucleus_on_association13.append(mxer)
        for mxerr in ntmp_pre_nucleus_on_association13:
            pre_nucleus_on_association13.remove(mxerr)
            
#        pre_nucleus_on_association13 = [mxer for mxer in pre_nucleus_on_association13 if not index_10 in mxer]
#        fibril_association13 = [mxee for mxee in fibril_association13 if not index_10 in mxee]
        for mxee in fibril_association13:
            if index_10 in mxee:
                ntmp_fibril_association13.append(mxee)
        for mxeee in ntmp_fibril_association13:
            fibril_association13.remove(mxeee)
#        second_monomer_attach13 = [mxet for mxet in second_monomer_attach13 if not index_10 in mxet]
        for mxet in second_monomer_attach13:
            if index_10 in mxet:
                ntmp_second_monomer_attach13.append(mxet)
        for mxett in ntmp_second_monomer_attach13:
            second_monomer_attach13.remove(mxet)
#        second_fibril_association13 = [mxty for mxty in second_fibril_association13 if not index_10 in mxty]
        for mxty in second_fibril_association13:
            if index_10 in mxty:
                ntmp_second_fibril_association13.append(mxty)
            elif msec in mxty:
                ntmp_second_fibril_association13.append(mxty)
        for mxtyy in ntmp_second_fibril_association13:
            second_fibril_association13.remove(mxtyy)
#        second_fibril_association13 = [mxiu for mxiu in second_fibril_association13 if not msec in mxiu]
#        second_pre_nucleus_association13 = [mxtg for mxtg in second_pre_nucleus_association13 if not index_10 in mxtg]  
        for mxtg in second_pre_nucleus_association13:
            if index_10 in mxtg:
                ntmp_second_pre_nucleus_association13.append(mxtg)
        for mxtgg in ntmp_second_pre_nucleus_association13:
            second_pre_nucleus_association13.remove(mxtgg)
#        second_dimer_association13 = [mxyh for mxyh in second_dimer_association13 if not index_10 in mxyh]   
        for mxyh in second_dimer_association13:
            if index_10 in mxyh:
                ntmp_second_dimer_association13.append(mxyh)
        for mxyhh in ntmp_second_dimer_association13:
            second_dimer_association13.remove(mxyhh)
#            monos_inter_diffusion_on1 = [x for x in monos_inter_diffusion_on1 if not index_10 in x]
        if index_10 in monos_inter_diffusion_off13:
            monos_inter_diffusion_off13.remove(index_10) 
#        pre_nucleus_off_association13 = [mxnb for mxnb in pre_nucleus_off_association13 if not index_10 in mxnb]
        for mxnb in pre_nucleus_off_association13:
            if index_10 in mxnb:
                ntmp_pre_nucleus_off_association13.append(mxnb)
        for mxnbb in ntmp_pre_nucleus_off_association13:
            pre_nucleus_off_association13.remove(mxnbb)
        JSEC[msec][2] = 30   
        mtmp_sec_el.append(33)
        mtmp_sec_el.append(mslope_sec_el)
        JSEC[index_10][:] = mtmp_sec_el[:]
        second_fibril_dissociation13.remove(msec)
        for mmg in mtmp_sec_mono:
            second_fibril_association13.append(mmg) #= second_fibril_association1 + tmp_sec_mono
        second_fibril_dissociation13.append(index_10)
        for mtg in mtmp_sec_mono_inter:
            monos_inter_diffusion_on13.append(mtg) # = monos_inter_diffusion_on1 + tmp_sec_mono_inter
            
    mtmp_sec_el  = []  
    mtmp_sec_mono = []
    mtmp_sec_mono_inter = []
    ntmp_dimer_on_association13 = []
    ntmp_dimer_off_association13 = []
    ntmp_pre_nucleus_on_association13 = []
    ntmp_fibril_association13 = []
    ntmp_second_monomer_attach13 = []
    ntmp_second_fibril_association13 = []
    ntmp_second_pre_nucleus_association13 = []
    ntmp_second_dimer_association13 = []
    ntmp_pre_nucleus_off_association13 = []

    return (JSEC)
############################################ end second elongation finction ###########################################
    
    
############################################ Detatch a monomer from an aggregate ########################################
## The funstion is detaching a monomer from the end of an aggregate:
# the monomer to be detached is n, and it becomes a monmomer
# all possible interactions for the newly formed monomer n are registered in tmp lists
# all possible interactions for the new end of the aggregate are registered in tmp lists
def monomer_fibril_detatch(MM,sigma,n, diff_length, mono_inter_diffusion_on3, mono_inter_diffusion_off3,
                    dimer_on_association3, dimer_off_association3, pre_nucleus_on_association3,
                    fibril_association3, second_monomer_attach3, second_fibril_association3, 
                    second_pre_nucleus_association3, second_dimer_association3, pre_nucleus_off_association3,
                    monomers3,pre_nucleus_on_dissociation3,fibril_dissociation3, dimer_on_diffusion3,
                    second_fibril_dissociation3, second_pre_nucleus_dissociation3, dimer_off_diffusion3,
                    pre_nucleus_off_dissociation3, trimer_on_diffusion3, trimer_off_diffusion3,
                    tetramer_on_diffusion3, tetramer_off_diffusion3, pentamer_on_diffusion3,
                    pentamer_off_diffusion3,dimer_off_dissociation3,dimer_on_dissociation3,second_monomer_detach3,
                    second_fibril_detach3,nucleus_off_numb3,STOP_TEST3):
    naggregate = [n] # the list that will contain all the monomers in the aggregate, with the aggregate to be dissocated as firsl entry in the list
#    print('dissociating a dimer or prenucleus from routine!!!',n, MM[n][2], MM[n])#, MM[n])
#    print("pre nucleus off dissociation", pre_nucleus_off_dissociation3)
    naggregate_left = [] #list of all the other monomers;.?
    nslope_ID = MM[n][3] # slope of the aggregate to be dissociated
    
    nnext_pos = [0] # it should have length 1    
    nfurther_pos = [0] #hods the other end of the aggregtae
#    ncenter_off = []
    

# find the aggregate that is going to be dissociated 
    for niy1 in range(0,len(MM)):
        if niy1 != n:
            if MM[niy1][3] == nslope_ID:
               naggregate.append(niy1) # the entire aggregate
#               print("naggregtae is:", niy1)
    naggregate_left = naggregate[:] 
    naggregate_left.remove(n)
#    print("naggregate_left",naggregate_left)
    
#    for nnvv in naggregate_left:
#        if nnvv == n:
#            ntmp_aggregate.append(nnvv)
#    for nnvv1 in ntmp_aggregate:
#        naggregate_left.remove(nnvv1) # aggregate_left has no n in it
#    print('nnaggregate, naagregate_left',naggregate, naggregate_left)
# find the next position in the aggregtae that has to become the new end in case n can be removed
    if MM[n][2] in [101,102,121,122,303,33,3020,3010]:
        nnext_pos_val = naggregate_left[0]    
        nfurther_pos[0] = naggregate_left[0]
        nnext_dist = dist(MM[naggregate_left[0]], MM[n])
        nnext_further_dist = dist(MM[naggregate_left[0]], MM[n])
        for naggr in naggregate_left:
            ndist_next = dist(MM[n], MM[naggr])
            if ndist_next <= nnext_dist:
                nnext_pos_val = naggr
                nnext_dist = ndist_next
            if ndist_next >= nnext_further_dist:
                nfurther_pos[0] = naggr
                nnext_further_dist = ndist_next
#                if MM[naggr][2] in [102,101]:
#                    nfurther_pos[0] = naggr
        nnext_pos[0] = nnext_pos_val
#        print('nnext_pos', nnext_pos[0])
    elif MM[n][2] in [20,21,22,23,24,25]:
        for nbt5 in naggregate:
            if MM[nbt5][2] == 20:
                ncenter_off = nbt5
#        print("center off", ncenter_off,MM[ncenter_off])


###################################################################################################################################################################################################################################################
###################################################################################################################################################################################################################################################    
# thare are 3 possible types of aggregates to be dissociated:
    # pre nuclei bigger than dimer and smaller than 6
    # a dimer into 2 monomers
    # a monomer from a side of the fibril that is a secondary nucleation    
    if len(naggregate) > 2:
        ntest_dist_detach = False # distance check for the aggregates >2
        ntmp_mono_on1 = [] # find free monomers  within interaction range in pathway
        ntmp_mono_off1 = [] # find free monomers within interaction range off pathway
        ntmp_mono_pre_on_nucleus1 = [] # find pre nucleus on ends monomers within interaction range
        ntmp_mono_fibrils1 = [] # find fibril ends monomers within interaction range
        ntmp_mono_second_mono_1 = [] # find fibril monomer within interaction range for secondary nucleation
        ntmp_mono_second_fibril_1 = [] #find second fibril end within interaction range for secondary fibirl elongation
        ntmp_mono_second_dimer_1 = [] #find second pre nucleus end within interaction range 
#        ntmp_mono_off_1 = [] # find off pre nucleus within interaction range
        ntmp_mono_second_pre_1 = [] # find areaction pair for a secondary nucleation pre nucleus formation
        ntmp_mono_inter_on1 = []# find monomers that are within interaction on range, since they diffuse slower
        ntmp_mono_inter_off1 = []# find monomers that are within interaction off range, since they diffuse slower
        ntmp1_pre_nucleus_on_association1 = []# tmp list to hold the interaction pairs that can lead to pre nucleus elongation
        ntmp_tetramer_on_diffusion1 = []# tmp list to hold the entire tertramer able to diffuse
        ntmp_pre_nucleus_on_dissociation1 = [] # tmp list to hold the monomers to be dissociated
        ntmp_pre_nucleus_on_association1 = [] # tmp list to hold the interaction pairs that can lead to pre nucleus elongation
        ntmp_fibril_association1 = [] # tmp list to hold the interaction pairs that can lead to fibril elongation
        ntmp1_fibril_dissociation1 = [] # tmp list to hold the monomers for fibril dissociation
        ntmp1_pre_nucleus_on_dissociation1 = []# tmp list to hold the monomers to be dissociated for a tetramer dissociating
        ntmp_pentamer_on_diffusion1 = []# tmp list to hold the entire fibril pentamer able to diffuse 
        ntmp_trimer_on_diffusion1 = []# tmp list to hold the entire on trimer able to diffuse
        ntmp_second_fibril_association1 = []
        ntmp_tetramer_off_diffusion1 = []
        ntmp_second_pre_nucleus_dissociation1 = []
        ntmp_trimer_off_diffusion1 = []
        ntmp_second_pre_nucleus_association1 = []
        ntmp_fibril_dissociation1 = []
        ntmp_second_pre_nucleus_dissociation1_1 = []
        ntmp_pre_nucleus_off_dissociation1 = []
        ntmp_pre_off1 = []
        ntmp_pentamer_off_diffusion1 = []
        ntmp_nucleus_off_numb1 = []
        # finding the new position of the dissociating monomer n
        ntmp_mono = []
        nangle_diff = random.uniform(-math.pi*2.0, math.pi*2.0) # chosing a random angle        
        ntmp_mono.append(MM[n][0] + math.cos(nangle_diff)*diff_length) # chosing a position within a difusion length of a monomer
        if ntmp_mono[0] > 0:
            ntmp_mono[0] = ntmp_mono[0]%boxSide
        elif ntmp_mono[0] < 0:
            ntmp_mono[0] = boxSide - abs(ntmp_mono[0])%boxSide
        ntmp_mono.append(MM[n][1] + math.sin(nangle_diff)*diff_length)
        if ntmp_mono[1] > 0:
            ntmp_mono[1] = ntmp_mono[1]%boxSide
        elif ntmp_mono[1] < sigma:
            ntmp_mono[1] = boxSide - abs(ntmp_mono[1])%boxSide

        for niq in range(len(MM)):
            if (niq != nnext_pos[0])and (niq != n):
                ntest_dist_mono = round(dist(ntmp_mono, MM[niq]),13) # the new formed free monomer
                ntest_dist_next_pos = round(dist(MM[nnext_pos[0]],MM[niq]),13) # the new end of aggregate
                if MM[n][2] in [20,21,22,23,24,25]:
                   ntest_dist_next_off = round(dist(MM[ncenter_off],MM[niq]),13)
                if ntest_dist_mono < round(2.0*sigma,13):
                    ntest_dist_detach = True
                    STOP_TEST3.append(0)
#                    print("NOT PASSED")
#                    print("dist in dissociation is", ntest_dist_mono, niq,ntest_dist_detach )
#                    break
                    return (MM)
#                elif MM[niq][2] in [12, 21, 22, 23, 24, 3010, 3020, 300,31, 32, 30]:
#                    pass
                elif  MM[niq][2] == 0:
                    if (ntest_dist_mono >= round(2.0*sigma,13)) and (ntest_dist_mono < inter_range_on):                
                        ntmp_mono_on1.append([niq,n]) # temporary list to append to dimer on 
                        ntmp_mono_inter_on1.append(niq)
                        ntmp_mono_inter_on1.append(n)
                        
                    if (ntest_dist_mono >= round(2.0 * sigma,13)) and (ntest_dist_mono < inter_range_off):
                        ntmp_mono_off1.append([niq,n])
                        ntmp_mono_inter_off1.append(niq)
                        ntmp_mono_inter_off1.append(n)
                        
                    if (ntest_dist_next_pos >= round(2.0*sigma,13)) and (ntest_dist_next_pos < inter_range_on):
                        if (len(naggregate) > 5) and (MM[naggregate[0]][2] in [122,121]):
                            ntmp_mono_fibrils1.append([niq,nnext_pos[0]])
                            ntmp_mono_inter_on1.append(niq)
                        elif (len(naggregate) == 5) and (MM[naggregate[0]][2] in [122,121]):
                            ntmp_mono_pre_on_nucleus1.append([niq,nnext_pos[0]]) 
                            ntmp_mono_inter_on1.append(niq)                        
                        elif (len(naggregate) < 5) and (MM[naggregate[0]][2] in [101,102]):
                            ntmp_mono_pre_on_nucleus1.append([niq,nnext_pos[0]]) 
                            ntmp_mono_inter_on1.append(niq)
                        elif (len(naggregate) < 5) and (MM[naggregate[0]][2] == 303):
                            ntmp_mono_second_pre_1.append([niq,nnext_pos[0]]) 
                            ntmp_mono_inter_on1.append(niq)
                            
                    elif MM[n][2] in [20,21,22,23,24,25]:
                        if (ntest_dist_next_off >= round(2.0 * sigma,13)) and (ntest_dist_next_off < inter_range_off):
                            ntmp_pre_off1.append([niq, ncenter_off])
                    
                    
                        
                elif (MM[niq][2] in [101, 102]) and (ntest_dist_mono >= round(2.0*sigma,13)) and (ntest_dist_mono < inter_range_on):
                    ntmp_mono_pre_on_nucleus1.append([n, niq])
                elif (MM[niq][2] in [121, 122]) and (ntest_dist_mono >= round(2.0*sigma,13)) and (ntest_dist_mono < inter_range_on):
                    ntmp_mono_fibrils1.append([n, niq])
                elif (MM[niq][2] == 1) and (ntest_dist_mono >= round(2.0*sigma,13)) and (ntest_dist_mono < inter_range_on):
                    ntmp_mono_second_mono_1.append([n, niq])            
#                elif MM[niq][2] == 33 and ntest_dist_mono >= round(2.0*sigma,13) and ntest_dist_mono < inter_range_on:
#                    ntmp_mono_second_fibril_1.append([n, niq])
                elif MM[niq][2] in [301, 302] and ntest_dist_mono >= round(2.0*sigma,13) and ntest_dist_mono < inter_range_on:
                    ntmp_mono_second_dimer_1.append([n, niq])
                elif MM[niq][2] == 303 and ntest_dist_mono >= round(2.0*sigma,13) and ntest_dist_mono < inter_range_on:
                    ntmp_mono_second_pre_1.append([n, niq])   
                elif MM[niq][2] == 20 and ntest_dist_mono >= round(2.0*sigma,13) and ntest_dist_mono < inter_range_off:
                    ntmp_pre_off1.append([n, niq])          
                
    # changing old coordinates for new ones and updating relevant population/reaction lists
        if ntest_dist_detach == False:
            STOP_TEST3.append(1)
#            print("PASSED")
            if (MM[n][2] == 101) or (MM[n][2] == 102) or(MM[n][2] == 122) or (MM[n][2] == 121) or (MM[n][2] == 303) or (MM[n][2] == 33):
                ntmp_fibril_dissociation1 = []
                if dist(MM[n],MM[nnext_pos[0]]) >= (1.e-06 - 2.0*sigma*len(naggregate)):
                    if (MM[n][2] == 101) or (MM[n][2] == 102) or (MM[n][2] == 121) or (MM[n][2] == 122):
                        if len(nfurther_pos) != 0:
                            MM[nfurther_pos[0]][2] = MM[n][2]
                    elif (MM[n][2] == 303) or (MM[n][2] == 33):
                        MM[nfurther_pos[0]][2] = MM[n][2]
                    ntmp_mono.append(0)
                    ntmp_mono.append(0)
                    MM[n][:] = ntmp_mono[:]
                elif dist(MM[n],MM[nnext_pos[0]]) <= (1.e-06 - 2.0*sigma*len(naggregate)):
                    if (MM[n][2] == 101) or (MM[n][2] == 102) or (MM[n][2] == 121) or (MM[n][2] == 122):
                        if len(nnext_pos) != 0:
                            MM[nnext_pos[0]][2] = MM[n][2]
                    elif (MM[n][2] == 303) or (MM[n][2] == 33):
                        MM[nnext_pos[0]][2] = MM[n][2]
                    ntmp_mono.append(0)
                    ntmp_mono.append(0)
                    MM[n][:] = ntmp_mono[:]
            if MM[n][2] in [20,21,22,23,24,25]:
                ntmp_mono.append(0)
                ntmp_mono.append(0)
                MM[n][:] = ntmp_mono[:]

# from trimer dissociated to a dimer
# lists changed: pre_nucleus_on_association  the monomer end, n, and all its possible interactions is being removed
                # trimer on diffusion list is loosing the trimer to be dissociated
                # pre nucleus on dissociation is loosing the ends of the trimer
                # the dimer  difusion is gaining the new dimer
                # pre nucleus on association is gaining the new end's association list .............................. look at the end of this if >2
                # dimer on dissociation is gainig the two new endings
                # monomers is gaining n, the dissociated monomer
                                
            if len(naggregate) == 3:
#                print("trimer disociation")
                if (MM[naggregate_left[0]][2] in [102, 101,12]):                    

#                    print("pre nucleus on dissociation in trimer dissociation",pre_nucleus_on_dissociation3)
#                    pre_nucleus_on_dissociation3.remove(n)
                    for nttrx1 in naggregate:
                        if nttrx1 in pre_nucleus_on_dissociation3:
                            ntmp_pre_nucleus_on_dissociation1.append(nttrx1)
                    for nttrx2 in ntmp_pre_nucleus_on_dissociation1:
                        pre_nucleus_on_dissociation3.remove(nttrx2)
                        

                    for ntrx2 in trimer_on_diffusion3:
                            if (n in ntrx2) or (naggregate_left[0] in ntrx2) or (naggregate_left[1] in ntrx2):
                                ntmp_trimer_on_diffusion1.append(ntrx2) # remove the dissocated trimer from the trimer diffusion list                    
                    for ntrx22 in ntmp_trimer_on_diffusion1:
                        trimer_on_diffusion3.remove(ntrx22)

                    for ntrx3 in pre_nucleus_on_association3:        
                        for ntrx331 in naggregate:
                            if ntrx331 in ntrx3:
                                ntmp_pre_nucleus_on_association1.append(ntrx3) # removing the the pre nucleus from the assoctioan list
                    for ntrx33 in ntmp_pre_nucleus_on_association1:
                        pre_nucleus_on_association3.remove(ntrx33)
                            
                    dimer_on_dissociation3.append(naggregate_left[0]) # append the end of the new dimer to the dimer dissociation list
                    dimer_on_dissociation3.append(naggregate_left[1])
                    dimer_on_diffusion3.append(naggregate_left) #appending the left dimer to the dimer diffusion list
#                    monomers3.append(n) #append the dissociated monomer to the monomer list
                if MM[naggregate_left[0]][2] in [303,300,3020,3010]: #3/12/20

                    for nxm15 in second_pre_nucleus_dissociation3:
                        if nxm15 == n:
                            ntmp_second_pre_nucleus_dissociation1.append(nxm15)
                    for nxm155 in ntmp_second_pre_nucleus_dissociation1:
                        second_pre_nucleus_dissociation3.remove(nxm155)
                        
                    for nfgr in second_pre_nucleus_association3:
                        if n in nfgr:
                            ntmp_second_pre_nucleus_association1.append(nfgr)
                    for nfgrr in ntmp_second_pre_nucleus_association1:
                        second_pre_nucleus_association3.remove(nfgrr)

                    second_pre_nucleus_dissociation3.append(nnext_pos[0])

                
                if MM[naggregate_left[0]][2] in [20,21,22,23,24,25]:
#                    print("diss a off trimer", MM[n])
                    for ntrx4 in pre_nucleus_off_dissociation3:
                            if n in ntrx4:
                                ntmp_pre_nucleus_off_dissociation1.append(ntrx4)
#                                print("nnvt",ntrx4)
                    for ntrx44 in ntmp_pre_nucleus_off_dissociation1:
#                        print("ntrx44",ntrx44)
                        pre_nucleus_off_dissociation3.remove(ntrx44)
                        
                    for ntrx76 in trimer_off_diffusion3:
                        if n in ntrx76:
                            ntmp_trimer_off_diffusion1.append(ntrx76)
                    for ntrx766 in ntmp_trimer_off_diffusion1:
                        trimer_off_diffusion3.remove(ntrx766)

                    dimer_off_diffusion3.append(naggregate_left)         
                    dimer_off_dissociation3.append(ncenter_off)

#                    print("new dimer off from dissociation", naggregate_left[0],naggregate_left[1], MM[naggregate_left[0]], MM[naggregate_left[1]])
                    
            elif len(naggregate) == 4:
#                print("tetramer dissociasion")
#                print("MM[naggregate[0]][2]",MM[naggregate_left[0]][2])
                if MM[naggregate_left[0]][2] in [102, 101,12]:
                    trimer_on_diffusion3.append(naggregate_left)   # adding the new formed trimer to the list of trimer on diffusion
                    
                    for ntrx5 in tetramer_on_diffusion3:                        
                            if (n in ntrx5) or (naggregate_left[0] in ntrx5):
#                                print('ntrx6 is', ntrx5)
                                ntmp_tetramer_on_diffusion1.append(ntrx5) # removing the dissociated tetramer from the tetramer on diffusion
                    for ntrx55 in ntmp_tetramer_on_diffusion1:
                        tetramer_on_diffusion3.remove(ntrx55)

                    for ntrx8 in pre_nucleus_on_association3:
                        for ntrx881 in naggregate:
                            if ntrx881 in ntrx8:
                                ntmp1_pre_nucleus_on_association1.append(ntrx8) # removing all interation partners with the dissociated end
                    for ntrx88 in ntmp1_pre_nucleus_on_association1:
                        pre_nucleus_on_association3.remove(ntrx88)
                                        
                    for nhgt21 in pre_nucleus_on_dissociation3:
                        if nhgt21 == n:
                            ntmp1_pre_nucleus_on_dissociation1.append(nhgt21)
                    for nhgt22 in ntmp1_pre_nucleus_on_dissociation1:
                        pre_nucleus_on_dissociation3.remove(nhgt22)

                    pre_nucleus_on_dissociation3.append(nnext_pos[0])                    
                    
                if MM[naggregate_left[0]][2] in [303,300,3020,3010]: #3/12/20
                    
                    for nxm115 in second_pre_nucleus_dissociation3:
                        if nxm115 == n:
                               ntmp_second_pre_nucleus_dissociation1_1.append(nxm115)
                    for nxm1155 in ntmp_second_pre_nucleus_dissociation1_1:
                        second_pre_nucleus_dissociation3.remove(nxm1155)
                    
                    for nfggr in second_pre_nucleus_association3:
                        if n in nfggr:
                            ntmp_second_pre_nucleus_association1.append(nfggr)
                    for nfggrr in ntmp_second_pre_nucleus_association1:
                        second_pre_nucleus_association3.remove(nfggrr)

                    second_pre_nucleus_dissociation3.append(nnext_pos[0])
#                    monomers3.append(n) # adding the dissociated monomer to the list
                if MM[naggregate_left[0]][2] in [20,21,22,23,24,25]:
                    
                    for ngftr4 in pre_nucleus_off_dissociation3:
                            if n in ngftr4:
                                ntmp_pre_nucleus_off_dissociation1.append(ngftr4)
                    for ngftr44 in ntmp_pre_nucleus_off_dissociation1:
                        pre_nucleus_off_dissociation3.remove(ngftr44)
                    
                    for ngfty76 in tetramer_off_diffusion3:
                        if n in ngfty76:
                            ntmp_tetramer_off_diffusion1.append(ngfty76)
                    for ngfty766 in ntmp_tetramer_off_diffusion1:
                        tetramer_off_diffusion3.remove(ngfty766)
                    
                    pre_nucleus_off_dissociation3.append(naggregate_left)
                    trimer_off_diffusion3.append(naggregate_left)

                    
            elif len(naggregate) == 5:
#                print("pentamer dissociation")
                if MM[naggregate_left[0]][2] in [122, 121,1]:
                    tetramer_on_diffusion3.append(naggregate_left)
                    for ntrx9 in pentamer_on_diffusion3:
                        if n in ntrx9:             
                                ntmp_pentamer_on_diffusion1.append(ntrx9) # removing the old dissociated pentamer
                    for ntrx99 in ntmp_pentamer_on_diffusion1:
                        pentamer_on_diffusion3.remove(ntrx99)
                                
                    for nrl1 in naggregate:
                        if nrl1 in fibril_dissociation3:
                            ntmp_fibril_dissociation1.append(nrl1)
                    for nrl11 in ntmp_fibril_dissociation1:
                        fibril_dissociation3.remove(nrl11)
                        
                    for nxm12 in second_fibril_association3:
                        for nxm121 in naggregate:
                            if nxm121 in nxm12:
                                ntmp_second_fibril_association1.append(nxm12)
                    for nxm122 in ntmp_second_fibril_association1:
                        second_fibril_association3.remove(nxm122)

                    for nbrf23 in naggregate_left:
                        if MM[nbrf23][2] == 122:
                            MM[nbrf23][2] == 102
                        elif MM[nbrf23][2] == 121:
                            MM[nbrf23][2] == 101
                        elif MM[nbrf23][2] == 1:
                            MM[nbrf23][2] == 12
#                    monomers3.append(n)
                    for nssq1 in naggregate_left:
                        if (MM[nssq1][2] == 102) or (MM[nssq1][2] == 101):                            
                            pre_nucleus_on_dissociation3.append(nssq1)          
                            
                            
                if MM[naggregate_left[0]][2] in [33,30,31,32]: #3/12/20
                    second_fibril_dissociation3.remove(n)
                    for nbfry in naggregate_left:
                        if MM[nbfry][2] == 33:
                            MM[nbfry][2] == 303
                        elif MM[nbfry][2] == 30:
                            MM[nbfry][2] == 300
                        elif MM[nbfry][2] == 31:
                            MM[nbfry][2] == 3010
                        elif MM[nbfry][2] == 32:
                            MM[nbfry][2] == 3020

                    for nzqy in naggregate:
                        for nzqyy in second_fibril_detach3:
                            if nzqy in nzqyy:
                                second_fibril_detach3.remove(nzqyy)
                    for nskli in naggregate_left:
                        if (MM[nskli][2] == 303):                   
                            second_pre_nucleus_dissociation3.append(nskli)          
                            
                if MM[naggregate_left[1]][2] in [20,21,22,23,24,25]:
                    for nafyu1 in pre_nucleus_off_dissociation3:
                            if n in nafyu1:
                                ntmp_pre_nucleus_off_dissociation1.append(nafyu1)
                    for nafyu11 in ntmp_pre_nucleus_off_dissociation1:
                        pre_nucleus_off_dissociation3.remove(nafyu11)
                        
                    for nhfty76 in pentamer_off_diffusion3:
                        if n in nhfty76:
                            ntmp_pentamer_off_diffusion1.append(nhfty76)
                    for nhfty766 in ntmp_pentamer_off_diffusion1:
                        pentamer_off_diffusion3.remove(nhfty766)
                        
                    pre_nucleus_off_dissociation3.append(naggregate_left)
                    tetramer_off_diffusion3.append(naggregate_left)
            elif len(naggregate) == 6:
#                
                if MM[naggregate_left[0]][2] in [122, 121,1]:
                    for ntrx13q in fibril_dissociation3:
                        if ntrx13q == n:
                            ntmp1_fibril_dissociation1.append(ntrx13q)
                    for ntrx16q in ntmp1_fibril_dissociation1:
                        fibril_dissociation3.remove(n)
                            
                    for ntrx14q in fibril_association3:
                        if n in ntrx14q:
                            ntmp_fibril_association1.append(ntrx14q)
                    for ntrx19q in ntmp_fibril_association1:
                        fibril_association3.remove(ntrx19q)
                    pentamer_on_diffusion3.append(naggregate_left)
                    fibril_dissociation3.append(nnext_pos[0])
#                    monomers3.append(n)
                if MM[naggregate_left[0]][2] in [20,21,22,23,24,25]:

                    for nalj1 in pre_nucleus_off_dissociation3:
                            if n in nalj1:
                                ntmp_pre_nucleus_off_dissociation1.append(nalj1)
                    for nalj2 in ntmp_pre_nucleus_off_dissociation1:
                        pre_nucleus_off_dissociation3.remove(nalj2)
                        
                    for nolk in nucleus_off_numb3:
                        if n in nolk:
                            ntmp_nucleus_off_numb1.append(nolk)
                    for nolk1 in ntmp_nucleus_off_numb1:
                        nucleus_off_numb3.remove(nolk1)
                    
                    pentamer_off_diffusion3.append(naggregate_left)
                    
            elif len(naggregate) > 6:

                    for ntrx13 in fibril_dissociation3:
                        if ntrx13 == n:
                            ntmp1_fibril_dissociation1.append(ntrx13)
                    for ntrx16 in ntmp1_fibril_dissociation1:
                        fibril_dissociation3.remove(ntrx16)
                            
                    for ntrx14 in fibril_association3:
                        for ntrx141 in naggregate:
                            if ntrx141 in ntrx14:
                                ntmp_fibril_association1.append(ntrx14)
                    for ntrx19 in ntmp_fibril_association1:
                        fibril_association3.remove(ntrx19)
                        
                    fibril_dissociation3.append(nnext_pos[0])


            monomers3.append(n)
            for ntj0 in ntmp_mono_inter_off1:           
                mono_inter_diffusion_off3.append(ntj0)# = mono_inter_diffusion_off1 + tmp_mono_inter_off   # adding to the list of monomers within interaction range for difusion    
            for ngc0 in ntmp_mono_on1:
                dimer_on_association3.append(ngc0)# = dimer_on_association1 + tmp_mono_on # adding to the dimer_on_association list 
            for nvb5 in ntmp_mono_inter_on1:                
                mono_inter_diffusion_on3.append(nvb5)# = mono_inter_diffusion_on1 + tmp_mono_inter_on 
            for ncv8 in ntmp_mono_pre_on_nucleus1:
                pre_nucleus_on_association3.append(ncv8)#= pre_nucleus_on_association1 + tmp_mono_pre_on_nucleus # adding to pre nucleus on association list            
            for nkm9 in ntmp_mono_fibrils1:
                fibril_association3.append(nkm9)# = fibril_association1 + tmp_mono_fibrils # adding to the fibril association list
            for nmh4 in ntmp_mono_second_mono_1:
                second_monomer_attach3.append(nmh4)# = second_monomer_attach1 + tmp_mono_second_mono_2# adding to the  secondary monomer attach list
            for nms1 in ntmp_mono_second_fibril_1:
                second_fibril_association3.append(nms1)# = second_fibril_association1 + tmp_mono_second_fibril_2 # adding to the secondary fibril elongation list
            for ndh6 in ntmp_mono_second_pre_1:
                second_pre_nucleus_association3.append(ndh6)# = second_pre_nucleus_association1 + tmp_mono_second_pre_2 # adding  to the secondary pre nucleus association list
            for nns7 in ntmp_mono_second_dimer_1:
                second_dimer_association3.append(nns7)# = second_dimer_association1 + tmp_mono_second_pre_2 # adding to the secondary dimer association list                            
            for nrf01 in ntmp_pre_off1:
                pre_nucleus_off_association3.append(nrf01)# = pre_nucleus_off_association1 + tmp_mono_off_2 # adding to the pre nucleus off association list
            for nxlav in ntmp_mono_off1:
                dimer_off_association3.append(nxlav)
        ntmp_mono_off1 = [] # find free monomers within interaction range off pathway
        ntmp_mono_pre_on_nucleus1 = [] # find pre nucleus on ends monomers within interaction range
        ntmp_mono_fibrils1 = [] # find fibril ends monomers within interaction range
        ntmp_mono_second_mono_1 = [] # find fibril monomer within interaction range for secondary nucleation
        ntmp_mono_second_fibril_1 = [] #find second fibril end within interaction range for secondary fibirl elongation
        ntmp_mono_second_dimer_1 = [] #find second pre nucleus end within interaction range 
#        ntmp_mono_second_pre_1 = [] # find areaction pair for a secondary nucleation pre nucleus formation
        ntmp_mono_inter_on1 = []# find monomers that are within interaction on range, since they diffuse slower
        ntmp_mono_inter_off1 = []# find monomers that are within interaction off range, since they diffuse slower
        ntmp1_pre_nucleus_on_association1 = []# tmp list to hold the interaction pairs that can lead to pre nucleus elongation
        ntmp_tetramer_on_diffusion1 = []# tmp list to hold the entire tertramer able to diffuse
        ntmp_mono_fibrils1 = [] # find fibril ends monomers within interaction range
        ntmp_mono_second_fibril_1 = [] #find second fibril end within interaction range for secondary fibirl elongation
#        ntmp_mono_off1 = [] # find off pre nucleus within interaction range
        ntmp_mono_second_pre_1 = [] # find areaction pair for a secondary nucleation dimer formation
        ntmp_pre_nucleus_on_dissociation1 = [] # tmp list to hold the monomers to be dissociated
        ntmp_pre_nucleus_on_association1 = [] # tmp list to hold the interaction pairs that can lead to pre nucleus elongation
        ntmp_fibril_association1 = [] # tmp list to hold the interaction pairs that can lead to fibril elongation
        ntmp1_fibril_dissociation1 = [] # tmp list to hold the monomers for fibril dissociation
        ntmp1_pre_nucleus_on_dissociation1 = []# tmp list to hold the monomers to be dissociated for a tetramer dissociating
        ntmp_pentamer_on_diffusion1 = []# tmp list to hold the entire fibril pentamer able to diffuse 
        ntmp_trimer_on_diffusion1 = []# tmp list to hold the entire on trimer able to diffuse
#        ntmp_second_dimer_association1 = []
#        ntmp_second_fibril_dissociation1 =[]
        ntmp_second_pre_nucleus_dissociation1 = []
        ntmp_pre_off1 = []
        ntmp_trimer_off_diffusion1 = []
#        ntmp_pre_nucleus_off_association1 = []
#        ntmp_second_monomer_attach1 = []
#        ntmp_pre_nucleus_off_dissociation1 = []
        ntmp_second_pre_nucleus_association1 = []
        ntmp_second_pre_nucleus_dissociation1_1 = []
        ntmp_mono = []
        naggregate = [] # the list that will contain all the monomers in the aggregate, with the aggregate to be dissocated as firsl entry in the list
        naggregate_left = [] #list of all the other monomers;.?
        ntmp_pre_nucleus_off_dissociation1 = []


###############################################################################################
####################################################################################################################################################################################################################################################
    elif len(naggregate) == 2: # a dimer dissociates into 2 monomers
#        print("dimer on or off dissociation")
#        print("the dimer is", naggregate[0], naggregate[1], MM[naggregate[0]], MM[naggregate[1]])
        ntmp_dimer_on_diffusion2 = []
        ntp_dimer_on_dissociation2 = []
        ntmp_mono_on2 = [] # find free monomers  within interaction range in pathway
        ntmp_mono_off2 = [] # find free monomers within interaction range off pathway
        ntmp_mono_pre_on_nucleus2 = [] # find pre nucleus on ends monomers within interaction range
        ntmp_mono_fibrils2 = [] # find fibril ends monomers within interaction range

        ntmp_mono_second_mono_2 = [] # find fibril monomer within interaction range for secondary nucleation
        ntmp_mono_second_fibril_2 = [] #find second fibril end within interaction range for secondary fibirl elongation
        ntmp_mono_second_dimer_2 = [] #find second pre nucleus end within interaction range 
        ntmp_mono_off_2 = [] # find off pre nucleus within interaction range
        ntmp_mono_second_pre_2 = [] 
        ntmp_mono_inter_on2 = []
        ntmp_mono_inter_off2 = []
#        print ("on dimer dissociation")
        ntmp_mono_1 = []
        ntmp_mono_2 = []
        ntest_dist_mono_1 = []
        ntest_dist_mono_2 = []
        ntmp2_pre_nucleus_on_association2 = []
        ntmp_second_pre_nucleus_association2 = []
        ntmp_second_pre_nucleus_dissociation2 = []
        ntmp_second_monomer_detach2 = []
        ntmp_dimer_off_diffusion2 = []
        ntmp_pre_nucleus_off_association2 = []
        ntmp_pre_nucleus_off_dissociation2 = []
        ntmp_dimer_off_dissociation2 = []
        ntest_dist_pass = False
        
        nangle_diff_1 = random.uniform(-math.pi*2.0, math.pi*2.0) # chosing a random angle
        ntmp_mono_1.append(MM[naggregate[0]][0] + math.cos(nangle_diff_1)*diff_length) # chosing a position within a difusion length of a monomer
        if ntmp_mono_1[0] > 0:
            ntmp_mono_1[0] = ntmp_mono_1[0]%boxSide
        elif ntmp_mono_1[0] < 0:
            ntmp_mono_1[0] = boxSide - abs(ntmp_mono_1[0])%boxSide
        ntmp_mono_1.append(MM[naggregate[0]][1] + math.sin(nangle_diff_1)*diff_length)
        if ntmp_mono_1[1] > 0:
            ntmp_mono_1[1] = ntmp_mono_1[1]%boxSide
        elif ntmp_mono_1[1] < sigma:
            ntmp_mono_1[1] = boxSide - abs(ntmp_mono_1[1])%boxSide
        ntmp_mono_1.append(0)
        ntmp_mono_1.append(0)
        nangle_diff_2 = random.uniform(-math.pi*2.0, math.pi*2.0)
        ntmp_mono_2.append(MM[naggregate[1]][0] + math.cos(nangle_diff_2)*diff_length) # chosing a position within a difusion length of a monomer
        if ntmp_mono_2[0] > 0:
            ntmp_mono_2[0] = ntmp_mono_2[0]%boxSide
        elif ntmp_mono_2[0] < 0:
            ntmp_mono_2[0] = boxSide - abs(ntmp_mono_2[0])%boxSide
        ntmp_mono_2.append(MM[naggregate[1]][1] + math.sin(nangle_diff_2)*diff_length)
        if ntmp_mono_2[1] > 0:
            ntmp_mono_2[1] = ntmp_mono_2[1]%boxSide
        elif ntmp_mono_2[1] < 0:
            ntmp_mono_2[1] = boxSide - abs(ntmp_mono_2[1])%boxSide
        ntmp_mono_2.append(0)
        ntmp_mono_2.append(0)
        
        for niqq in range(len(MM)):            
            if (niqq != naggregate[0]) and (niqq != naggregate[1]):
                ntest_dist_mono_1 = round(dist(ntmp_mono_1, MM[niqq]),13)                
                ntest_dist_mono_2 = round(dist(ntmp_mono_2, MM[niqq]),13)   
                if( ntest_dist_mono_1 < round(2.0*sigma,13)) or (ntest_dist_mono_2 < round(2.0*sigma,13)):
                    ntest_dist_pass = True
                    STOP_TEST3.append(0)
#                    print("NOT PASSED")
    #                break
                    return (MM)
    #            elif MM[niqq][2] in [12, 21, 22, 23, 24, 3010, 3020, 300,31, 32, 30]:
    #                    pass
                elif  MM[niqq][2] == 0:
                    if  (niqq != naggregate[0]) and (ntest_dist_mono_1 >= round(2.0*sigma,13)) and (ntest_dist_mono_1 < inter_range_on):                
                        ntmp_mono_on2.append([niqq,naggregate[0]]) 
                        ntmp_mono_inter_on2.append(niqq)
                    elif (niqq != naggregate[0]) and (ntest_dist_mono_1 >= round(2.0*sigma,13)) and (ntest_dist_mono_1 < inter_range_off):
                            ntmp_mono_off2.append([niqq,naggregate[0]])
                            ntmp_mono_inter_off2.append(niqq)
                    if (niqq != naggregate[1]) and (ntest_dist_mono_2 >= round(2.0*sigma,13)) and (ntest_dist_mono_2 < inter_range_on):
                            ntmp_mono_on2.append([niqq,naggregate[1]]) 
                            ntmp_mono_inter_on2.append(niqq)
                    elif (niqq != naggregate[1]) and (ntest_dist_mono_2 >= round(2.0*sigma,13)) and (ntest_dist_mono_2 < inter_range_off):
                            ntmp_mono_off2.append([niqq,naggregate[1]])
                            ntmp_mono_inter_off2.append(niqq)                        
                elif MM[niqq][2] in [101, 102]:
                    if (niqq != naggregate[0]) and (ntest_dist_mono_1 >= round(2.0*sigma,13)) and (ntest_dist_mono_1 < inter_range_on):
                        ntmp_mono_pre_on_nucleus2.append([naggregate[0],niqq])
                    if (niqq != naggregate[1]) and (ntest_dist_mono_2 >= round(2.0*sigma,13)) and (ntest_dist_mono_2 < inter_range_on):
                        ntmp_mono_pre_on_nucleus2.append([naggregate[1],niqq])
                elif MM[niqq][2] in [121, 122]:
                    if (niqq != naggregate[0]) and (ntest_dist_mono_1 >= round(2.0*sigma,13)) and (ntest_dist_mono_1 < inter_range_on):
                        ntmp_mono_fibrils2.append([naggregate[0],niqq])
                    if (niqq != naggregate[1]) and (ntest_dist_mono_2 >= round(2.0*sigma,13)) and (ntest_dist_mono_2 < inter_range_on):
                        ntmp_mono_fibrils2.append([naggregate[1],niqq])
                elif MM[niqq][2] == 1:
                    if (ntest_dist_mono_1 >= round(2.0*sigma,13)) and (ntest_dist_mono_1 < inter_range_on):
                        ntmp_mono_second_mono_2.append([naggregate[0],niqq])    
                    if (ntest_dist_mono_2 >= round(2.0*sigma,13)) and (ntest_dist_mono_2 < inter_range_on):
                        ntmp_mono_second_mono_2.append([naggregate[1],niqq])                         
    
                elif MM[niqq][2] in [301, 302]:
                    if (niqq != naggregate[0]) and (ntest_dist_mono_1 >= round(2.0*sigma,13)) and (ntest_dist_mono_1 < inter_range_on):
                        ntmp_mono_second_dimer_2.append([naggregate[0],niqq])
                    if (niqq != naggregate[1]) and (ntest_dist_mono_2 >= round(2.0*sigma,13)) and (ntest_dist_mono_2 < inter_range_on):
                        ntmp_mono_second_dimer_2.append([naggregate[1],niqq])                        
                elif MM[niqq][2] == 303:
                    if (niqq != naggregate[0]) and (ntest_dist_mono_1 >= round(2.0*sigma,13)) and (ntest_dist_mono_1 < inter_range_on):
                        ntmp_mono_second_pre_2.append([naggregate[0],niqq])  
                    if (niqq != naggregate[1]) and (ntest_dist_mono_2 >= round(2.0*sigma,13)) and (ntest_dist_mono_2 < inter_range_on):
                        ntmp_mono_second_pre_2.append([naggregate[1],niqq])                         
                elif MM[niqq][2] == 20:
                    if (niqq != naggregate[0]) and (ntest_dist_mono_1 >= round(2.0*sigma,13)) and (ntest_dist_mono_1 < inter_range_off):
                        ntmp_mono_off_2.append([naggregate[0],niqq])    
                    if (niqq != naggregate[1]) and (ntest_dist_mono_2 >= round(2.0*sigma,13)) and (ntest_dist_mono_2 < inter_range_off):
                        ntmp_mono_off_2.append([naggregate[1],niqq])  
        if ntest_dist_pass == False:
            STOP_TEST3.append(1)
#            print("PASSED")
#            print('dimer on dissoc after', dimer_on_dissociation3)
            MM[naggregate[0]][:] = ntmp_mono_1[:]
#            print('in a dimer_dissociation', MM[naggregate[0]])
            MM[naggregate[1]][:] = ntmp_mono_2[:]
           
#            print('dimer on dissoc before', dimer_on_dissociation3)
            for nxq4 in dimer_off_diffusion3:
                if (naggregate[0] in nxq4) or (naggregate[1] in nxq4):
                    ntmp_dimer_off_diffusion2.append(nxq4)
            for nxq44 in ntmp_dimer_off_diffusion2:
                dimer_off_diffusion3.remove(nxq44)
#            dimer_off_diffusion3 = [nxq4 for nxq4 in dimer_off_diffusion3 if not naggregate[0] in nxq4]            
            for nxq5 in pre_nucleus_off_association3:
                if naggregate[0] in nxq5:
                    ntmp_pre_nucleus_off_association2.append(nxq5)
                elif naggregate[1] in nxq5:
                    ntmp_pre_nucleus_off_association2.append(nxq5)
            for nxq6 in ntmp_pre_nucleus_off_association2:
                pre_nucleus_off_association3.remove(nxq6)

            for nxq8 in pre_nucleus_off_dissociation3:
                if  naggregate[0] in nxq8:
                    ntmp_pre_nucleus_off_dissociation2.append(nxq8)
#                    print("pre nucleus off in dimer off diss", nxq8)
                elif naggregate[1] in nxq8:
                    ntmp_pre_nucleus_off_dissociation2.append(nxq8)
#                    print("pre nucleus off in dimer off diss", nxq8)
            for nxq7 in ntmp_pre_nucleus_off_dissociation2:
                pre_nucleus_off_dissociation3.remove(nxq7)

            for nxq9 in dimer_off_dissociation3:
                if nxq9 == naggregate[0]:
                    ntmp_dimer_off_dissociation2.append(nxq9)
#                    print("dimer off in dimer off diss", nxq9)
                elif nxq9 == naggregate[1]:
                    ntmp_dimer_off_dissociation2.append(nxq9)
#                    print("dimer off in dimer off diss routine", nxq9)
            for nxq10 in ntmp_dimer_off_dissociation2:
                dimer_off_dissociation3.remove(nxq10)
            
                        
            for nxq1 in pre_nucleus_on_association3:
                if naggregate[0] in nxq1:                    
                    ntmp2_pre_nucleus_on_association2.append(nxq1)
                elif naggregate[1] in nxq1:
                     ntmp2_pre_nucleus_on_association2.append(nxq1)
            for nxq22 in ntmp2_pre_nucleus_on_association2:
                pre_nucleus_on_association3.remove(nxq22)
            
#            dimer_on_diffusion3 = [nxq3 for nxq3 in dimer_on_diffusion3 if not naggregate_left[0] in nxq3]
            for nxq3 in dimer_on_diffusion3:
                if (naggregate[0] in nxq3) or (naggregate[1] in nxq3):
                    ntmp_dimer_on_diffusion2.append(nxq3)
            for nxq33 in ntmp_dimer_on_diffusion2:
                if nxq33 in dimer_on_diffusion3:
                    dimer_on_diffusion3.remove(nxq33)
            
            for nxq11 in second_pre_nucleus_association3:
                if naggregate[0] in nxq11:
                    ntmp_second_pre_nucleus_association2.append(nxq11)
                elif naggregate[1] in nxq11:
                    ntmp_second_pre_nucleus_association2.append(nxq11)
            for nxq111 in ntmp_second_pre_nucleus_association2:
                second_pre_nucleus_association3.remove(nxq111)
                
            for nxq12 in second_pre_nucleus_dissociation3:
                if nxq12 == naggregate[0]:
                    ntmp_second_pre_nucleus_dissociation2.append(nxq12)
                elif nxq12 == naggregate[1]:
                    ntmp_second_pre_nucleus_dissociation2.append(nxq12)
            for nxq121 in ntmp_second_pre_nucleus_dissociation2:
                second_pre_nucleus_dissociation3.remove(nxq121)
                
            for nxnx1 in second_monomer_detach3:
                if nxnx1 == naggregate[0]:
                    ntmp_second_monomer_detach2.apppend(nxnx1)
                elif nxnx1 == naggregate[1]:
                    ntmp_second_monomer_detach2.append(nxnx1)
            for nxnx11 in ntmp_second_monomer_detach2:
                second_monomer_detach3.remove(nxnx11)
            
            for ngbc1 in dimer_on_dissociation3:
                if ngbc1 == naggregate[0]: 
                    ntp_dimer_on_dissociation2.append(ngbc1)
                elif ngbc1 == naggregate[1]:
                    ntp_dimer_on_dissociation2.append(ngbc1)
#                    print('finding a end to be removed', ngbc1, MM[ngbc1])                    
            for ngbc11 in ntp_dimer_on_dissociation2:
                dimer_on_dissociation3.remove(ngbc11)
#                print('removing from dimer association in dimer dissociation routine', ngbc1, MM[ngbc1])
            monomers3.append(naggregate[0])
            monomers3.append(naggregate[1])
            
            for nit in ntmp_mono_inter_off2:
                mono_inter_diffusion_off3.append(nit) # = mono_inter_diffusion_off1 + tmp_mono_inter_off   # adding to the list of monomers within interaction range for difusion            
            for nhf in ntmp_mono_on2:
                dimer_on_association3.append(nhf)# = dimer_on_association1 + tmp_mono_on # adding to the dimer_on_association list 
            for nts in ntmp_mono_inter_on2:
                mono_inter_diffusion_on3.append(nts)# = mono_inter_diffusion_on1 + tmp_mono_inter_on 
            for ndg in ntmp_mono_pre_on_nucleus2:
                pre_nucleus_on_association3.append(ndg)# = pre_nucleus_on_association1 + tmp_mono_pre_on_nucleus # adding to pre nucleus on association list
            for nbc in ntmp_mono_fibrils2:
                fibril_association3.append(nbc)# = fibril_association1 + tmp_mono_fibrils # adding to the fibril association list
            for ncx in ntmp_mono_second_mono_2:
                second_monomer_attach3.append(ncx)# = second_monomer_attach1 + tmp_mono_second_mono_2# adding to the  secondary monomer attach list
            for nfv in ntmp_mono_second_fibril_2:
                second_fibril_association3.append(nfv)# = second_fibril_association1 + tmp_mono_second_fibril_2 # adding to the secondary fibril elongation list
            for nbf in ntmp_mono_second_pre_2:
                second_pre_nucleus_association3.append(nbf)# = second_pre_nucleus_association1 + tmp_mono_second_pre_2 # adding  to the secondary pre nucleus association list
            for ngv in ntmp_mono_second_dimer_2:
                second_dimer_association3.append(ngv)# = second_dimer_association1 + tmp_mono_second_pre_2 # adding to the secondary dimer association list                            
            for nbd in ntmp_mono_off_2:
                pre_nucleus_off_association3.append(nbd)# = pre_nucleus_off_association1 + tmp_mono_off_2 # adding to the pre nucleus off association list
            for nmpz in ntmp_mono_off2:
                dimer_off_association3.append(nmpz)
                
        ntmp_dimer_on_diffusion2 = []
        ntp_dimer_on_dissociation2 = []
        ntmp_mono_on2 = [] # find free monomers  within interaction range in pathway
        ntmp_mono_off2 = [] # find free monomers within interaction range off pathway
        ntmp_mono_pre_on_nucleus2 = [] # find pre nucleus on ends monomers within interaction range
        ntmp_mono_fibrils2 = [] # find fibril ends monomers within interaction range

        ntmp_mono_second_mono_2 = [] # find fibril monomer within interaction range for secondary nucleation
        ntmp_mono_second_fibril_2 = [] #find second fibril end within interaction range for secondary fibirl elongation
        ntmp_mono_second_dimer_2 = [] #find second pre nucleus end within interaction range 
        ntmp_mono_off_2 = [] # find off pre nucleus within interaction range
        ntmp_mono_second_pre_2 = [] 
        ntmp_mono_inter_on2 = []
        ntmp_mono_inter_off2 = []

        ntmp_mono_1 = []
        ntmp_mono_2 = []
        ntest_dist_mono_1 = []
        ntest_dist_mono_2 = []
        ntmp2_pre_nucleus_on_association2 = []
        ntmp_second_pre_nucleus_association2 = []
        ntmp_second_pre_nucleus_dissociation2 = []
        naggregate = [] # the list that will contain all the monomers in the aggregate, with the aggregate to be dissocated as firsl entry in the list
        naggregate_left = [] #list of all the other monomers;.? 
        ntmp_second_monomer_detach2 = []
        ntmp_dimer_off_diffusion2 = []
        ntmp_pre_nucleus_off_association2 = []
        ntmp_pre_nucleus_off_dissociation2 = []
        ntmp_dimer_off_dissociation2 = []


        
#####################################################################################################################################################################################################################################
#####################################################################################################################################################################################################################################                  
    elif len(naggregate) == 1: #thisis only for the case of a second monomer detaching from the fibril-parent
#        print("dissociating monomer------------in second-----")
        ntest_dist_pass_detach = False
    
        ntmp_mono_on3 = [] # find free monomers  within interaction range in pathway
        ntmp_mono_off3 = [] # find free monomers within interaction range off pathway
        ntmp_mono_pre_on_nucleus3 = [] # find pre nucleus on ends monomers within interaction range
        ntmp_mono_fibrils3 = [] # find fibril ends monomers within interaction range
        ntmp_mono_second_mono_3 = [] # find fibril monomer within interaction range for secondary nucleation
        ntmp_mono_second_fibril_3 = [] #find second fibril end within interaction range for secondary fibirl elongation
        ntmp_mono_second_dimer_3 = [] #find second pre nucleus end within interaction range 
        ntmp_mono_off_3 = [] # find off pre nucleus within interaction range
        ntmp_mono_second_pre_3 = [] 
        ntmp_mono_inter_on3 = []
        ntmp_mono_inter_off3 = []
        ntmp_mono_3 = []
        ntest_dist_mono_3 = []
        ntmp_second_monomer_detach3 = []
        ntmp_second_dimer_association3 = []
        
        nangle_diff_3 = random.uniform(-math.pi*2.0, math.pi*2.0) # chosing a random angle
        ntmp_mono_3.append(MM[n][0] + math.cos(nangle_diff_3)*diff_length) # chosing a position within a difusion length of a monomer
        if ntmp_mono_3[0] > 0:
            ntmp_mono_3[0] = ntmp_mono_3[0]%boxSide
        elif ntmp_mono_3[0] < 0:
            ntmp_mono_3[0] = boxSide - abs(ntmp_mono_3[0])%boxSide 
        ntmp_mono_3.append(MM[n][1] + math.sin(nangle_diff_3)*diff_length)
        if ntmp_mono_3[1] > 0:
            ntmp_mono_3[1] = ntmp_mono_3[1]%boxSide
        elif ntmp_mono_3[1] < 0:
            nnew_coord_31 = boxSide - abs(ntmp_mono_3[1])%boxSide 
            ntmp_mono_3[1] = nnew_coord_31
        ntmp_mono_3.append(0)
        ntmp_mono_3.append(0)
        for nii in range(len(MM)):
            if nii != n:
                ntest_dist_mono_3 = round(dist(ntmp_mono_3, MM[nii]),13) 
                if ntest_dist_mono_3 < round(2.0*sigma,13):
                    ntest_dist_pass_detach = True
                    STOP_TEST3.append(0)
#                    print("NOT PASSED")
    #                break
                    return (MM)
#                elif MM[nii][2] in [12, 21, 22, 23, 24, 3010, 3020, 300,31, 32, 30]:
#                        pass
                elif  MM[nii][2] == 0:
                        if ntest_dist_mono_3 >= round(2.0*sigma,13) and ntest_dist_mono_3 < inter_range_on:                
                            ntmp_mono_on3.append([n,nii]) # difusing monomer position in L, stationary monomer position in L
                            ntmp_mono_inter_on3.append(nii)
                            ntmp_mono_inter_on3.append(n)
                        if ntest_dist_mono_3 >= round(2.0*sigma,13) and ntest_dist_mono_3 < inter_range_off:
                            ntmp_mono_off3.append([n,nii])
                            ntmp_mono_inter_off3.append(nii)
                            ntmp_mono_inter_off3.append(n)
                elif MM[nii][2] in [101, 102] and ntest_dist_mono_3 >= round(2.0*sigma,13) and ntest_dist_mono_3 < inter_range_on:
                    ntmp_mono_pre_on_nucleus3.append([n,nii])
                elif MM[nii][2] in [121, 122] and ntest_dist_mono_3 >= round(2.0*sigma,13) and ntest_dist_mono_3 < inter_range_on:
                    ntmp_mono_fibrils3.append([n,nii])
                elif MM[nii][2] == 1 and ntest_dist_mono_3 >= round(2.0*sigma,13) and ntest_dist_mono_3 < inter_range_on:
                    ntmp_mono_second_mono_3.append([n,nii])            
    #            elif MM[nii][2] == 33 and ntest_dist_mono_3 >= round(2.0*sigma,13) and ntest_dist_mono_3 < inter_range_on:
    #                ntmp_mono_second_fibril_3.append([n,nii])
                elif MM[nii][2] in [301, 302] and ntest_dist_mono_3 >= round(2.0*sigma,13) and ntest_dist_mono_3 < inter_range_on:
                    ntmp_mono_second_dimer_3.append([n,nii])
                elif MM[nii][2] == 303 and ntest_dist_mono_3 >= round(2.0*sigma,13) and ntest_dist_mono_3 < inter_range_on:
                    ntmp_mono_second_pre_3.append([n,nii])   
                elif MM[nii][2] == 20 and ntest_dist_mono_3 >= round(2.0*sigma,13) and ntest_dist_mono_3 < inter_range_off:
                    ntmp_mono_off_3.append([n,nii])          
                
    # changing old coordinates for new ones and updating relevant population/reaction lists
        if ntest_dist_pass_detach == False:
            STOP_TEST3.append(1)
#            print("PASSED")
            for nxc3 in second_monomer_detach3:
                if nxc3 == n:
                    ntmp_second_monomer_detach3.append(nxc3)
            for nxc33 in ntmp_second_monomer_detach3:
                second_monomer_detach3.remove(nxc33)
#            pre_nucleus_off_association3 = [nxc2 for nxc2 in pre_nucleus_off_association3 if not n in nxc2]
#            second_monomer_attach3 = [nxc1 for nxc1 in second_monomer_attach3 if not nnext_pos in nxc1]
            for nxc0 in second_dimer_association3:
                if n in nxc0:
                    ntmp_second_dimer_association3.append(nxc0)
            for nxc01 in ntmp_second_dimer_association3:
                second_dimer_association3.remove(nxc01)
            MM[naggregate[0]][:] = ntmp_mono_3[:]
#            print('in a second mono_dissociation', MM[naggregate[0]])
            
            monomers3.append(n)           
            for nit0 in ntmp_mono_inter_off3:
                mono_inter_diffusion_off3.append(nit0) # = mono_inter_diffusion_off1 + tmp_mono_inter_off   # adding to the list of monomers within interaction range for difusion            
            for nhf0 in ntmp_mono_on3:
                dimer_on_association3.append(nhf0)# = dimer_on_association1 + tmp_mono_on # adding to the dimer_on_association list 
            for ntye in ntmp_mono_off3:
                dimer_off_association3.append(ntye)
            for nts0 in ntmp_mono_inter_on3:
                mono_inter_diffusion_on3.append(nts0)# = mono_inter_diffusion_on1 + tmp_mono_inter_on 
            for ndg0 in ntmp_mono_pre_on_nucleus3:
                pre_nucleus_on_association3.append(ndg0)# = pre_nucleus_on_association1 + tmp_mono_pre_on_nucleus # adding to pre nucleus on association list
            for nbc0 in ntmp_mono_fibrils3:
                fibril_association3.append(nbc0)# = fibril_association1 + tmp_mono_fibrils # adding to the fibril association list
            for ncx0 in ntmp_mono_second_mono_3:
                second_monomer_attach3.append(ncx0)# = second_monomer_attach1 + tmp_mono_second_mono_2# adding to the  secondary monomer attach list
            for nfv0 in ntmp_mono_second_fibril_3:
                second_fibril_association3.append(nfv0)# = second_fibril_association1 + tmp_mono_second_fibril_2 # adding to the secondary fibril elongation list
            for nbf0 in ntmp_mono_second_pre_3:
                second_pre_nucleus_association3.append(nbf0)# = second_pre_nucleus_association1 + tmp_mono_second_pre_2 # adding  to the secondary pre nucleus association list
            for ngv0 in ntmp_mono_second_dimer_3:
                second_dimer_association3.append(ngv0)# = second_dimer_association1 + tmp_mono_second_pre_2 # adding to the secondary dimer association list                            
            for nbd0 in ntmp_mono_off_3:
                pre_nucleus_off_association3.append(nbd0)# = pre_nucleus_off_association1 + tmp_mono_off_2 # adding to the pre nucleus off association list

        ntmp_mono_on3 = [] # find free monomers  within interaction range in pathway
        ntmp_mono_off3 = [] # find free monomers within interaction range off pathway
        ntmp_mono_pre_on_nucleus3 = [] # find pre nucleus on ends monomers within interaction range
        ntmp_mono_fibrils3 = [] # find fibril ends monomers within interaction range
        ntmp_mono_second_mono_3 = [] # find fibril monomer within interaction range for secondary nucleation
        ntmp_mono_second_fibril_3 = [] #find second fibril end within interaction range for secondary fibirl elongation
        ntmp_mono_second_dimer_3 = [] #find second pre nucleus end within interaction range 
        ntmp_mono_off_3 = [] # find off pre nucleus within interaction range
        ntmp_mono_second_pre_3 = [] 
        ntmp_mono_inter_on3 = []
        ntmp_mono_inter_off3 = []
        ntmp_mono_3 = []
        ntest_dist_mono_3 = []
        ntmp_second_monomer_detach3 = []
        ntmp_second_dimer_association3 = []
        naggregate = [] # the list that will contain all the monomers in the aggregate, with the aggregate to be dissocated as firsl entry in the list
        naggregate_left = [] #list of all the other monomers;.?

        
    return (MM)
############################################ end monomer dissociation function #######################################
    
    
#############################################  detatches a secondary fibril from the side of a fibril and starts a fibril WITH PARTICLE ID AS END and FIBRIL ID AS SLOPE
def detatch_second(VV,sigma,o, otrans_diff_length, second_fibril_association14,second_fibril_dissociation14,
                   pentamer_on_diffusion14,fibril_dissociation14,fibril_association14,
                   second_monomer_attach14,second_fibril_detach14, STOP_TEST14):     # if there are no overlaps with other particles    
#    print("second detach routine", VV[o[0]])
    oslope_new_fibril =  VV[o[0]][3]
#    print("slope is second",oslope_new_fibril)
    ofibril_new = []
    otmp_fibril_dissociation14 = []
    otmp_fibril_association14 = []
    otmp_second_monomer_attach14 = []
#    ofree_secondary_site = []
    otest_dist_second_detach = False
    otmp_second_fibril_dissociation14 = []
    tmp_second_fibril_detach14 = []
    
#    for i in range(len(VV)):
#        if VV[i][3] == slope_new_fibril:
#           fibril_new.append(i) 
#    ovisc = 0.92e-3
#    op2 = len(o)
#    otemp = 323.15 # temperature in kelvin, K
#    ok_b = 1.38064852e-23 # boltzman constant in J/K
#    oniu = math.log(op2) + 0.312 + 0.565/op2 - 0.1/(op2**2)
#    otrans_diff_length = ((ok_b*otemp)/(3.0*math.pi*ovisc*op2*2.0*sigma))*oniu 
#    print("diif length second fibril",otrans_diff_length)
    
    for odd1 in o:
        ofibril_new.append(VV[odd1][:])
#        print("ofibril",VV[odd1])
#    print("")
    for odd in ofibril_new:        
        odd[0] = odd[0] + otrans_diff_length*math.cos(math.atan(oslope_new_fibril))
        odd[1] = odd[1] + otrans_diff_length*math.sin(math.atan(oslope_new_fibril))
        
    for odf1 in ofibril_new:
        
        if odf1[0] > 0:            
            odf1[0] = odf1[0]%boxSide
        if odf1[0] < 0:
#            print("go one lower than box!", djj[0])
            odf1[0] = boxSide - abs(odf1[0])%boxSide
        if odf1[1] > 0:
            odf1[1] = odf1[1]%boxSide
        if odf1[1] < 0:
#            print("go one higher than box!", djj[1])
            odf1[1] = boxSide - abs(odf1[1])%boxSide    
        
    on_count = 0
    for osk1 in ofibril_new:
        for ojb1 in range(len(VV)):
            if ojb1 not in o:
                otest_dist_detach = round(dist(osk1,VV[ojb1]),13)
                if otest_dist_detach < round(2.0*sigma,13):
                    otest_dist_second_detach = True
                    STOP_TEST14.append(0)
    #                break
                    return (VV)
                elif osk1[2] in [31,32,33]:
                    otmp_fibril_dissociation14.append(o[on_count])
                    if( VV[ojb1][2] == 0) and (otest_dist_detach >= round(2.0*sigma,13)) and (otest_dist_detach <= inter_range_on):
                        otmp_fibril_association14.append([ojb1,o[on_count]])
                elif (osk1[2] == 30) and (VV[ojb1][2] == 0) and (otest_dist_detach >= round(2.0*sigma,13))  and (otest_dist_detach <= inter_range_on):
                    otmp_second_monomer_attach14.append([ojb1,o[on_count]])
        on_count = on_count + 1

    if otest_dist_second_detach == False:          
        STOP_TEST14.append(1)
#        print("detaching a second fibril !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", VV[o[0]],ofibril_new[0])
            
        for odf in ofibril_new:
            if oslope_new_fibril < 0:
                if odf[2] == 32:
                    odf[2] = 121
                    for osd in ofibril_new:
                        if osd[2] == 33:
                            osd[2] = 122
                
                elif odf[2] == 31:
                    odf[2] = 122
                    for osd in ofibril_new:
                        if osd[2] == 33:
                            osd[2] = 121                        
                elif odf[2] == 30:
                    odf[2] = 1
                            
            elif oslope_new_fibril > 0:
                if odf[2] == 32:
                    odf[2] = 122
                    for osd in ofibril_new:
                        if osd[2] == 33:
                            osd[2] = 121
                elif odf[2] == 31:
                    odf[2] = 122
                    for osd in ofibril_new:
                        if osd[2] == 33:
                            osd[2] = 121       
                elif odf[2] == 30:
                    odf[2] = 1        
                    
        for okzx in range(0,len(ofibril_new)):
            VV[o[okzx]] = ofibril_new[okzx]
                    
        for oxb1 in second_fibril_dissociation14:
            if oxb1 == o[0]:
                otmp_second_fibril_dissociation14.append(oxb1)
        for oxb11 in otmp_second_fibril_dissociation14:
            second_fibril_dissociation14.remove(oxb11)
        
        for objt in second_fibril_detach14:
            if (o[0] in objt) or (o[1] in objt):
                tmp_second_fibril_detach14.append(objt)
        for objt1 in tmp_second_fibril_detach14:
            second_fibril_detach14.remove(objt1)
        
        if len(ofibril_new) == 5:
            pentamer_on_diffusion14.append(o)
        for otg in otmp_fibril_dissociation14:
            fibril_dissociation14.append(otg)# = fibril_dissociation1 + tmp_fibril_dissociation1
        for ovd in otmp_fibril_association14:
            fibril_association14.append(ovd)#= fibril_association1 + tmp_fibril_association1
        for otr in otmp_second_monomer_attach14:
            second_monomer_attach14.append(otr)# = second_monomer_attach1 + tmp_second_monomer_attach1
            
    otmp_second_fibril_dissociation14 = []
    otmp_fibril_dissociation14 = []
    otmp_fibril_association14 = []
    otmp_second_monomer_attach14 = []
    tmp_second_fibril_detach14 = []
    ofibril_new = []
#    ofibril_new_1 = []
    return (VV)
############################################ end secondary fibril detach ###############################################
    


#############################################  moves an aggregate below a 5mer a distance proportional to its difusion length
def aggregate_diffusion(VA,sigma,p,trans_diffusion_length, pre_nucleus_on_association15,
                        pre_nucleus_off_association15,fibril_association15,second_monomer_attach15,monos_inter_diffusion_on15,
                        monos_inter_diffusion_off15,dimer_off_association15,STOP_TEST15):     # if there are no overlaps with other particles    
    pslope_old_aggregate = VA[p[0]][3]
#    print('aggregate diffusiong', p,VA[p[0]])
    pp = []
    pp1 = []
#    VA_copy = []
    ptmp_pre_nucleus_on_association15 = []
    ptmp_pre_nucleus_off_association15 =[]
    ptmp_fibril_association15 = []
    ptmp15_pre_nucleus_on_association15 = []
    pend_positions = []
    ptmp15_fibril_association15 = []
    ptmp_second_mono_association15 = []
    ptmp_mono_diffusion_on15 = []
    ptmp_mono_diffusion_off15 = []
    ptmp15_pre_nucleus_off_association15 = []
    ptmp15_dimer_off_association15 = []
    ptmp_second_monomer_attach15_1 = []

#    print("VA_copy initiallength",len(VA_copy))
    for pyu in range(len(VA)):
#        VA_copy.append(VA[pyu])
        if VA[pyu][3] == pslope_old_aggregate:
            pp.append(VA[pyu][:]) # getting all the particles belonging to the aggregate
            pp1.append(pyu)
#            VA_copy.remove(VA[pyu])
#    print(len(VA), len(VA_copy))

            
    ptmp_agg = [] # holds the temporary new position where aggregate would diffuse
#    print("difusing an aggr")
    pdist_agg_test = []
    ptest_dist_mono_pass = False    

    sign_diff = random.choice([-1, 1])  #select randomly between two possible directions to diffuse along the slope line
    for pdj in range (len(pp)):        
        ptmp_agg.append([pp[pdj][0] + sign_diff*trans_diffusion_length*math.cos(math.atan(pslope_old_aggregate)),\
                        pp[pdj][1] + sign_diff*trans_diffusion_length*math.sin(math.atan(pslope_old_aggregate)),\
                        pp[pdj][2],pp[pdj][3]])
    
    ppn1 = 0
    for pij in ptmp_agg:
#        print([pij])
        if pij[2] == 101 or pij[2] == 102:
            pend_positions.append(pp1[ppn1])
#            print('end positions',pend_positions)
        if pij[2] == 121 or pij[2] == 122:
            pend_positions.append(pp1[ppn1])
#            print('end positions',pend_positions)

    for ppfk in pp1:
        if VA[ppfk][2] == 20:
            pcenter_off = ppfk
            
    for pdjj in ptmp_agg:
        if pdjj[0] > 0:            
            pdjj[0] = pdjj[0]%boxSide
        if pdjj[0] < 0:
#            print("go one lower than box!", djj[0])
            pdjj[0] = boxSide - abs(pdjj[0])%boxSide
        if pdjj[1] > 0:
            pdjj[1] = pdjj[1]%boxSide
        if pdjj[1] < 0:
#            print("go one higher than box!", djj[1])
            pdjj[1] = boxSide - abs(pdjj[1])%boxSide            

#    print("after adjustment", tmp_agg)      
    pnn = 0
    for pxi in ptmp_agg:
        for p1 in range(len(VA)): 
            if p1 not in pp1:
                pdist_agg_test = round(dist(pxi, VA[p1]),13) 
                if pdist_agg_test < round(2.0*sigma,13):
                    ptest_dist_mono_pass = True
                    STOP_TEST15.append(0)
#                    print("NOT PASSED")
#                    break
                    return (VA)
                elif pdist_agg_test > round(2.0*sigma,13) and pdist_agg_test <= inter_range_on:
                    if VA[p1][2] == 0 and ((pxi[2] == 101) or  (pxi[2] == 102)):
                        ptmp_pre_nucleus_on_association15.append([p1,pp1[pnn]])
                        ptmp_mono_diffusion_on15.append(p1)
                    elif VA[p1][2] == 0 and ((pxi[2] == 121)  or (pxi[2] == 122)):
                        ptmp_fibril_association15.append([p1,pp1[pnn]])   
                        ptmp_mono_diffusion_on15.append(p1)
                    elif VA[p1][2] == 0 and (pxi[2] == 1):
                        ptmp_second_mono_association15.append([p1,pp1[pnn]])   
                        ptmp_mono_diffusion_on15.append(p1)
                elif pdist_agg_test > round(2.0*sigma,13) and pdist_agg_test <= inter_range_off:       
                    if (pxi[2] == 20) and (len(pp) < nuc_size) and (VA[p1][2] == 0):
                        ptmp_pre_nucleus_off_association15.append([p1,pp1[pnn]])
                        ptmp_mono_diffusion_off15.append(p1)
        pnn = pnn + 1
            
    if ptest_dist_mono_pass == False:
        STOP_TEST15.append(1)
#        print("PASSED")
        if VA[p[0]][2] in [101,102,122,121,1,12]:
            if VA[pend_positions[0]][2] in [101,102]:
                if VA[pend_positions[0]][1] > VA[pend_positions[1]][1]:
                    VA[pend_positions[0]][2] = 101
                    VA[pend_positions[1]][2] = 102
                elif VA[pend_positions[0]][1] < VA[pend_positions[1]][1]:
                    VA[pend_positions[0]][2] = 102
                    VA[pend_positions[1]][2] = 101
            elif VA[pend_positions[0]][2] in [121,122]:
                if VA[pend_positions[0]][1] > VA[pend_positions[1]][1]:
                    VA[pend_positions[0]][2] = 121
                    VA[pend_positions[1]][2] = 122
                elif VA[pend_positions[0]][1] < VA[pend_positions[1]][1]:
                    VA[pend_positions[0]][2] = 122
                    VA[pend_positions[1]][2] = 121
            for pdjk in range (len(pp)):
                VA[pp1[pdjk]][:] = ptmp_agg[pdjk][:]   
    #            print(pdjk, VA[pp1[pdjk]])
            
        
            if len(pp) <=5:
                for pxz05 in pre_nucleus_on_association15:
                    if (pend_positions[0] in pxz05) or (pend_positions[1] in pxz05 ):
                        ptmp15_pre_nucleus_on_association15.append(pxz05)
                for pxz06 in ptmp15_pre_nucleus_on_association15:
                    pre_nucleus_on_association15.remove(pxz06)
    
    #            pre_nucleus_off_association15 = [pxz03 for pxz03 in pre_nucleus_off_association15 if not pend_positions[0] in pxz03] 
                for pxz07 in fibril_association15:
                    if (pend_positions[0] in pxz07) or (pend_positions[1] in pxz07 ):
                        ptmp15_fibril_association15.append(pxz07)
                for pxz08 in ptmp15_fibril_association15:
                    fibril_association15.remove(pxz08)
                    
                for pxz09 in p:
                    if VA[pxz09][2] == 1:
                        for pxz109 in second_monomer_attach15:
                            if pxz09 in pxz109:
                                ptmp_second_monomer_attach15_1.append(pxz109)
                for pxz099 in ptmp_second_monomer_attach15_1:
                    second_monomer_attach15.remove(pxz099)

        if VA[p[0]][2] in [20,21,22,23,24,25]:
            if len(pp) <nuc_size:    
                for pxz050 in pre_nucleus_off_association15:
                    if pcenter_off in pxz050:
                        ptmp15_pre_nucleus_off_association15.append(pxz050)
                for pxz060 in ptmp15_pre_nucleus_off_association15:
                    pre_nucleus_off_association15.remove(pxz060)
                    
                for pxz0550 in dimer_off_association15:
                    if pcenter_off in pxz0550:
                        ptmp15_dimer_off_association15.append(pxz0550)
                for pxz0660 in ptmp15_dimer_off_association15:
                    dimer_off_association15.remove(pxz0660)

        for ptvf in ptmp_pre_nucleus_on_association15:
            pre_nucleus_on_association15.append(ptvf)# = pre_nucleus_on_association1 + tmp_pre_nucleus_on_association1
        for pbgf in ptmp_pre_nucleus_off_association15:
            pre_nucleus_off_association15.append(pbgf)# = pre_nucleus_off_association1 + tmp_pre_nucleus_off_association1
        for pgef in ptmp_fibril_association15:
            fibril_association15.append(pgef)# = fibril_association1 + tmp_fibril_association1
        for phbt in ptmp_second_mono_association15:
            second_monomer_attach15.append(phbt)
        for potr in ptmp_mono_diffusion_on15:
            monos_inter_diffusion_on15.append(potr) #monomers within on pathway range interaction
        for pitu in ptmp_mono_diffusion_off15:
            monos_inter_diffusion_off # monomers within off pathway interaction
    ptmp_pre_nucleus_on_association15 = []
    ptmp_pre_nucleus_off_association15 =[]
    ptmp_fibril_association15 = []
    ptmp15_pre_nucleus_on_association15 = []
    pend_positions = []
    ptmp15_fibril_association15 = []
    ptmp_second_mono_association15 = []
    ptmp15_pre_nucleus_off_association15 = []
    ptmp15_dimer_off_association15 = []
    ptmp_second_monomer_attach15_1 = []

    return (VA)
############################################ end of diffusion function ##################################################
    

################################## determining the diffusion coefficient based on aggregation length ##################
def diff_coeff_aggregat(listFibril):
    p2 = len(listFibril)
    niu = math.log(p2) + 0.312 + 0.565/p2 - 0.1/(p2**2)
    diffCoeffAggr = ((k_b*temp)/(3.0*math.pi*visc*p2*2.0*sigma))*niu
    return diffCoeffAggr
########################################################################################
    
    
####################### Building the event table ###############################################################################
################################################################################################################################
################################################################################################################################    
#L = direct_disks_box(N, sigma)
colors_1 = []
nuc_size = 6

if os.path.isfile('monomers.txt'):
    with open('monomers.txt', 'r') as fff:
        monomers_dynamics = json.loads(fff.read())
else:
    monomers_dynamics = []
    
        
if os.path.isfile('pre_on_nuclei.txt'):
    with open('pre_on_nuclei.txt', 'r') as f_f:
        pre_on_nuclei_dynamics = json.loads(f_f.read())
else:
    pre_on_nuclei_dynamics = []

if os.path.isfile('pre_off_nuclei.txt'):
    with open('pre_off_nuclei.txt', 'r') as f_ff:
        pre_off_nuclei_dynamics = json.loads(f_ff.read())
else:
    pre_off_nuclei_dynamics = []

if os.path.isfile('fibrils.txt'):
    with open('fibrils.txt', 'r') as ff_fff:
        fibrils_dynamics = json.loads(ff_fff.read())
else:
    fibrils_dynamics = []
    
if os.path.isfile('off_nuclei.txt'):
    with open('off_nuclei.txt', 'r') as ff_ff:
        off_nuclei_dynamics = json.loads(ff_ff.read())
else:
    off_nuclei_dynamics = []
    
if os.path.isfile('time.txt'):
    with open('time.txt', 'r') as ff:
        time_dynamics = json.loads(ff.read())
        time = time_dynamics[len(time_dynamics)-1]
else:
    time_dynamics = []
    time = 0.0
#print("time dynamics is ", time_dynamics)
#print("monomers is ", monomers_dynamics)
    
######################################## Making the list of all the monomers

monomers = avail_monos(L)

for i5 in range(N):
    for j5 in range(N):
        if i5 != j5:
            distance = dist(L[i5],L[j5])
            if distance < inter_range_on:
                dimer_on_association.append([i5,j5])
                if i5 not in monos_inter_diffusion_on:
                    monos_inter_diffusion_on.append(i5)                
            if distance < inter_range_off:
                dimer_off_association.append([i5,j5])
#                print("dimer off",[i5,j5])
                if i5 not in monos_inter_diffusion_off:
                    monos_inter_diffusion_off.append(i5)
              
#########################################################################

for th in range(iter_nums):
#    print('dimer off ', dimer_off_association)
#    print('dimer on ', dimer_on_association)
    sum_interval = 0
    intervals = [0.00]
    action = []
    if len(dimer_on_association) > 0:
        sum_interval = sum_interval + len(dimer_on_association)*kinetic_rates['dimer_on_association']
        intervals.append(sum_interval)
        action.append('dimer_on_association')       
        
    if len(dimer_off_association) > 0:
        sum_interval = sum_interval + len(dimer_off_association)*kinetic_rates['dimer_off_association']
        intervals.append(sum_interval)
        action.append('dimer_off_association')
        
    if len(dimer_on_dissociation) > 0:
        sum_interval = sum_interval + len(dimer_on_dissociation)*kinetic_rates['pre_nucleus_on_dissociation']
        intervals.append(sum_interval)
        action.append('dimer_on_dissociation')
        
    if len(dimer_off_dissociation) > 0:
        sum_interval = sum_interval + 2.0*len(dimer_off_dissociation)*kinetic_rates['dimer_off_dissociation']
        intervals.append(sum_interval)
        action.append('dimer_off_dissociation')    
    
    if len(pre_nucleus_on_association) > 0:
        sum_interval = sum_interval + len(pre_nucleus_on_association)*kinetic_rates['pre_nucleus_on_association']
        intervals.append(sum_interval)
        action.append('pre_nucleus_on_association')
        
    if len(pre_nucleus_off_association) > 0:
        sum_interval = sum_interval + 2.0*len(pre_nucleus_off_association)*kinetic_rates['pre_nucleus_off_association'] 
        intervals.append(sum_interval)
        action.append('pre_nucleus_off_association')
        
    if len(pre_nucleus_on_dissociation) > 0:
        sum_interval = sum_interval + len(pre_nucleus_on_dissociation)*kinetic_rates['pre_nucleus_on_dissociation']
        intervals.append(sum_interval)
        action.append('pre_nucleus_on_dissociation')
        
    if len(pre_nucleus_off_dissociation) > 0:
        sum_interval = sum_interval + 2.0*len(pre_nucleus_off_dissociation)*kinetic_rates['aggregate_off_dissociation']
        intervals.append(sum_interval)
        action.append('pre_nucleus_off_dissociation')
            
    if len(fibril_association) > 0:
        sum_interval = sum_interval + len(fibril_association)*kinetic_rates['fibril_association']
        intervals.append(sum_interval)
        action.append('fibril_association')
        
    if len(fibril_dissociation) > 0:
        sum_interval = sum_interval + len(fibril_dissociation)*kinetic_rates['fibril_dissociation']
        intervals.append(sum_interval)
        action.append('fibril_dissociation')
        
    if len(second_monomer_attach) > 0:
        sum_interval = sum_interval + len(second_monomer_attach)*kinetic_rates['second_monomer_attach']
        intervals.append(sum_interval)
        action.append('second_monomer_attach')
        
    if len(second_monomer_detach) > 0:
        sum_interval = sum_interval + len(second_monomer_detach)*kinetic_rates['second_monomer_detach']
        intervals.append(sum_interval)
        action.append('second_monomer_detach')
        
    if len(second_dimer_association) > 0:
        sum_interval = sum_interval + len(second_dimer_association)*kinetic_rates['second_pre_nucleus_association']
        intervals.append(sum_interval)
        action.append('second_dimer_association')
        
    if len(second_pre_nucleus_association) > 0:
        sum_interval = sum_interval + len(second_pre_nucleus_association)*kinetic_rates['second_pre_nucleus_association']
        intervals.append(sum_interval)
        action.append('second_pre_nucleus_association')
#        
    if len(second_pre_nucleus_dissociation) > 0:
        sum_interval = sum_interval + len(second_pre_nucleus_dissociation)*kinetic_rates['pre_nucleus_on_dissociation']
        intervals.append(sum_interval)
        action.append('second_pre_nucleus_dissociation')
#    
#    if len(second_fibril_association) > 0:
#        sum_interval = sum_interval + len(second_fibril_association)*kinetic_rates['fibril_association']
#        intervals.append(sum_interval)
#        action.append('second_fibril_association')
#        
    if len(second_fibril_dissociation) > 0:
        sum_interval = sum_interval + len(second_fibril_dissociation)*kinetic_rates['fibril_dissociation']
        intervals.append(sum_interval)
        action.append('second_fibril_dissociation')
#    
    if len(second_fibril_detach) > 0:
        sum_interval = sum_interval + len(second_fibril_detach)*kinetic_rates['second_fibril_detach']
        intervals.append(sum_interval)
        action.append('second_fibril_detach')
##    
    if len(monomers) > 0:
        sum_interval = sum_interval + len(monomers)*kinetic_rates['monomer_diffusion']
        intervals.append(sum_interval)
        action.append('monomer_diffusion')
        
#    if len(monos_inter_diffusion_on) > 0:
#        sum_interval = sum_interval + len(monos_inter_diffusion_on)*kinetic_rates['mono_inter_diffusion_on']
#        intervals.append(sum_interval)
#        action.append('monomer_diffusion')#action.append('mono_inter_diffusion_on')     
#        
#    if len(monos_inter_diffusion_off) > 0:
#        sum_interval = sum_interval + len(monos_inter_diffusion_off)*kinetic_rates['mono_inter_diffusion_off']
#        intervals.append(sum_interval)
#        action.append('monomer_diffusion')#action.append('mono_inter_diffusion_off') 
    
    if len(dimer_on_diffusion) > 0:
        sum_interval = sum_interval + len(dimer_on_diffusion)*kinetic_rates['pre_on_diffusion']
        intervals.append(sum_interval)
        action.append('dimer_on_diffusion')
        
    if len(trimer_on_diffusion) > 0:
        sum_interval = sum_interval + len(trimer_on_diffusion)*kinetic_rates['pre_on_diffusion']
        intervals.append(sum_interval)
        action.append('trimer_on_diffusion')
        
    if len(tetramer_on_diffusion) > 0:
        sum_interval = sum_interval + len(tetramer_on_diffusion)*kinetic_rates['pre_on_diffusion']
        intervals.append(sum_interval)
        action.append('tetramer_on_diffusion')
        
    if len(pentamer_on_diffusion) > 0:
        sum_interval = sum_interval + len(pentamer_on_diffusion)*kinetic_rates['pre_on_diffusion']
        intervals.append(sum_interval)
        action.append('pentamer_on_diffusion')
        
    if len(dimer_off_diffusion) > 0:
#        print("dimer off diffusion", dimer_off_diffusion)
        sum_interval = sum_interval + len(dimer_off_diffusion)*kinetic_rates['pre_off_diffusion']
        intervals.append(sum_interval)
        action.append('dimer_off_diffusion')
#        print("length of dimer off diffusion", len(dimer_off_diffusion))
        
    if len(trimer_off_diffusion) > 0:
        sum_interval = sum_interval + len(trimer_off_diffusion)*kinetic_rates['pre_off_diffusion']
        intervals.append(sum_interval)
        action.append('trimer_off_diffusion')
        
    if len(tetramer_off_diffusion) > 0:
        sum_interval = sum_interval + len(tetramer_off_diffusion)*kinetic_rates['pre_off_diffusion']
        intervals.append(sum_interval)
        action.append('tetramer_off_diffusion')
        
    if len(pentamer_off_diffusion) > 0:
        sum_interval = sum_interval + len(pentamer_off_diffusion)*kinetic_rates['pre_off_diffusion']
        intervals.append(sum_interval)
        action.append('pentamer_off_diffusion')
        

    intervals = [(i/sum_interval + 0.1) for i in intervals]
#    print("sumi ntervals", intervals)
    t = random.uniform(0.0,1.0)
    time_step = (1.0/sum_interval)*math.log(1.0/t)
#    print("time step os",time_step)
    time = time + time_step/3600
#    time_dynamics.append(time)
    b = random.uniform(0.1,1.1)
    diffusion_length = math.sqrt(diff_0*time_step)
#    print("diff length", diffusion_length)
#    print(intervals,b)
#    print(action)
    STOP_TEST = []
    for jlk90 in range(1,len(intervals)):
#        print("start cycle",jlk90-1, intervals[jlk90-1],b, intervals[jlk90])
        if intervals[jlk90-1]<= b <= intervals[jlk90]:
            j = jlk90
#            print("j is",j)
#            print("pre nucle on association main",pre_nucleus_on_association)
#            print('')
#            print('monos_inter_diffusion_on main',monos_inter_diffusion_on)
#            print('dimer on from main')
#            print('dimer_on_association', dimer_on_association)
#            print('')
#            print('monomers', monomers)
#            print(dimer_on_diffusion)
#           print('dimer on association before', dimer_on_association)
#    print(" j is", j)
    if action[j-1] == 'dimer_on_association':
#       print('dimer on association')
       dimerOn = random.choice(dimer_on_association)
       dimer_on(L,dimerOn[0],sigma,dimerOn[1],monomers,monos_inter_diffusion_on, monos_inter_diffusion_off,
            dimer_on_association, dimer_off_association, pre_nucleus_on_association,
            fibril_association, second_monomer_attach, second_fibril_association, 
            second_pre_nucleus_association, second_dimer_association, pre_nucleus_off_association,
            dimer_on_dissociation, dimer_on_diffusion,STOP_TEST) 

#               print('')
    elif action[j-1] == 'dimer_off_association':                
#       print("dime off association")
       dimerOff = random.choice(dimer_off_association)
       dimer_off(L,dimerOff[0],sigma,dimerOff[1],monomers,monos_inter_diffusion_on, monos_inter_diffusion_off,
            dimer_on_association, dimer_off_association, pre_nucleus_on_association,
            fibril_association, second_monomer_attach, second_fibril_association, 
            second_pre_nucleus_association, second_dimer_association, pre_nucleus_off_association,
            dimer_on_dissociation, dimer_on_diffusion,dimer_off_dissociation,dimer_off_diffusion,STOP_TEST) 
#       print("made a off dimer!!!", dimerOff) 
       
    elif action[j-1] == 'dimer_on_dissociation':      
#       print('dissociated a dimer on!!!!!!!')
       dimerOnDiss = random.choice(dimer_on_dissociation)
       monomer_fibril_detatch(L,sigma,dimerOnDiss, diffusion_length, monos_inter_diffusion_on, monos_inter_diffusion_off,
            dimer_on_association, dimer_off_association, pre_nucleus_on_association,
            fibril_association, second_monomer_attach, second_fibril_association, 
            second_pre_nucleus_association, second_dimer_association, pre_nucleus_off_association,
            monomers,pre_nucleus_on_dissociation,fibril_dissociation, dimer_on_diffusion,
            second_fibril_dissociation, second_pre_nucleus_dissociation, dimer_off_diffusion,
            pre_nucleus_off_dissociation, trimer_on_diffusion, trimer_off_diffusion,
            tetramer_on_diffusion, tetramer_off_diffusion, pentamer_on_diffusion,
            pentamer_off_diffusion,dimer_off_dissociation,dimer_on_dissociation,second_monomer_detach,second_fibril_detach,
            nucleus_off_numb,STOP_TEST)   # Don't forget to change it to relevant length!!!                       
#       print('second prenucleus dissociation', second_pre_nucleus_dissociation)
    elif action[j-1] == 'dimer_off_dissociation': 
#        print("dimer off dissociation")
        aggrOffDiss = random.choice(dimer_off_dissociation)
        dimerOffDiss =   aggrOffDiss              
        monomer_fibril_detatch(L,sigma,dimerOffDiss, diffusion_length,monos_inter_diffusion_on, monos_inter_diffusion_off,
            dimer_on_association, dimer_off_association, pre_nucleus_on_association,
            fibril_association, second_monomer_attach, second_fibril_association, 
            second_pre_nucleus_association, second_dimer_association, pre_nucleus_off_association,
            monomers,pre_nucleus_on_dissociation,fibril_dissociation, dimer_on_diffusion,
            second_fibril_dissociation, second_pre_nucleus_dissociation, dimer_off_diffusion,
            pre_nucleus_off_dissociation, trimer_on_diffusion, trimer_off_diffusion,
            tetramer_on_diffusion, tetramer_off_diffusion, pentamer_on_diffusion,
            pentamer_off_diffusion,dimer_off_dissociation,dimer_on_dissociation,second_monomer_detach,second_fibril_detach,
            nucleus_off_numb,STOP_TEST) 
#        print("dimer off dissociation")                       
    elif action[j-1] == 'pre_nucleus_on_association': 
#        print("pre nucle on association")
        preNucOn = random.choice(pre_nucleus_on_association)
#        print('pre nucleus on', preNucOn, L[preNucOn[0]], L[preNucOn[1]])
        if L[preNucOn[1]][2] == 102:
            nucleus_on_low_end(L,preNucOn[0],sigma,preNucOn[1],monomers,monos_inter_diffusion_on,
            monos_inter_diffusion_off,pre_nucleus_off_association, second_dimer_association,
            dimer_on_association, dimer_off_association, pre_nucleus_on_association,
            fibril_association, second_monomer_attach, second_fibril_association, 
            second_pre_nucleus_association, fibril_dissociation,trimer_on_diffusion,
            dimer_on_dissociation, dimer_on_diffusion,pre_nucleus_on_dissociation,
            tetramer_on_diffusion,pentamer_on_diffusion,STOP_TEST)               
        elif L[preNucOn[1]][2] == 101:
            nucleus_on_high_end(L,preNucOn[0],sigma,preNucOn[1],monomers,monos_inter_diffusion_on,
            monos_inter_diffusion_off,pre_nucleus_off_association, second_dimer_association,
            dimer_on_association, dimer_off_association, pre_nucleus_on_association,
            fibril_association, second_monomer_attach, second_fibril_association, 
            second_pre_nucleus_association, fibril_dissociation,trimer_on_diffusion,
            dimer_on_dissociation, dimer_on_diffusion,pre_nucleus_on_dissociation,
            tetramer_on_diffusion,pentamer_on_diffusion,STOP_TEST)          
                                                 
    elif action[j-1] == 'pre_nucleus_off_association':                
#        print("pre off association")
        preNucOff = random.choice(pre_nucleus_off_association)                
#        print("pre nuc off assoc",preNucOff)
        nucleus_off(L,preNucOff[0],sigma,preNucOff[1],nuc_size, monomers,monos_inter_diffusion_on, monos_inter_diffusion_off,
            dimer_on_association, dimer_off_association, pre_nucleus_on_association,
            fibril_association, second_monomer_attach, second_fibril_association, 
            second_pre_nucleus_association, second_dimer_association, pre_nucleus_off_association,
            dimer_on_dissociation, dimer_off_diffusion,pre_nucleus_off_dissociation,
            tetramer_off_diffusion,pentamer_off_diffusion,fibril_dissociation,
            pre_nucleus_on_dissociation, trimer_off_diffusion,dimer_off_dissociation,nucleus_off_numb,STOP_TEST)  
#        print("pre nucleus off association stop ", STOP_TEST)
        
    elif action[j-1] == 'pre_nucleus_on_dissociation': 
        preNuclOnDiss = random.choice(pre_nucleus_on_dissociation) 
#        print("from main pre nucl dissociation list")
#        print()
        monomer_fibril_detatch(L,sigma,preNuclOnDiss, diffusion_length, monos_inter_diffusion_on, monos_inter_diffusion_off,
            dimer_on_association, dimer_off_association, pre_nucleus_on_association,
            fibril_association, second_monomer_attach, second_fibril_association, 
            second_pre_nucleus_association, second_dimer_association, pre_nucleus_off_association,
            monomers,pre_nucleus_on_dissociation,fibril_dissociation, dimer_on_diffusion,
            second_fibril_dissociation, second_pre_nucleus_dissociation, dimer_off_diffusion,
            pre_nucleus_off_dissociation, trimer_on_diffusion, trimer_off_diffusion,
            tetramer_on_diffusion, tetramer_off_diffusion, pentamer_on_diffusion,
            pentamer_off_diffusion,dimer_off_dissociation,dimer_on_dissociation,second_monomer_detach,second_fibril_detach,
            nucleus_off_numb,STOP_TEST)
#        print('pre nuclues on dissociation')
        
    elif action[j-1] == 'pre_nucleus_off_dissociation':                
        nuclOffDiss = random.choice(pre_nucleus_off_dissociation)
#        print("nuclOffDiss")
        nuclOFF = []
        for xix in range(len(nuclOffDiss)):
            nuclOFF.append(L[nuclOffDiss[xix]][:])
            nuclOFF[xix].append(nuclOffDiss[xix])
#                print(nuclOFF)
        nuclOFF.sort(key = lambda x: x[2])
#        print("sorted nuclOFF", nuclOFF)
        preNuclOffDiss =   nuclOFF[len(nuclOffDiss)-1][4]
#        print("selected part to dissociate pre off", preNuclOffDiss,L[preNuclOffDiss])
        monomer_fibril_detatch(L,sigma,preNuclOffDiss , diffusion_length,monos_inter_diffusion_on, monos_inter_diffusion_off,
            dimer_on_association, dimer_off_association, pre_nucleus_on_association,
            fibril_association, second_monomer_attach, second_fibril_association, 
            second_pre_nucleus_association, second_dimer_association, pre_nucleus_off_association,
            monomers,pre_nucleus_on_dissociation,fibril_dissociation, dimer_on_diffusion,
            second_fibril_dissociation, second_pre_nucleus_dissociation, dimer_off_diffusion,
            pre_nucleus_off_dissociation, trimer_on_diffusion, trimer_off_diffusion,
            tetramer_on_diffusion, tetramer_off_diffusion, pentamer_on_diffusion,
            pentamer_off_diffusion,dimer_off_dissociation,dimer_on_dissociation,second_monomer_detach,second_fibril_detach,
            nucleus_off_numb,STOP_TEST) 
#                print("pre nucleus off dissociation")
    elif action[j-1] == 'fibril_association':                
#        print("fibril association")
        fibrilOnElong = random.choice(fibril_association)
        if L[fibrilOnElong[1]][2] == 122:
            elongation_on_low_end(L,fibrilOnElong[0],sigma,fibrilOnElong[1], monomers,monos_inter_diffusion_on, 
                                  monos_inter_diffusion_off,
            dimer_on_association, dimer_off_association, pre_nucleus_on_association,
            fibril_association, second_monomer_attach, second_fibril_association, second_fibril_dissociation,
            second_pre_nucleus_association, second_dimer_association, pre_nucleus_off_association,
            pre_nucleus_on_dissociation,second_monomer_detach, second_pre_nucleus_dissociation,
            fibril_dissociation,pentamer_on_diffusion,STOP_TEST)
        elif L[fibrilOnElong[1]][2] == 121:
            elongation_on_high_end(L,fibrilOnElong[0],sigma,fibrilOnElong[1],  monomers,monos_inter_diffusion_on, 
                                  monos_inter_diffusion_off,
            dimer_on_association, dimer_off_association, pre_nucleus_on_association,
            fibril_association, second_monomer_attach, second_fibril_association, second_fibril_dissociation,
            second_pre_nucleus_association, second_dimer_association, pre_nucleus_off_association,
            pre_nucleus_on_dissociation,second_monomer_detach, second_pre_nucleus_dissociation,
            fibril_dissociation,pentamer_on_diffusion,STOP_TEST) 
#            print("fibril association")          
        
    elif action[j-1] == 'fibril_dissociation':                
#        print("fibril dissociation")
        fibrilOnDiss = random.choice(fibril_dissociation) 
        monomer_fibril_detatch(L,sigma,fibrilOnDiss, diffusion_length,monos_inter_diffusion_on, monos_inter_diffusion_off,
            dimer_on_association, dimer_off_association, pre_nucleus_on_association,
            fibril_association, second_monomer_attach, second_fibril_association, 
            second_pre_nucleus_association, second_dimer_association, pre_nucleus_off_association,
            monomers,pre_nucleus_on_dissociation,fibril_dissociation, dimer_on_diffusion,
            second_fibril_dissociation, second_pre_nucleus_dissociation, dimer_off_diffusion,
            pre_nucleus_off_dissociation, trimer_on_diffusion, trimer_off_diffusion,
            tetramer_on_diffusion, tetramer_off_diffusion, pentamer_on_diffusion,
            pentamer_off_diffusion,dimer_off_dissociation,dimer_on_dissociation,second_monomer_detach,second_fibril_detach,
            nucleus_off_numb,STOP_TEST)  
#        print("fibril dissociation")             
                     
    elif action[j-1] == 'second_monomer_attach':  
#        print("second mono attach")
        secondMonoAttach = random.choice(second_monomer_attach)              
        mono_second(L,secondMonoAttach[0],sigma,secondMonoAttach[1], monomers, monos_inter_diffusion_on, 
                    monos_inter_diffusion_off,
            dimer_on_association, dimer_off_association, pre_nucleus_on_association,
            fibril_association, second_monomer_attach, second_fibril_association, 
            second_pre_nucleus_association, second_dimer_association, pre_nucleus_off_association,
            pre_nucleus_on_dissociation,second_monomer_detach,pentamer_on_diffusion,STOP_TEST)                           
    elif action[j-1] == 'second_monomer_detach':
#        print("second monodetach")
        secondMonoDetach = random.choice(second_monomer_detach)                
        monomer_fibril_detatch(L,sigma,secondMonoDetach, diffusion_length,monos_inter_diffusion_on, monos_inter_diffusion_off,
            dimer_on_association, dimer_off_association, pre_nucleus_on_association,
            fibril_association, second_monomer_attach, second_fibril_association, 
            second_pre_nucleus_association, second_dimer_association, pre_nucleus_off_association,
            monomers,pre_nucleus_on_dissociation,fibril_dissociation, dimer_on_diffusion,
            second_fibril_dissociation, second_pre_nucleus_dissociation, dimer_off_diffusion,
            pre_nucleus_off_dissociation, trimer_on_diffusion, trimer_off_diffusion,
            tetramer_on_diffusion, tetramer_off_diffusion, pentamer_on_diffusion,
            pentamer_off_diffusion,dimer_off_dissociation,dimer_on_dissociation,second_monomer_detach,second_fibril_detach,
            nucleus_off_numb,STOP_TEST)                           
    elif action[j-1] == 'second_dimer_association':                
#        print("second dimer association")
        secondDimer = random.choice(second_dimer_association)  
        dimer_second(L,secondDimer[0],sigma,secondDimer[1],monomers,monos_inter_diffusion_on,
                     monos_inter_diffusion_off,
            dimer_on_association, dimer_off_association, pre_nucleus_on_association,
            fibril_association, second_monomer_attach, second_fibril_association, 
            second_pre_nucleus_association, second_dimer_association, pre_nucleus_off_association,
            pre_nucleus_on_dissociation,second_monomer_detach, second_pre_nucleus_dissociation,STOP_TEST)                    
    elif action[j-1] == 'second_pre_nucleus_association':                
#        print("second pre nucleus association")
        secondNucl = random.choice(second_pre_nucleus_association)  
        nucleus_second(L,secondNucl[0],sigma,secondNucl[1], monomers,monos_inter_diffusion_on, 
                       monos_inter_diffusion_off,
            dimer_on_association, dimer_off_association, pre_nucleus_on_association,
            fibril_association, second_monomer_attach, second_fibril_association, second_fibril_dissociation,
            second_pre_nucleus_association, second_dimer_association, pre_nucleus_off_association,
            pre_nucleus_on_dissociation,second_monomer_detach, second_pre_nucleus_dissociation,second_fibril_detach,STOP_TEST)             
    elif action[j-1] == 'second_pre_nucleus_dissociation':                
#        print("second pre nucl dissociation")
        secondNuclDiss = random.choice(second_pre_nucleus_dissociation)                
        monomer_fibril_detatch(L,sigma,secondNuclDiss, diffusion_length,monos_inter_diffusion_on, monos_inter_diffusion_off,
            dimer_on_association, dimer_off_association, pre_nucleus_on_association,
            fibril_association, second_monomer_attach, second_fibril_association, 
            second_pre_nucleus_association, second_dimer_association, pre_nucleus_off_association,
            monomers,pre_nucleus_on_dissociation,fibril_dissociation, dimer_on_diffusion,
            second_fibril_dissociation, second_pre_nucleus_dissociation, dimer_off_diffusion,
            pre_nucleus_off_dissociation, trimer_on_diffusion, trimer_off_diffusion,
            tetramer_on_diffusion, tetramer_off_diffusion, pentamer_on_diffusion,
            pentamer_off_diffusion,dimer_off_dissociation,dimer_on_dissociation,second_monomer_detach,second_fibril_detach,
            nucleus_off_numb,STOP_TEST)            
    elif action[j-1] == 'second_fibril_association':                
#        print("second fibril association")
        secondFibrilAss = random.choice(second_fibril_association)           
        elongation_second(L,secondFibrilAss[0],sigma,secondFibrilAss[1],monomers,monos_inter_diffusion_on, 
                          monos_inter_diffusion_off,dimer_on_association, dimer_off_association, pre_nucleus_on_association,
                          fibril_association, second_monomer_attach, second_fibril_association, second_fibril_dissociation,
                          second_pre_nucleus_association, second_dimer_association, pre_nucleus_off_association,
                          pre_nucleus_on_dissociation,second_monomer_detach, second_pre_nucleus_dissociation,
                          fibril_dissociation,STOP_TEST)
    elif action[j-1] == 'second_fibril_dissociation':                
#        print("second fibril dissociation")
        secondFibrilDiss = random.choice(second_fibril_dissociation)                
        monomer_fibril_detatch(L,sigma,secondFibrilDiss, diffusion_length,monos_inter_diffusion_on, monos_inter_diffusion_off,
            dimer_on_association, dimer_off_association, pre_nucleus_on_association,
            fibril_association, second_monomer_attach, second_fibril_association, 
            second_pre_nucleus_association, second_dimer_association, pre_nucleus_off_association,
            monomers,pre_nucleus_on_dissociation,fibril_dissociation, dimer_on_diffusion,
            second_fibril_dissociation, second_pre_nucleus_dissociation, dimer_off_diffusion,
            pre_nucleus_off_dissociation, trimer_on_diffusion, trimer_off_diffusion,
            tetramer_on_diffusion, tetramer_off_diffusion, pentamer_on_diffusion,
            pentamer_off_diffusion,dimer_off_dissociation,dimer_on_dissociation,second_monomer_detach,second_fibril_detach,
            nucleus_off_numb,STOP_TEST)                            
    elif action[j-1] == 'second_fibril_detach':                
#        print("second fibril setach")
        secondFibrilDetach = random.choice(second_fibril_detach)
        diffCoeffSec = math.sqrt(diff_coeff_aggregat(second_fibril_detach)*time_step)
        detatch_second(L,sigma,secondFibrilDetach,diffCoeffSec,second_fibril_association,second_fibril_dissociation,
           pentamer_on_diffusion,fibril_dissociation,fibril_association,second_monomer_attach,second_fibril_detach,STOP_TEST)
    elif action[j-1] == 'monomer_diffusion':                
        monoDiffuse = random.choice(monomers)  
#        print("diffusing a mono")
        markov_mono_box(L,monoDiffuse,sigma,diffusion_length,monomers,monos_inter_diffusion_on, monos_inter_diffusion_off,
            dimer_on_association, dimer_off_association, pre_nucleus_on_association,
            fibril_association, second_monomer_attach, second_fibril_association, 
            second_pre_nucleus_association, second_dimer_association, pre_nucleus_off_association,STOP_TEST)        
#    elif action[j-1] == 'mono_inter_diffusion_on':                
#        monoInterDiffuse = random.choice(monos_inter_diffusion_on)  
##        print("diffusing a mono within range")
#        markov_mono_box(L,monoDiffuse,sigma,diffusion_length,monomers,monos_inter_diffusion_on,
#                        monos_inter_diffusion_off,
#            dimer_on_association, dimer_off_association, pre_nucleus_on_association,
#            fibril_association, second_monomer_attach, second_fibril_association, 
#            second_pre_nucleus_association, second_dimer_association, pre_nucleus_off_association,STOP_TEST)
#    elif action[j-1] == 'mono_inter_diffusion_off':                
#        monoInterDiffuse = random.choice(monos_inter_diffusion_off)  
##                print("diffusing a mono within range", monoDiffuse)
#        markov_mono_box(L,monoDiffuse,sigma,diffusion_length,monomers,monos_inter_diffusion_on,
#                        monos_inter_diffusion_off,
#            dimer_on_association, dimer_off_association, pre_nucleus_on_association,
#            fibril_association, second_monomer_attach, second_fibril_association, 
#            second_pre_nucleus_association, second_dimer_association, pre_nucleus_off_association,STOP_TEST)    
    elif action[j-1] == 'dimer_on_diffusion':     
#        print('dimer on diffudion')
        dimerOnDiff = random.choice(dimer_on_diffusion)
        diffCoeffDimer = math.sqrt(diff_coeff_aggregat(dimerOnDiff)*time_step)
        aggregate_diffusion(L,sigma,dimerOnDiff,diffCoeffDimer,pre_nucleus_on_association,
                pre_nucleus_off_association,fibril_association,second_monomer_attach,monos_inter_diffusion_on,
                monos_inter_diffusion_off,dimer_off_association,STOP_TEST)
    elif action[j-1] == 'dimer_off_diffusion':      
#        print('dimer off diffusion')
        dimerOffDiff = random.choice(dimer_off_diffusion)
        diff_aggr_length = math.sqrt(diff_m*(1.0/(1.75))*time_step)
        aggregate_diffusion(L,sigma,dimerOffDiff,diff_aggr_length,pre_nucleus_on_association,
                pre_nucleus_off_association,fibril_association,second_monomer_attach,monos_inter_diffusion_on,
                monos_inter_diffusion_off,dimer_off_association,STOP_TEST)  
    elif action[j-1] == 'trimer_on_diffusion':                
#        print("trimer on diffusion")
        trimerOnDiff = random.choice(trimer_on_diffusion)
        diffCoeffTrimer = math.sqrt(diff_coeff_aggregat(trimerOnDiff)*time_step)
        aggregate_diffusion(L,sigma,trimerOnDiff,diffCoeffTrimer,pre_nucleus_on_association,
                pre_nucleus_off_association,fibril_association,second_monomer_attach,monos_inter_diffusion_on,
                monos_inter_diffusion_off,dimer_off_association,STOP_TEST)
    elif action[j-1] == 'trimer_off_diffusion':                
#        print("trimer off diffusion")
        trimerOffDiff = random.choice(trimer_off_diffusion)
        diffCoeffOff = math.sqrt(diff_m*(1.0/(2.73205))*time_step)
        aggregate_diffusion(L,sigma,trimerOffDiff,diffCoeffOff,pre_nucleus_on_association,
                pre_nucleus_off_association,fibril_association,second_monomer_attach,monos_inter_diffusion_on,
                monos_inter_diffusion_off,dimer_off_association,STOP_TEST)
    elif action[j-1] == 'tetramer_on_diffusion':                
#        print("tetramer on diffusion")
        tetramerOnDiff = random.choice(tetramer_on_diffusion)
        diffCoeffTetramer = math.sqrt(diff_coeff_aggregat(tetramerOnDiff)*time_step)
        aggregate_diffusion(L,sigma,tetramerOnDiff,diffCoeffTetramer,pre_nucleus_on_association,
                pre_nucleus_off_association,fibril_association,second_monomer_attach,monos_inter_diffusion_on,monos_inter_diffusion_off,
                dimer_off_association,STOP_TEST)
    elif action[j-1] == 'tetramer_off_diffusion':                
#        print("tetramer off diffusion")
        tetramerOffDiff = random.choice(tetramer_off_diffusion)
        diffCoeffOffTetra = math.sqrt(diff_m*(1.0/(2.82842))*time_step)
        aggregate_diffusion(L,sigma,tetramerOffDiff,diffCoeffOffTetra,pre_nucleus_on_association,
                pre_nucleus_off_association,fibril_association,second_monomer_attach,monos_inter_diffusion_on,monos_inter_diffusion_off,
                dimer_off_association,STOP_TEST)
    elif action[j-1] == 'pentamer_on_diffusion':                
#        print("pentamer on diffusion")
        pentamerOnDiff = random.choice(pentamer_on_diffusion)
        diffCoeffPentamer = math.sqrt(diff_coeff_aggregat(pentamerOnDiff)*time_step)
        aggregate_diffusion(L,sigma,pentamerOnDiff,diffCoeffPentamer,pre_nucleus_on_association,
                pre_nucleus_off_association,fibril_association,second_monomer_attach,monos_inter_diffusion_on,monos_inter_diffusion_off,
                dimer_off_association,STOP_TEST)
    elif action[j-1] == 'pentamer_off_diffusion':                
#        print("pentamer off diffusion")
        pentamerOffDiff = random.choice(pentamer_off_diffusion)
        diffCoeffOffPenta = math.sqrt(diff_m*(1.0/(3))*time_step)
        aggregate_diffusion(L,sigma,pentamerOffDiff,diffCoeffOffPenta,pre_nucleus_on_association,
                pre_nucleus_off_association,fibril_association,second_monomer_attach,monos_inter_diffusion_on,monos_inter_diffusion_off,
                dimer_off_association,STOP_TEST)

    if STOP_TEST[0] == 0:
        time = time - time_step/3600
#    print("end cycle")
    if th%10000 == 0:
#            print(th)
#            print(time)
#                    print(monomers)
            colors = []
            time_dynamics.append(time)
            monomers_dynamics.append(len(monomers))
            pre_second_sum = 0
            for ff1 in L:
                if ff1[2] == 301 or ff1[2] == 302 or ff1[2] == 303 or ff1[2] == 3020 or ff1[2] == 3010 or ff1[2] == 300:
                    pre_second_sum = pre_second_sum + 1
            pre_on_nuclei_dynamics.append((2.0*len(dimer_on_diffusion) + 3.0*len(trimer_on_diffusion) + 4.0*len(tetramer_on_diffusion))\
                                            + pre_second_sum)
    #            print(pre_on_nuclei_dynamics[len(pre_on_nuclei_dynamics)-1])
            pre_off_nuclei_dynamics.append(2.0*len(dimer_off_diffusion) + 3.0*len(trimer_off_diffusion) + 4.0*len(tetramer_off_diffusion) + 5.0*len(pentamer_off_diffusion))
            
            fibril_sum = 0
            for ff0 in L:
                if ff0[2] == 1 or ff0[2] == 121 or ff0[2] == 122 or ff0[2]== 30 or ff0[2] == 31 or ff0[2] == 32 or ff0[2] == 33:
                    fibril_sum = fibril_sum + 1        
            fibrils_dynamics.append(fibril_sum)
            off_nuclei_dynamics.append(6*len(nucleus_off_numb))

#            for i in range(N):
#            
#                if L[i][2] == 101:
#            
#                    colors.append('r')
##                            print('101',L[i])
#                    
#                elif L[i][2] == 102:
#                    
#                    colors.append('royalblue')
##                            print('102',L[i])
#            
#                elif L[i][2] == 12:
#            
#                    colors.append('orange')
#                    
#                elif L[i][2] == 11:
#            
#                    colors.append('rosybrown')
#                    
#                elif L[i][2] == 1:
#            
#                    colors.append('y')
#            
#                elif L[i][2] == 21:
#            
#                    colors.append('orange')
#            
#                elif L[i][2] == 0:
#            
#                    colors.append('k')
#            
#                elif L[i][2] == 20:
#            
#                    colors.append('orange')    
#            
#                elif L[i][2] == 301:
#            
#                    colors.append('gold')
#                    
#                elif L[i][2] == 302:
#            
#                    colors.append('yellow')
#                    
#                elif L[i][2] == 303:
#            
#                    colors.append('hotpink')
#            
#                elif L[i][2] == 30 or L[i][2] == 31 or L[i][2] == 32 or L[i][2] == 33:
#            
#                    colors.append('c')
#                    
#                else:
#            
#                    colors.append('g')
#            
##                    snapshot(L, colors)
#            snapshot2(L, colors)
#            pylab.plot(time_dynamics, monomers_dynamics,'b', label = 'monomers')
#            pylab.plot(time_dynamics, pre_on_nuclei_dynamics,'y', label = 'pre_nucl._on_pathway')
#            pylab.plot(time_dynamics, fibrils_dynamics,'r', label = 'fibrils')
#            pylab.plot(time_dynamics, pre_off_nuclei_dynamics,'c', label = 'pre_nucl._off_pathway')
#            pylab.plot(time_dynamics, off_nuclei_dynamics,'g', label = 'off_pathway_nuclei')
#            pylab.xlabel('$Time\ (h)$', fontsize = 14)
#            pylab.ylabel(r'$N$', fontsize = 14)
##                    pylab.yticks(np.linspace(0, 900, 50, endpoint=True))
#            #pylab.yscale('log')
#            pylab.title(str(N)+ ' ' +'particles' +','+ ' '+ str(iter_nums)+' '+'runs',  fontsize = 14)
#            pylab.legend(loc= 'upper right')
#            pylab.savefig('off_trial.png')
#            #plt.savefig('600.svg')
#            pylab.show()

            if os.path.isfile('monomers.txt'):
                with open('monomers.txt', 'w') as fff:
                    fff.write(json.dumps(monomers_dynamics))
            else:
                with open('monomers.txt', 'w') as fff:
                    fff.write(json.dumps(monomers_dynamics))
                
                    
            if os.path.isfile('pre_on_nuclei.txt'):
                with open('pre_on_nuclei.txt', 'w') as f_f:
                    f_f.write(json.dumps(pre_on_nuclei_dynamics))
            else:
                with open('pre_on_nuclei.txt', 'w') as f_f:
                    f_f.write(json.dumps(pre_on_nuclei_dynamics))
            
            if os.path.isfile('pre_off_nuclei.txt'):
                with open('pre_off_nuclei.txt', 'w') as f_ff:
                    f_ff.write(json.dumps(pre_off_nuclei_dynamics))
            else:
                with open('pre_off_nuclei.txt', 'w') as f_ff:
                    f_ff.write(json.dumps(pre_off_nuclei_dynamics))
            
            if os.path.isfile('fibrils.txt'):
                with open('fibrils.txt', 'w') as ff_fff:
                    ff_fff.write(json.dumps(fibrils_dynamics))
            else:
                with open('fibrils.txt', 'w') as ff_fff:
                    ff_fff.write(json.dumps(fibrils_dynamics))
                
            if os.path.isfile('off_nuclei.txt'):
                with open('off_nuclei.txt', 'w') as ff_ff:
                    ff_ff.write(json.dumps(off_nuclei_dynamics))
            else:
                with open('off_nuclei.txt', 'w') as ff_ff:
                    ff_ff.write(json.dumps(off_nuclei_dynamics))
                
            if os.path.isfile('time.txt'):
                with open('time.txt', 'w') as ff:
                    ff.write(json.dumps(time_dynamics))
            else:
                with open('time.txt', 'w') as ff:
                    ff.write(json.dumps(time_dynamics))
            
            with open('test_L.txt', 'w') as f:
                f.write(json.dumps(L))



if os.path.isfile('monomers.txt'):
    with open('monomers.txt', 'w') as fff:
        fff.write(json.dumps(monomers_dynamics))
else:
    with open('monomers.txt', 'w') as fff:
        fff.write(json.dumps(monomers_dynamics))
    
        
if os.path.isfile('pre_on_nuclei.txt'):
    with open('pre_on_nuclei.txt', 'w') as f_f:
        f_f.write(json.dumps(pre_on_nuclei_dynamics))
else:
    with open('pre_on_nuclei.txt', 'w') as f_f:
        f_f.write(json.dumps(pre_on_nuclei_dynamics))

if os.path.isfile('pre_off_nuclei.txt'):
    with open('pre_off_nuclei.txt', 'w') as f_ff:
        f_ff.write(json.dumps(pre_off_nuclei_dynamics))
else:
    with open('pre_off_nuclei.txt', 'w') as f_ff:
        f_ff.write(json.dumps(pre_off_nuclei_dynamics))

if os.path.isfile('fibrils.txt'):
    with open('fibrils.txt', 'w') as ff_fff:
        ff_fff.write(json.dumps(fibrils_dynamics))
else:
    with open('fibrils.txt', 'w') as ff_fff:
        ff_fff.write(json.dumps(fibrils_dynamics))
    
if os.path.isfile('off_nuclei.txt'):
    with open('off_nuclei.txt', 'w') as ff_ff:
        ff_ff.write(json.dumps(off_nuclei_dynamics))
else:
    with open('off_nuclei.txt', 'w') as ff_ff:
        ff_ff.write(json.dumps(off_nuclei_dynamics))
    
if os.path.isfile('time.txt'):
    with open('time.txt', 'w') as ff:
        ff.write(json.dumps(time_dynamics))
else:
    with open('time.txt', 'w') as ff:
        ff.write(json.dumps(time_dynamics))

with open('test_L.txt', 'w') as f:
    f.write(json.dumps(L))
    

#pylab.plot(time_dynamics, pre_on_nuclei_dynamics,'y', label = 'pre_nucl._on_pathway')
#pylab.plot(time_dynamics, fibrils_dynamics,'r', label = 'fibrils')
#pylab.plot(time_dynamics, pre_off_nuclei_dynamics,'c', label = 'pre_nucl._off_pathway')
#pylab.plot(time_dynamics, off_nuclei_dynamics,'g', label = 'off_pathway_nuclei')
#pylab.xlabel('$Time\ (h)$', fontsize = 14)
#pylab.ylabel(r'$N$', fontsize = 14)
##pylab.yscale('log')
#pylab.title(str(N)+ ' ' +'particles' +','+ ' '+ str(iter_nums)+' '+'runs',  fontsize = 14)
#pylab.legend(loc= 'upper right')
#pylab.savefig('off_trial2.png')
#pylab.savefig('900.svg')
#pylab.show()
#
#pylab.plot(time_dynamics, pre_on_nuclei_dynamics,'y', label = 'pre_nucl._on_pathway')
#pylab.plot(time_dynamics, fibrils_dynamics,'r', label = 'fibrils')
#pylab.plot(time_dynamics, pre_off_nuclei_dynamics,'c', label = 'pre_nucl._off_pathway')
#pylab.plot(time_dynamics, off_nuclei_dynamics,'g', label = 'off_pathway_nuclei')
#pylab.xlabel('$Time\ (h)$', fontsize = 14)
#pylab.ylabel(r'$N$', fontsize = 14)
##pylab.ylim([0.1,1600])
##pylab.yscale('log')
#pylab.title(str(N)+ ' ' +'particles' +','+ ' '+ str(iter_nums)+' '+'runs',  fontsize = 14)
#pylab.legend(loc= 'upper right')
#pylab.savefig('off_trial2.png')
#pylab.savefig('900.svg')
#pylab.show()
