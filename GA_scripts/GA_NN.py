#!/usr/bin/env python3 -u
# -----------------------------------------------------------
# Genetic Algorithm code for Metal Phthalocyanine           
# 
# Copyright (C) 2020 Zisheng Zhang
# Email: zisheng@chem.ucla.edu
# -----------------------------------------------------------
# Dependencies: Python3; ASE; OpenBabel; xTB
# -----------------------------------------------------------

import os, sys, subprocess
import multiprocessing as mp
import ase.io as ai
import numpy as np
from time import ctime
import torch

# Dictionary LIST of SMILES strings of substituent groups (10 per line)
grpDict = [
    '[H]', 'OC', 'C#N', 'C', 'CC', 'CCC', 'C(C)C', 'CCCC', 'C(C)CC', 'C(C)(C)C',
   'C=C','C#C','C[Cl]','C([F])([F])[F]','C=O','C(=O)C','C(=O)O','C(=O)OC','[Si](C)(C)C','[F]',
   '[Cl]', '[Br]', '[I]', 'O', 'OCC','OC=O', 'OC(=O)C', 'S', 'SC', 'S(=O)(=O)C',
   'P', 'PC', 'PCC', 'N(=O)=O', 'N','NC', 'N(C)C', 'NC(=O)', 'NC(=O)C', 'B',
   'C=N', 'C(=O)N'
]
# Global home directory for the whole run
homedir = os.getcwd()
nnpot = torch.load('model-pot.pkl')
nnsta = torch.load('model-sta.pkl')

def gene2smi(gene):
    '''Convert a gene to corresponding SMILES code.
    [gene]: a gene LIST of integers (indeces for grpDict).'''
    gene_i = [grpDict[i] for i in gene[:8]]
    gene_o = [grpDict[i] for i in gene[-8:]]
    return  'C1(%s)=C(%s)C(%s)=C2C(=C1(%s))C3=NC4=NC(=NC5=NC(=NC6=NC(=NC2=N3)C7=C(%s)C(%s)=C(%s)C(%s)=C76)C8=C(%s)C(%s)=C(%s)C(%s)=C85)C9=C(%s)C(%s)=C(%s)C(%s)=C94'%(
            gene_i[0], gene_i[1], gene_o[0], gene_o[1],
            gene_o[2], gene_i[2], gene_i[3], gene_o[3],
            gene_o[4], gene_i[4], gene_i[5], gene_o[5],
            gene_o[6], gene_i[6], gene_i[7], gene_o[7]
        )

def smi2xyz(smi):
    '''Convert a SMILES code to Atoms object using OpenBabel.
    Please cite: *J. Cheminf.* (2019) **11**, Art. 49.
    [smi]:  a SMILES string.'''
    # You may have to change the name or path of your OpenBabel executable
    subprocess.run([
        "obabel",
        '-:%s'%smi,
        '-oxyz',
        '-h',
        '--gen3D',
        '-O',
        'tmp.xyz'
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    s = ai.read('tmp.xyz')
    subprocess.run(['rm', 'tmp.xyz'])
    # Recenter the molecule. Works not good for very aymmetric molecules
    center = s.get_center_of_mass()
    s.translate( - center)
    return s

def ifDuplicate(gene, pool):
    '''Check if an gene is already in a gene pool.
    Canonical gene LIST input is expected since circular boundary conditions is applied.
    [gene]: a gene LIST.
    [pool]: a LIST of gene LIST.'''
    shiftRight = lambda lst: [lst[-4]] + lst[:-4]
    gene_tmp = gene.copy()
    pool_tmp = pool.copy()
    gene_tmp = ringGene(gene_tmp)
    pool_tmp = [ringGene(g) for g in pool_tmp]
    dup = False
    for i in range(4):
        gene_tmp = shiftRight(gene_tmp)
        if gene_tmp in pool or reversed(gene_tmp) in pool:
            dup = True
            break
    return dup

def addFrag(molBase, fragName):
    '''Add a molecular fragment to a Atoms object.
    [molBase]:  an Atoms object as the base structure.
    [fragName]: the name of XYZ file containing the fragment.'''
    s = molBase.copy()
    s.extend(ai.read(fragName))
    return s

def ringGene(gene):
    '''Convert a gene LIST to the ring order.
    [gene]:     an gene LIST'''
    return [gene[3],  gene[0],  gene[1],  gene[2],
            gene[4],  gene[5],  gene[6],  gene[7],
            gene[8],  gene[9],  gene[10], gene[11],
            gene[12], gene[13], gene[14], gene[15],
    ]

def gene2str(geneLst):
    '''Convert a gene LIST to gene STRING'''
    tmp = [str(i) for i in geneLst]
    return '-'.join(tmp)

def str2gene(geneStr):
    '''Convert a gene STRING to gene LIST'''
    tmp = [eval(i) for i in geneStr.split('-')]
    return tmp

def mutate(geneLst, dictMut, numMut):
    '''Introduce some mutation into a gene LIST according to a dictionary.
    [geneLst]:  a gene LIST or STRING.
    [dictMut]:  the dictionaty used to define possible mutations.
    [numMut]:   number of mutations to be introduced.'''
    if type(geneLst) == type('s'):
        gene_tmp = str2gene(geneLst)
    else:
        gene_tmp = geneLst.copy()
    for i in np.random.choice(len(gene_tmp)-8, size=numMut, replace=False):
        # MODIFIED TO RESTRICT THE POSITION TO OUTER SIDE
         gene_tmp[i] = np.random.choice(len(dictMut))
    return gene_tmp

def simCheck(gene1, gene2):
    if type(gene1) == type('s'):
        gene1, gene2 = str2gene(gene1), str2gene(gene2)
    sim = 0
    for i in range(len(gene1)):
        if gene1[i] == gene2[i]: sim += 1
    return sim/len(gene1)

def mating(gene1, gene2):
    '''Breed an offspring with random genetic crossover.
    [gene1&2]:  two gene LISTs or STRINGs.'''
    if type(gene1) == type('s'):
        gene1, gene2 = str2gene(gene1), str2gene(gene2)
    gene_tmp = []
    for i in range(len(gene1)):
        gene_tmp.append(np.random.choice([gene1[i], gene2[i]]))
    return gene_tmp

def mating2(gene1, gene2):
    '''Breed an offspring with random genetic crossover.
    [gene1&2]:  two gene LISTs or STRINGs.'''
    if type(gene1) == type('s'):
        gene1, gene2 = str2gene(gene1), str2gene(gene2)
    gene_tmp = gene2.copy()
    for i in np.random.choice(range(8), size=4,replace=False):
        gene_tmp[i] = gene1[i]
    return gene_tmp

def calculate(atoms):
    '''Launch xTB calculation and return selected values of an non-PBC Atoms object.
    NOTE: ONLY WORKS WITH METAL PHTHALOCYANINE GENERATED WITH THIS CODE!!!
    Returns: atomic charge of the Ni; bond order of Ni-N; LUMO energy; HOMO energy.'''
    ai.write('tmp.xyz', atoms)
    with open('out', 'w+') as fout:
        subprocess.run([
            'xtb',
            'tmp.xyz',
            '-opt',
            '-gfn',
            '2',
        ], stdout=fout, stderr=subprocess.DEVNULL)
    if 'wbo' not in os.listdir() or 'charges' not in os.listdir():
        return 0,0,0,0,atoms,0
    chrg = eval(open('charges').readlines()[-1])
    bond = [eval(i.split()[2]) for i in open('wbo').readlines()[-4:]]
    if min(bond) < 0.70 or max(bond) > 0.78:
        return 0,0,0,0,atoms,0
    bond = sum(bond)/4
    lumo = [i for i in open('out').readlines() if '(LUMO)' in i][-1]
    if 'NaN' in lumo:
        return 0,0,0,0,atoms,0
    lumo = eval(lumo.split()[-2])
    homo = [i for i in open('out').readlines() if '(HOMO)' in i][-1]
    if 'NaN' in homo:
        return 0,0,0,0,atoms,0
    homo = eval(homo.split()[-2])
    geom = ai.read('xtbopt.xyz')
    etot = [i for i in open('out').readlines() if 'TOTAL' in i][-1]
    if 'NaN' in etot:
        return 0,0,0,0,atoms,0
    etot = eval(etot.split()[-3])
    subprocess.run(['rm', 'charges', 'wbo', 'xtbopt.xyz',
                    'xtbrestart', 'xtbopt.log', 'out', 'tmp.xyz'])
    return chrg, bond, lumo, homo, geom, etot

def calcAdsGraphene(geom_mol, emol, esub=-483.051663392591):
    complx = addFrag(geom_mol, homedir+'/graphene.xyz')
    ai.write('tmp.xyz', complx)
    with open('out', 'w+') as fout:
        subprocess.run([
            'xtb',
            'tmp.xyz',
            '-opt',
            'loose',
            '-gfn',
            '2',
        ], stdout=fout, stderr=subprocess.DEVNULL)
    if 'out' not in os.listdir():
        return -100.0
    ecom = [i for i in open('out').readlines() if 'TOTAL' in i][-1]
    if 'NaN' in ecom:
        return -100.0
    ecom = eval(ecom.split()[-3])
    return ecom - emol - esub

def encode(geneLst):
    vec = []
    for i in geneLst[:8]:
        tmp = [0 for i in range(42)]
        tmp[i] = 1
        vec += tmp
    return np.array(vec)

def worker(geneLst):
    '''Worker function for parallelization. Calculate one gene in a tmp folder.'''
    inp = torch.tensor(encode(geneLst), dtype=torch.float)
    per = (nnpot(inp) + 1.2 * nnsta(inp))/2.2
    return float(per)

def descriptor(chrg, lumo, wt=[1.2, 1],\
#	ref=[-0.232506267,0.747526544,-8.168966667,-8.8394]):	# Param set on Bridges
    ref=[-0.233129,0.747946,-8.205,-8.883]):		    	# Param set on Taiyi
    '''Calculate self-defined descriptor using calculated quantities.'''
    stabilt = 100 * -(chrg - ref[0]) / (np.abs(ref[0]))
    overpot = 100 * -(lumo - ref[2]) / (np.abs(ref[2]))
    return stabilt, overpot, (stabilt*wt[0] + overpot*wt[1])/sum(wt)

def genSymm1(mydict):
    '''Generate 4-substituted MPc with all defined substituents. C4 or D4h symmetry'''
    tmp = []
    for i in range(len(mydict)):
        tmp.append([i,0,i,0,i,0,i,0,0,0,0,0,0,0,0,0])
    return tmp

def genSymm2(mydict):
    '''Generate 8-substituted MPc with all defined substituents. C4 or D4h symmetry'''
    tmp = []
    for i in range(len(mydict)):
        tmp.append([i,i,i,i,i,i,i,i,0,0,0,0,0,0,0,0])
    return tmp

def updateResult(present, history, alive):
    '''Update the results in report.log to history file'''
    with open(history, 'a') as fout:
        subprocess.run(['cat', present], stdout=fout)
    with open(alive, 'a') as fout:
        subprocess.run(['cat', present], stdout=fout)
    
def getFittest(alive, popSize):
    '''Rank the history and return the fittest population
    [history]:  the history file containing path.
    NOTE:   it returns the gene in STRING form!'''
    data = open(alive, 'r').readlines()
    data = list(set(data))
    data = [i.split() for i in data]
    data = [[i[0], eval(i[1])] for i in data]
    data.sort(key = lambda x: -x[1])
    f = open(alive, 'w')
    for d in data[:popSize]:
        f.write('%s\t%.7f\n'%(
            d[0], d[1]
        ))
    f.close()
    return data[:popSize]

def mateAssign(popDat, history, popSize=20, mutRate=0.2, mutNum=1):
    '''Assign which individuals to mate, and decide which to mutate.
    [popDat]:   gene LIST of the parent genration
    Returns a LIST of gene LIST of children generation.'''
    old = [i.split()[0] for i in open(history, 'r').readlines()]
    old = [ringGene(str2gene(i)) for i in old]
    children = []
    weight = np.array([i[1] for i in popDat])
    weight = weight - min(weight)
    weight = weight / max(weight)
    while len(children) < popSize:
        pair = np.random.choice(len(popDat), size=2, replace=False)
        rseed = np.random.rand(3)
        if rseed[0] > weight[pair[0]] and rseed[1] > weight[pair[1]]:
            continue
        child = mating2(popDat[pair[0]][0], popDat[pair[1]][0])
        sim = simCheck(popDat[pair[0]][0], popDat[pair[1]][0])
        if rseed[2] < mutRate * (1 + sim):
            child = mutate(child, grpDict, round(mutNum * (1 + sim)))
        if ifDuplicate(child, old) or ifDuplicate(child, children):
            continue
        print(' > new offspring: %s'%(gene2str(child)))
        children.append(child)
    return children

def randPop(seed, history, popSize=100, mutNum=8):
    old = [i.split()[0] for i in open(history, 'r').readlines()]
    old = [ringGene(str2gene(i)) for i in old]
    pop_rand = []
    while len(pop_rand) < popSize:
        tmp_rand = mutate(seed, grpDict, mutNum)
        if ifDuplicate(tmp_rand, old) or ifDuplicate(tmp_rand, pop_rand):
            continue
        pop_rand.append(tmp_rand)
    return pop_rand


def parCalc(genePop, par=4):
    '''Parallelization with multiprocessing.pool'''
    genePop = [gene2str(i) for i in genePop]
    genePop = list(set(genePop))
    genePop = [str2gene(i) for i in genePop]
    print(' --- Parallelized Batch Calculation ---')
    print(' > %i\tjobs,\t%i processes'%(len(genePop), par))
    proc_pool = mp.Pool(processes=par)
    # Result: chrg, bond, lumo, homo, dsta, dpot, dper, geom
    results = proc_pool.map(worker, genePop)
    print('\nFinished on %s!\n --- Post-processing the Results ---'%ctime())
    proc_pool.close()
    proc_pool.join()
    f = open('report.log', 'w+')
    traj = []
    for i in range(len(results)):
        f.write('%s\t%.7f\n'%(
            gene2str(genePop[i]),results[i]))
    f.close()

#nprocs = mp.cpu_count()
nprocs = 4
popSize, mutRate, numMut = 200, 0.33, 1
template=str2gene('0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0')



# Generate initial population
print(' --- Initial Population Generation ---')
subprocess.run(['mkdir', 'gen000'])
os.chdir(homedir + '/gen000')
symm = genSymm2(grpDict)+genSymm1(grpDict)
parCalc(symm, par = nprocs)
os.chdir(homedir)
updateResult('gen000/report.log', 'history.dat', 'alive.dat')

# Initiate the convergence settings
currGM = ''
convCount = 0

for n in range(1, 1000):
    parents = getFittest('alive.dat', popSize)
    # Check convergence
    if open('alive.dat', 'r').readlines()[0] == currGM:
        convCount += 1
    else:
        convCount = 0
        currGM = open('alive.dat', 'r').readlines()[0]
    print('Convergence: %i/100'%convCount)
    if convCount >= 100:
        print('Converged!')
        break
    # Launch new generation: get children
    print('\n --- STRATING GENERATION %i on %s---'%(n, ctime()))
    children = mateAssign(parents, homedir+'/history.dat',\
                          popSize, mutRate, numMut)
    wkdir = '%s/gen%s'%(homedir, str(n).zfill(3))
    subprocess.run(['mkdir', wkdir])
    os.chdir(wkdir)
    parCalc(children, par = nprocs)
    updateResult('report.log', homedir+'/history.dat', homedir+'/alive.dat')
    os.chdir(homedir)
