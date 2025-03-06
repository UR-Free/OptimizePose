import os
from openff.toolkit import Molecule
from openmmforcefields.generators import SystemGenerator
from openmm import unit, LangevinIntegrator
from openmm.app import PDBFile, Simulation
from pdbfixer import PDBFixer
import traceback
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from rdkit import Chem
import warnings
from openmm import unit, Platform, State
from joblib import wrap_non_picklable_objects
from joblib import delayed
import re, sys


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cleanup import clean_structure,fix_pdb
from openmm.app.internal.pdbstructure import PdbStructure
import io
import subprocess
from loguru import logger


def read_molecule(molecule_file, sanitize=False, calc_charges=False, remove_hs=False):
    if molecule_file.endswith('.mol2'):
        mol = Chem.MolFromMol2File(molecule_file, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.sdf'):
        supplier = Chem.SDMolSupplier(molecule_file, removeHs=False)
        mols = [mol for mol in supplier if mol is not None]
        mol = mols[0]
    elif molecule_file.endswith('.pdbqt'):
        with open(molecule_file) as file:
            pdbqt_data = file.readlines()
        pdb_block = ''
        for line in pdbqt_data:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                content = line[:66]
                pdb_block += '{}\n'.format(content)
        mol = Chem.MolFromPDBBlock(pdb_block, sanitize=True, removeHs=False)
        mol_block = Chem.MolToMolBlock(mol)
        mol = Chem.MolFromMolBlock(mol_block, sanitize=True, removeHs=False)        
    elif molecule_file.endswith('.pdb'):
        mol = Chem.MolFromPDBFile(molecule_file, sanitize=False, removeHs=False)
    else:
        raise ValueError('Expect the format of the molecule_file to be '
                         'one of .mol2, .sdf, .pdbqt and .pdb, got {}'.format(molecule_file))
    
    
    try:
        if sanitize or calc_charges:
            Chem.SanitizeMol(mol)

        if calc_charges:
            # Compute Gasteiger charges on the molecule.
            try:
                AllChem.ComputeGasteigerCharges(mol)
            except:
                warnings.warn('Unable to compute charges for the molecule.')
        if remove_hs:
            mol = Chem.RemoveHs(mol, sanitize=sanitize)
    except Exception as e:
        logger.info(e)
        logger.info("RDKit was unable to read the molecule.")
        return None
    return mol

def UpdatePose(lig, system_generator, modeller, protein_atoms, out_dir, device_num=0):
    """优化分子构象并保存结果
    
    参数：
        lig: 配体文件路径
        system_generator: 力场系统生成器
        modeller: 包含蛋白质的OpenMM Modeller对象
        protein_atoms: 蛋白质原子列表
        out_dir: 输出目录
        device_num: GPU设备号
    
    返回：
        0 表示成功，1 表示失败
    """
    try:
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, os.path.basename(lig).split(".")[0] + '_minimized.sdf')

        # 读取配体分子
        dockingpose = read_molecule(lig, remove_hs=False, sanitize=True)
        lig_mol = Molecule.from_rdkit(dockingpose, allow_undefined_stereo=True)
        lig_mol.assign_partial_charges(partial_charge_method='gasteiger')
        
        # 转换为OpenMM拓扑结构
        lig_top = lig_mol.to_topology()
        
        # 将配体添加到Modeller中
        modeller.add(lig_top.to_openmm(), lig_top.get_positions().to_openmm())
        
        # 关键修改：先添加分子到system_generator
        system_generator.add_molecules([lig_mol])
        
        # 创建模拟系统
        system = system_generator.create_system(modeller.topology)
        
        # 固定蛋白质原子，使其在模拟中保持静止
        for atom in protein_atoms:
            system.setParticleMass(atom.index, 0.000*unit.dalton)
        
        # 开始模拟
        platform = GetPlatform()
        simulation = EnergyMinimized(modeller, system, platform, verbose=False, device_num=device_num)
        
        # 获取能量最小化后的构象
        ligand_atoms = list(filter(lambda atom: atom.residue.name in ['UNK', 'UNL'], list(modeller.topology.atoms())))
        ligand_index = [atom.index for atom in ligand_atoms]
        new_coords = simulation.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.angstrom)[ligand_index]
        
        # 更新RDKit分子的坐标
        lig_mol = lig_mol.to_rdkit()
        conf = lig_mol.GetConformer()
        for i in range(lig_mol.GetNumAtoms()):
            x, y, z = new_coords.astype(np.double)[i]
            conf.SetAtomPosition(i, Point3D(x, y, z))
        
        # 保存结果
        try:
            writer = Chem.SDWriter(out_file)
            writer.write(lig_mol)
            writer.close()
        except Exception as e:
            logger.info(f'写入文件 {out_file} 失败！')
            logger.info(e)
            
        return 0
    except Exception as e:
        error_info = traceback.format_exc()
        logger.info(error_info)
        logger.warning(f' : {e}')
        return 1

def DescribeState(state: State, name: str):
    """logger.info energy and force information about a simulation state."""
    max_force = max(np.linalg.norm([v.x, v.y, v.z]) for v in state.getForces())
    logger.info(f"{name} has energy {state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole):.2f} kJ/mol "
          f"with maximum force {max_force:.2f} kJ/(mol nm)")
    
def GetFFGenerator(protein_forcefield = 'amber/ff14SB.xml',water_forcefield = 'amber/tip3p_standard.xml',small_molecule_forcefield = 'openff-2.0.0',ignoreExternalBonds=False):
    """
    Get forcefield generator by different forcefield files
    """
    forcefield_kwargs = {'constraints': None, 'rigidWater': True, 'removeCMMotion': False, 'ignoreExternalBonds': ignoreExternalBonds, 'hydrogenMass': 4*unit.amu }
    # forcefield_kwargs = {'constraints': None, 'rigidWater': True, 'removeCMMotion': False, 'hydrogenMass': 4*unit.amu }
    system_generator = SystemGenerator(
                forcefields=[protein_forcefield, water_forcefield ],
                small_molecule_forcefield=small_molecule_forcefield,
                forcefield_kwargs=forcefield_kwargs)
    return system_generator


def GetfixedPDB(receptor_path):
    
    temp_fixd_pdbs = f'{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/fixed_pdbs'
    os.makedirs(temp_fixd_pdbs,exist_ok=True)
    if not os.path.exists(os.path.join(temp_fixd_pdbs,os.path.basename(receptor_path).replace('.pdb','_fixer_processed_cleanup.pdb'))):
        alterations_info  = {}
        fixed_pdb = fix_pdb(receptor_path, alterations_info)
        fixed_pdb_file = io.StringIO(fixed_pdb)
        pdb_structure = PdbStructure(fixed_pdb_file)
        clean_structure(pdb_structure, alterations_info)
        fixer = PDBFile(pdb_structure)
        logger.info("Protein loaded with success!")
        PDBFile.writeFile(fixer.topology, fixer.positions, open(os.path.join(temp_fixd_pdbs,os.path.basename(receptor_path).replace('.pdb','_fixer_processed_cleanup.pdb')), 'w'))
        logger.info('Dont have processed by fixer try fix and save in disk')
    else:
        fixer = PDBFixer(os.path.join(temp_fixd_pdbs,os.path.basename(receptor_path).replace('.pdb','_fixer_processed_cleanup.pdb')))
        logger.info('There have a precessed pdb file use it!')
    return fixer

import copy
from rdkit.Geometry import Point3D
def GetDockingPose(graph):
    mol = copy.deepcopy(graph.mol[0] if type(graph.mol) == list else graph.mol)
    mol = Chem.RemoveHs(mol)
    docking_position = graph['ligand'].pos.detach().cpu().numpy() # without Hs and dont match with raw pocket
    docking_position = docking_position + graph.original_center.detach().cpu().numpy()
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        x,y,z = docking_position.astype(np.double)[i]
        conf.SetAtomPosition(i,Point3D(x,y,z))
    return mol

@delayed
@wrap_non_picklable_objects
def GetPlatformPara():
    """Determine the best simulation platform available."""
    platform_name = os.getenv('PLATFORM')
    # properties = {'CudaDeviceIndex': '0'}
    if platform_name:
        platform = Platform.getPlatformByName(platform_name)
    else:
        platform = max((Platform.getPlatform(i) for i in range(Platform.getNumPlatforms())), key=lambda x: x.getSpeed())
    logger.info(f'Using platform {platform.getName()}')
    if platform.getName() in ['CUDA', 'OpenCL']:
        platform.setPropertyDefaultValue('Precision', 'mixed')
        logger.info(f'Set precision for platform {platform.getName()} to mixed')
    return platform
# @delayed
# @wrap_non_picklable_objects
def GetPlatform():
    """Determine the best simulation platform available."""
    platform_name = os.getenv('PLATFORM')
    # properties = {'CudaDeviceIndex': '0'}
    if platform_name:
        platform = Platform.getPlatformByName(platform_name)
    else:
        platform = max((Platform.getPlatform(i) for i in range(Platform.getNumPlatforms())), key=lambda x: x.getSpeed())
    logger.info(f'Using platform {platform.getName()}')
    if platform.getName() in ['CUDA', 'OpenCL']:
        platform.setPropertyDefaultValue('Precision', 'mixed')
        logger.info(f'Set precision for platform {platform.getName()} to mixed')
    return platform

def EnergyMinimized(modeller,system, platform,verbose=False,device_num = 0):
    integrator = LangevinIntegrator(
    300 * unit.kelvin,
    1 / unit.picosecond,
    0.002 * unit.picoseconds,
    )
    properties = {'CudaDeviceIndex': f'{device_num}'}
    simulation = Simulation(modeller.topology, system = system, integrator = integrator, platform=platform,platformProperties=properties)
    simulation.context.setPositions(modeller.positions)
    if verbose:
        DescribeState(
            simulation.context.getState(
                getEnergy=True,
                getForces=True,
            ),
            "Original state",
        )


    simulation.minimizeEnergy()
    if verbose:
        DescribeState(
            simulation.context.getState(
                getEnergy=True, 
                getForces=True),
            "Minimized state",
        )
    return simulation