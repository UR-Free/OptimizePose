import os, sys
from openmm.app import Modeller, ForceField
from joblib import Parallel,delayed
import argparse
from glob import glob
import traceback
import time
import numpy as np
from loguru import logger

from utils import GetfixedPDB, GetFFGenerator, UpdatePose, GetPlatformPara

path_to_remove = "/opt/software/amber20/"
sys.path = [p for p in sys.path if p not in path_to_remove]

if __name__ == '__main__':
    logger.remove()  # 移除默认处理器
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True,
        backtrace=True,  # 详细的异常回溯
        diagnose=True    # 诊断信息
    )
    # 添加文件日志记录
    logger.add(
        "minimization.log",
        rotation="10 MB",  # 达到10MB轮转
        compression="zip",
        level="INFO"
    )
    
    parser = argparse.ArgumentParser(description='Process protein-ligand files.')
    parser.add_argument('--num_process', type=int, default=35, help='Number of parallel workers.')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device index.')
    parser.add_argument('--protein', type=str, required=True, help='Path to protein PDB file')
    parser.add_argument('--ligand_dir', type=str, required=True, help='Path to a SDF files')
    parser.add_argument('--out_dir', type=str, default='./minimized_results', help='Directory to save results')
    args = parser.parse_args()
    os.environ['OMP_NUM_THREADS'] = '1'
    
    # 初始化力场
    start_time = time.time()
    platform = GetPlatformPara()
    system_generator = GetFFGenerator(ignoreExternalBonds=True)
    system_generator_gaff = GetFFGenerator(small_molecule_forcefield = 'gaff-2.11', ignoreExternalBonds=True)
    
    # 确保输出目录存在
    os.makedirs(args.out_dir, exist_ok=True)
    
    protein_path = args.protein
    compound_path = args.ligand_dir
    
    # 从蛋白质文件名获取蛋白ID
    pdbid = os.path.basename(protein_path).split()[0]
    try:
        logger.info(f"开始为靶点 {pdbid} 进行能量最小化...")
        logger.info("使用默认力场")
        receptor_path = protein_path
        
        # 加载并修复蛋白质
        fixer = GetfixedPDB(receptor_path)
        modeller = Modeller(fixer.topology, fixer.positions)
        disulfide_bond_list = []
        for bond in modeller.topology.bonds():
            if bond.atom1.name == 'SG' and bond.atom2.name == 'SG':
                disulfide_bond_list.append(bond)
        modeller.delete(disulfide_bond_list)
        forcefield = ForceField('amber14/protein.ff14SB.xml')
        modeller.addHydrogens(forcefield)
        protein_atoms = list(modeller.topology.atoms())
        
        # 处理配体文件
        logger.info(f"将处理化合物文件: {compound_path}")
        
        files = glob(os.path.join(compound_path, '*.*'))
        
        logger.info(f"将对文件中的{len(files)}个化合物进行能量最小化")
        
        
        with Parallel(n_jobs=args.num_process,) as parallel:
            new_data_list = parallel(delayed(UpdatePose)(lig_path,system_generator,modeller,protein_atoms,args.out_dir) for lig_path in files)
            # selected the failed samples and try to use gaff-2.11 forcefield
            if sum(new_data_list) != 0:
                result = np.array(new_data_list)
                indices = np.where(result == 1)
                failed_files = [files[i] for i in indices[0]]
                logger.info(f'openff 优化未完成的样本数: {len(failed_files)}')
                with Parallel(n_jobs=args.num_process) as parallel:
                    new_data_list = parallel(delayed(UpdatePose)(lig_path,system_generator_gaff,modeller,protein_atoms,args.out_dir) for lig_path in failed_files)
        
        # 如果仍有失败的样本，保存未优化的构象
        if sum(new_data_list) != 0:
            logger.warning(f'优化未完全完成')
        
        logger.success(f'完成靶点 {pdbid} 的能量最小化')
        
    except Exception as e:
        logger.error(f"处理靶点 {pdbid} 时出错: {str(e)}")
        logger.error(traceback.format_exc())
        
    end_time = time.time()
    logger.success(f"优化完成，共用时间: {end_time - start_time:.2f} 秒")