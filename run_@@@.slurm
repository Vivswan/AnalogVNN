#!/bin/bash
#SBATCH --job-name=run_@@@RunScript@@@_@@@run_combination@@@
#SBATCH --output=slurm_%x_%A.out
#SBATCH --mail-user=vis77@pitt.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=@@@cpu@@@
#SBATCH --cluster=gpu
#SBATCH --partition=@@@partition@@@
#SBATCH --gres=gpu:1
#SBATCH --time=@@@time@@@
#SBATCH --chdir="/ihome/nyoungblood/vis77"
#SBATCH --requeue
@@@extra@@@

source ~/.bashrc

RunDirectoryLocation=$HOME/Vivswan-AnalogVNN
RunScript=@@@RunScript@@@.py
CondaEnv="${HOME}"/storage/envs/AnalogVNN_"${CPU_ARCHITECTURE}"_3.7

StorageDirectory="${HOME}/storage/"


cd ~ || exit
conda activate $CondaEnv

echo ""
echo "####################################### nvidia-smi #######################################"
echo ""
nvidia-smi
echo ""
echo ""

echo ""
echo "####################################### lsb_release #######################################"
echo ""
/usr/bin/lsb_release -a
echo ""
echo ""

echo ""
echo "####################################### printenv #######################################"
echo ""
printenv
echo ""
echo ""

echo "####################################### Conda Environment #######################################"
echo ""
echo ""
conda list
echo ""
echo ""

SlurmScratchName="slurm_${SLURM_JOB_NAME}_${SLURM_JOBID}"
SlurmScratchDirectory="${SLURM_SCRATCH}/${SlurmScratchName}"
mkdir -p "${SlurmScratchDirectory}"
mkdir -p "${SlurmScratchDirectory}"/_results
mkdir -p "${SlurmScratchDirectory}"/_results/datasets/
mkdir -p "${SlurmScratchDirectory}"/_results/runtime/
mkdir -p "${SlurmScratchDirectory}"/_results/models/
mkdir -p "${SlurmScratchDirectory}"/_results/tensorboard/
mkdir -p "${SlurmScratchDirectory}"/_results/logs/
rsync -ar "${HOME}"/storage/_datasets/ "${SlurmScratchDirectory}"/_results/datasets/
rsync -ar "${RunDirectoryLocation}"/ "${SlurmScratchDirectory}"/

run_on_exit(){
  echo "####################################### Exit Began #######################################"
  rm -rf "${SlurmScratchDirectory}/_results/datasets/"
  cd "${SLURM_SCRATCH}" || exit
  tar -czf "${SlurmScratchName}.tar.gz" "${SlurmScratchName}"
  mv       "${SlurmScratchName}.tar.gz" "${StorageDirectory}"

  echo ""
  echo "####################################### Billing #######################################"
  echo ""
  sacct -M gpu -j $SLURM_JOBID --format=AllocTRES%50,elapsed
  echo ""

  echo ""
  echo "####################################### crc-job-stats.py #######################################"
  echo ""
  crc-job-stats.py
  echo ""
  echo "!!!!!!Completed!!!!!!!"
  echo ""
}
trap run_on_exit EXIT

cd "${SlurmScratchDirectory}" || exit

echo ""
echo "####################################### Variables #######################################"
echo ""
echo "CPU_ARCHITECTURE = ${CPU_ARCHITECTURE}"
echo "SlurmScratchName = ${SlurmScratchName}"
echo "StorageDirectory = ${StorageDirectory}"
echo "SlurmScratchDirectory = ${SlurmScratchDirectory}"
echo "RunDirectoryLocation = ${RunDirectoryLocation}"
echo "RunScript = ${RunScript}"
echo "Array ID = ${SLURM_ARRAY_TASK_ID}"
echo "CondaEnv = ${CondaEnv}"
echo "which conda = $( which conda )"
echo "which python = $( which python )"
echo "which python3 = $( which python3 )"
echo "pytorch version" = $( python3 -c "import torch; print(torch.__version__)" )
echo "tensorflow version" = $( python3 -c 'import tensorflow as tf; print(tf.__version__)' )
echo "tensorboard version" = $( python3 -c 'from tensorboard import version; print(version.VERSION)' )
echo ""
echo ""

echo ""
echo "####################################### Main Program: Starting #######################################"
echo ""

# time \
srun python3 $RunScript \
--memory_required @@@VideoMemoryRequired@@@ \
--data_folder ./_results \
--run_combination @@@run_combination@@@ \
--cuda_memory @@@cuda_memory@@@ \
--tensorboard \
--save_data \
#--single_run
#exit 125


echo ""
echo "####################################### Main Program: Finished #######################################"
echo ""

