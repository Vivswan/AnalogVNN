#!/bin/bash
#SBATCH --job-name=analog_vnn_run_combination_$$$
#SBATCH --output=slurm_%x_%A.out
#SBATCH --mail-user=vis77@pitt.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --cluster=gpu
#SBATCH --partition=power9
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --chdir="/ihome/nyoungblood/vis77"

source ~/.bashrc

VideoMemoryRequired=1800
RunDirectoryLocation=$HOME/Vivswan-AnalogVNN
RunScript=parallel_main.py
CondaEnv="${HOME}"/storage/envs/AnalogVNN_"${CPU_ARCHITECTURE}"_3.7

StorageDirectory="${HOME}/storage/"


cd ~ || exit
conda activate $CondaEnv

if [[ $SLURM_JOB_PARTITION = "a100" ]]; then
  (( NumberProcesses=40536/$VideoMemoryRequired ))
elif [[ $SLURM_JOB_PARTITION = "gtx1080" ]]; then
  (( NumberProcesses=8119/$VideoMemoryRequired ))
elif [[ $SLURM_JOB_PARTITION = "power9" ]]; then
  (( NumberProcesses=32510/$VideoMemoryRequired ))
elif [[ $SLURM_JOB_PARTITION = "titanx" ]]; then
  (( NumberProcesses=12212/$VideoMemoryRequired ))
elif [[ $SLURM_JOB_PARTITION = "v100" ]]; then
  (( NumberProcesses=32510/$VideoMemoryRequired ))
elif [[ $SLURM_JOB_PARTITION = "k40" ]]; then
  (( NumberProcesses=6144 ))
elif [[ $SLURM_JOB_PARTITION = "scavenger" ]]; then
  (( NumberProcesses=32510/$VideoMemoryRequired ))
else
  NumberProcesses=1
fi

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
  tar -czvf "${SlurmScratchName}.tar.gz" "${SlurmScratchName}"
  mv "slurm_${SLURM_JOB_NAME}_${SLURM_JOBID}.tar.gz" "${StorageDirectory}"

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
#trap run_on_exit FAIL

cd "${SlurmScratchDirectory}" || exit

echo ""
echo "####################################### Variables #######################################"
echo ""
echo "CPU_ARCHITECTURE = ${CPU_ARCHITECTURE}"
echo "VideoMemoryRequired = ${VideoMemoryRequired}"
echo "NumberProcesses = ${NumberProcesses}"
echo "StorageDirectory = ${StorageDirectory}"
echo "SlurmScratchDirectory = ${SlurmScratchDirectory}"
echo "RunDirectoryLocation = ${RunDirectoryLocation}"
echo "RunScript = ${RunScript}"
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
python3 $RunScript \
--num_process $NumberProcesses \
--data_folder ./_results \
--run_combination $$$ \
--tensorboard \
--save_data \
#--single_run
#exit 125


echo ""
echo "####################################### Main Program: Finished #######################################"
echo ""

