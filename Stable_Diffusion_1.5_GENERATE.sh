#!/bin/bash
#SBATCH --job-name=stable_diffusion
#SBATCH --partition=boost_usr_prod  # Usa la coda booster per le GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --gpus-per-node=3
#SBATCH --time=01:00:00
#SBATCH --output=output.txt

# Attiviamo l'ambiente virtuale
source ~/stable-diffusion-env/bin/activate

# Eseguiamo il codice per generare un'immagine
python ~/stable-diffusion/Stable_Diffusion_1.5_CODE.py
