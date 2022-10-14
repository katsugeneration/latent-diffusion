wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh
bash Miniconda3-py38_4.12.0-Linux-x86_64.sh -b -f -p /usr/local
conda env update -n base -f environment-cpu.yaml
wget https://ommer-lab.com/files/latent-diffusion/celeba.zip -O models/ldm/celeba256/celeba-256.zip
cd models/ldm/celeba256 && unzip -o celeba-256.zip && cd -
wget -O models/first_stage_models/vq-f4/model.zip https://ommer-lab.com/files/latent-diffusion/vq-f4.zip
cd models/first_stage_models/vq-f4 && unzip -o model.zip && cd -