# rerail_stretchit_segmentation

This repo uses Detic to get the segmentation data of all objects in the input image.

Majority of the contents of this repo are either copied directly from or modified versions of the original Detic repo from facebook. https://github.com/facebookresearch/Detic/tree/main
To setup Detic, you can follow the following steps:
1. Create a conda environment, activate it, and download pytorch, torchvision, and torchaudio e.g.
   conda create --name detic python=3.8 -y
   conda activate detic
   conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
2. Install Detectron2 e.g.
   -- Do this under your working directory --
   git clone git@github.com:facebookresearch/detectron2.git
   cd detectron2
   pip install -e .
3. Install the edited version of Detic from this repo.
   cd ..
   git clone https://github.com/TofunmiSodimu/rerail_stretchit_segmentation/Detic.git --recurse-submodules
   cd Detic
   pip install -r requirements.txt

Please note: you would need to download the model weights before running.
After pulling from this repo, do the following in your terminal to download the model weights:
1. cd Detic
2. mkdir models
3. wget https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth -O models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
