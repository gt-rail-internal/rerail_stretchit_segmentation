# rerail_stretchit_segmentation

This repo uses Detic to get the segmentation data of all objects in the input image.

## Majority of the contents of this repo are either copied directly from or modified versions of the original Detic repo from facebook. https://github.com/facebookresearch/Detic/tree/main.

To setup Detic, you can follow the following steps (cuda should alreday be installed on your system before running these steps):
1. Create a conda environment, activate it, and download pytorch, torchvision, and torchaudio e.g.
   - conda create --name detic python=3.8 -y
   - conda activate detic
   - conda install cudatoolkit=11.1 -c nvidia 
   - conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 -c pytorch
2. Install Detectron2 e.g.
   -- Do this under your working directory --
   - git clone https://github.com/facebookresearch/detectron2.git
   - cd detectron2
   - pip install -e .
3. Install the edited version of Detic from this repo.
   - cd ..
   - git clone https://github.com/gt-rail-internal/rerail_stretchit_segmentation.git --recurse-submodules
   - cd rerail_stretchit_segmentation/Detic
   - pip install -r requirements.txt

- Please note: you would need to download the model weights before running.

After pulling from this repo, do the following in your terminal to download the model weights:
1. cd Detic
2. mkdir models
3. wget https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth -O models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth

- See the 'get_segmentation_data.py' code for a sample on how to call the 'detect' function from 'demo_derail.py' in your code. Please do not move the 'demo_derail.py' code out of it's current directory as it relies on many of the other files in that directory. Consider adding the path to 'demo_derail.py' into whatever code you're working on if it is in a different directory.
