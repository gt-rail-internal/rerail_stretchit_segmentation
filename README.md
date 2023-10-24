# rerail_stretchit_segmentation

This repo uses Detic to get the segmentation data of all objects in the input image.

## Majority of the contents of this repo are either copied directly from or modified versions of the original Detic repo from facebook. https://github.com/facebookresearch/Detic/tree/main.

To setup Detic, you can follow the following steps:
1. Install CUDA 11.3 if you haven't (or whatever version that is compatible with Detic - please check out their documentation). The link below will work you through the steps to install CUDA 11.3 specifically.
   - https://gist.github.com/Mahedi-61/2a2f1579d4271717d421065168ce6a73
2. Install pytorch, torchvision, torchaudio that is compatible with CUDA 11.3
   -  pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
3. Install Detectron2 e.g.
   -- Do this under your working directory --
   - git clone https://github.com/facebookresearch/detectron2.git
   - cd detectron2
   - pip install -e .
4. Install the edited version of Detic from this repo.
   - cd ..
   - git clone https://github.com/gt-rail-internal/rerail_stretchit_segmentation.git --recurse-submodules
   - cd rerail_stretchit_segmentation/Detic
   - pip install -r requirements.txt
   - replace the 'visualizer.py' file in the detectron2 folder with the 'visualizer.py' file from the Detic repo

- Please note: you would need to download the model weights before running.

After pulling from this repo, do the following in your terminal to download the model weights:
1. cd rerail_stretchit_segmentation/Detic
2. mkdir models
4. wget https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth -O models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth

- See the 'get_segmentation_data.py' code for a sample on how to call the 'detect' function from 'demo_derail.py' in your code. Please do not move the 'demo_derail.py' code out of it's current directory as it relies on many of the other files in that directory. Consider adding the path to 'demo_derail.py' into whatever code you're working on if it is in a different directory.
