# 1. Requirements

There are multiple possible installation options. Here, we provide the environment setup used in our local experiments for reproducibility:

# Environments:
cuda==12.1
python==3.10

# Dependencies:
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install natten==0.17.4+torch210cu121 -f https://shi-labs.com/natten/wheels/
pip install timm==0.6.0
pip install mmengine==0.2.0
pip install numpy scipy opencv-python pyyaml tqdm
