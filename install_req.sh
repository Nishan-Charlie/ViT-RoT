
#!/bin/bash
pip install  bcolz mxnet tensorboardX matplotlib easydict opencv-python einops --no-cache-dir -U | cat
pip install  scikit-image imgaug PyTurboJPEG --no-cache-dir -U | cat
pip install  scikit-learn --no-cache-dir -U | cat
#pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html  --no-cache-dir -U | cat
pip install   termcolor imgaug prettytable --no-cache-dir -U | cat
pip install  timm==0.3.4 --no-cache-dir -U | cat

