from pathlib import Path
import torch
import torchvision

# ---------- hyperparameters ----------
BATCH        = 256

try:
    _mobilenet = torchvision.models.mobilenet_v2(weights=None)
except:
    _mobilenet = torchvision.models.mobilenet_v2(pretrained=False)
TOTAL_LAYERS = len(list(_mobilenet.features))
CUTS         = list(range(1, TOTAL_LAYERS + 1))  

DATA_ROOT    = Path('data')        
SPLIT_ROOT   = Path('data_split')  
RESULT_FIG   = 'result.png'
RESULT_TXT   = 'result.txt'
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
RANDOM_SEED  = 42                  


IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']
CLASS_NAMES = ['bird', 'cat', 'dog']
SPLITS = ['train', 'val', 'test']
