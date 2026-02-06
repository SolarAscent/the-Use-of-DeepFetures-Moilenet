import random
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageEnhance
import config
from multiprocessing import Pool, cpu_count

# ---------- split dataset ----------
def split_dataset():
    """
    60%/20%/20%:train/val/test
    """
    if not config.DATA_ROOT.exists():
        raise FileNotFoundError(f'folder {config.DATA_ROOT} isn\'t exist')
    
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    
    print('Splitting the dataset (60% train / 20% val / 20% test)...')
    
    for split in config.SPLITS:
        for class_name in config.CLASS_NAMES:
            (config.SPLIT_ROOT / split / class_name).mkdir(parents=True, exist_ok=True)
    
    total_stats = {'train': 0, 'val': 0, 'test': 0}
    
    for class_folder in sorted(config.DATA_ROOT.iterdir()):
        if not class_folder.is_dir():
            continue
        
        class_name = class_folder.name
        print(f'  Class: {class_name}')
        
        image_files = []
        for ext in config.IMAGE_EXTENSIONS:
            image_files.extend(list(class_folder.rglob(ext)))
        
        if len(image_files) == 0:
            print(f'    Error: {class_name} isn\'t contain any image files!')
            print(f'    make sure {class_folder} folder contains images')
            continue
        
        random.shuffle(image_files)
        
        total = len(image_files)
        train_end = int(total * 0.6)
        val_end = train_end + int(total * 0.2)
        
        train_files = image_files[:train_end]
        val_files = image_files[train_end:val_end]
        test_files = image_files[val_end:]
        
        # copy files
        for idx, img_path in enumerate(tqdm(train_files, desc=f'    copy to train')):
            dest = config.SPLIT_ROOT / 'train' / class_name / f'{idx:05d}_{img_path.name}'
            shutil.copy2(img_path, dest)
            total_stats['train'] += 1
        
        for idx, img_path in enumerate(tqdm(val_files, desc=f'    copy to val')):
            dest = config.SPLIT_ROOT / 'val' / class_name / f'{idx:05d}_{img_path.name}'
            shutil.copy2(img_path, dest)
            total_stats['val'] += 1
        
        for idx, img_path in enumerate(tqdm(test_files, desc=f'    copy to test')):
            dest = config.SPLIT_ROOT / 'test' / class_name / f'{idx:05d}_{img_path.name}'
            shutil.copy2(img_path, dest)
            total_stats['test'] += 1
        
        print(f'    {class_name}: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}')
    
    print(f'\nComplete dataset split:')
    print(f'  train: {total_stats["train"]} pics')
    print(f'  val:   {total_stats["val"]} pics')
    print(f'  test:  {total_stats["test"]} pics')
    print(f'  total: {sum(total_stats.values())} pics')
    
    # check
    if sum(total_stats.values()) == 0:
        raise ValueError(
            f'\nError：No images found in {config.DATA_ROOT}\n'
            f'Please make sure {config.DATA_ROOT} bird、cat、dog folders exist and contain images.\n'
        )

# ---------- Enchance----------

def augment_image(img):
    augmented_images = []
    width, height = img.size
    
    # Rotation (-30,+30 degrees)
    for angle in [-30, -15, 15, 30]:
        augmented_images.append(img.rotate(angle, fillcolor=(128, 128, 128)))
    
    # Translation (translation in all directions)
    for dx, dy in [(10, 0), (-10, 0), (0, 10), (0, -10), (10, 10), (-10, -10)]:
        augmented_images.append(img.transform(
            (width, height), Image.AFFINE, (1, 0, dx, 0, 1, dy), fillcolor=(128, 128, 128)
        ))
    
    # Black and white (grayscale)
    if img.mode == 'RGB':
        augmented_images.append(img.convert('L').convert('RGB'))
    
    # Colour inversion
    augmented_images.append(Image.fromarray(255 - np.array(img)))
    
    # Brightness adjustment
    enhancer = ImageEnhance.Brightness(img)
    for factor in [0.5, 0.7, 1.3, 1.5]:
        augmented_images.append(enhancer.enhance(factor))
    
    # Contrast enhancement and reduction
    enhancer = ImageEnhance.Contrast(img)
    for factor in [0.5, 0.7, 1.3, 1.5]:
        augmented_images.append(enhancer.enhance(factor))
    
    
    img_array = np.array(img, dtype=np.float32)
    
    # Gaussian noise
    noise = np.random.normal(0, 15, img_array.shape).astype(np.float32)
    augmented_images.append(Image.fromarray(np.clip(img_array + noise, 0, 255).astype(np.uint8)))
    
    # salt-and-pepper noise
    noisy_saltpepper = img_array.copy()
    salt_pepper_ratio = 0.05
    num_salt = np.ceil(salt_pepper_ratio * img_array.size * 0.5)
    num_pepper = np.ceil(salt_pepper_ratio * img_array.size * 0.5)
    
    # white dots
    coords = [np.random.randint(0, i-1, int(num_salt)) for i in img_array.shape]
    noisy_saltpepper[coords[0], coords[1], coords[2]] = 255
    
    # black dots
    coords = [np.random.randint(0, i-1, int(num_pepper)) for i in img_array.shape]
    noisy_saltpepper[coords[0], coords[1], coords[2]] = 0
    
    augmented_images.append(Image.fromarray(noisy_saltpepper.astype(np.uint8)))
    
    # Rotation + Brightness
    rotated_bright = img.rotate(15, fillcolor=(128, 128, 128))
    augmented_images.append(ImageEnhance.Brightness(rotated_bright).enhance(1.2))
    
    # Translation + Contrast
    translated_contrast = img.transform(
        (width, height), Image.AFFINE, (1, 0, 5, 0, 1, 5), fillcolor=(128, 128, 128)
    )
    augmented_images.append(ImageEnhance.Contrast(translated_contrast).enhance(1.2))
    
    return augmented_images

def _worker_process_image(args):
    img_path, class_folder = args
    try:
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        base_name = img_path.stem
        
        for idx, aug_img in enumerate(augment_image(img)):
            # convent to .jpg
            aug_path = class_folder / f'{base_name}_aug{idx}.jpg'
            aug_img.save(aug_path, quality=95)
        return True
    except Exception as e:
        return f"Error {img_path}: {e}"


def augment_dataset():
    print('\n Start processing data augmentation...')
    
    tasks = []
    for split in ['train', 'val']:
        split_dir = config.SPLIT_ROOT / split
        if not split_dir.exists(): continue
        
        for class_folder in sorted(split_dir.iterdir()):
            if not class_folder.is_dir(): continue
            
            all_images = []
            for ext in config.IMAGE_EXTENSIONS:
                all_images.extend(list(class_folder.rglob(ext)))
            
            original_images = [img for img in all_images if '_aug' not in img.stem]
            for img_path in original_images:
                tasks.append((img_path, class_folder))

    num_workers = 16 
    with Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap_unordered(_worker_process_image, tasks), total=len(tasks), desc="the main process"))

    print(f'\ndata enhancement completed using multiprocessing.')

def count_images_in_split():
    total_files = 0
    for split in config.SPLITS:
        for class_name in config.CLASS_NAMES:
            split_dir = config.SPLIT_ROOT / split / class_name
            if split_dir.exists():
                for ext in config.IMAGE_EXTENSIONS:
                    total_files += len(list(split_dir.glob(ext)))
    return total_files
