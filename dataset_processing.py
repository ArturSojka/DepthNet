import h5py
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import numpy as np
from datasets import load_dataset
import random
from tqdm import tqdm
from huggingface_hub import snapshot_download
import matplotlib.pyplot as plt

class DatasetProcessor:
    
    def __init__(self):
        self.outfile = "D:\\processed_dataset.h5"
        self.target_size = 480
        
    def crop_and_resize(self, image, depth_in_meters):
        height, width, _ = image.shape
        smaller_dim = min(width, height)

        x_offset = random.randint(0, width - smaller_dim)
        y_offset = random.randint(0, height - smaller_dim)

        image_cropped = image[y_offset:y_offset + smaller_dim, x_offset:x_offset + smaller_dim, :]
        depth_cropped = depth_in_meters[y_offset:y_offset + smaller_dim, x_offset:x_offset + smaller_dim]

        if smaller_dim != self.target_size:
            image_cropped = cv2.resize(image_cropped, (self.target_size,self.target_size), interpolation=cv2.INTER_LINEAR)
            depth_cropped = cv2.resize(depth_cropped, (self.target_size,self.target_size), interpolation=cv2.INTER_LINEAR)
            
        return image_cropped, depth_cropped
    
    def visualize_result(self, cmap='Spectral_r'):
        with h5py.File(self.outfile, "r") as h5f:
            images = h5f['images']
            depths = h5f['depths']
            
            n_images = images.shape[0]
            img1 = images[0,:,:,:]
            depth1 = depths[0,:,:]
            x = random.randint(1,n_images-2)
            img2 = images[x,:,:,:]
            depth2 = depths[x,:,:]
            img3 = images[n_images-1,:,:,:]
            depth3 = depths[n_images-1,:,:]
        
        
        fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3)
        
        ax1.imshow(img1)
        ax1.axis('off')
        ax1.set_title(f'image 1')

        ax2.imshow(img2)
        ax2.axis('off')
        ax2.set_title(f'image {x+1}')

        ax3.imshow(img3)
        ax3.axis('off')
        ax3.set_title(f'image {n_images}')
        
        ax4.imshow(depth1, cmap)
        ax4.axis('off')
        ax4.set_title(f'depth 1')

        ax5.imshow(depth2, cmap)
        ax5.axis('off')
        ax5.set_title(f'depth {x+1}')

        ax6.imshow(depth3, cmap)
        ax6.axis('off')
        ax6.set_title(f'image {n_images}')
        
        plt.show()

class NYUv2Processor(DatasetProcessor):
    
    def __init__(self):
        super(NYUv2Processor, self).__init__()
        self.outfile = "D:\\processed_nyu_depth_v2.h5"
        self.cache_dir = "D:\\"
        
    def download_data(self):
        load_dataset("0jl/NYUv2", trust_remote_code=True, cache_dir=self.cache_dir, split="train")
        
    def process(self):
        ds = load_dataset("0jl/NYUv2", trust_remote_code=True, cache_dir=self.cache_dir, split="train")
        with h5py.File(self.outfile, "w") as h5f:
            # Create datasets for images and depth maps
            n_images = len(ds)
            images = h5f.create_dataset("images", (n_images, 480, 480, 3), dtype="uint8")
            depths = h5f.create_dataset("depths", (n_images, 480, 480), dtype="float32")
            
            for idx,sample in enumerate(tqdm(ds)):
                img, depth = self.crop_and_resize(
                    np.array(sample['image'], dtype=np.uint8),
                    np.array(sample['depth'], dtype=np.float32)
                )
                images[idx] = img
                depths[idx] = 1.0 / (depth + 1e-6)
                
        print(f"Data saved to {self.outfile}.")

class CarlaHDProcessor(DatasetProcessor):
    
    def __init__(self):
        super(CarlaHDProcessor, self).__init__()
        self.outfile = "D:\\processed_carla_hd.h5"
        self.cache_dir = "D:\\"
        
    def download_data(self):
        load_dataset("naufalso/carla_hd", trust_remote_code=True ,cache_dir=self.cache_dir)
        
    def process(self):
        ds = load_dataset("naufalso/carla_hd", trust_remote_code=True ,cache_dir=self.cache_dir)
        with h5py.File(self.outfile, "w") as h5f:
            n_images = len(ds['train']) + len(ds['test']) + len(ds['validation'])
            images = h5f.create_dataset("images", (n_images, 480, 480, 3), dtype="uint8")
            depths = h5f.create_dataset("depths", (n_images, 480, 480), dtype="float32")
            idx = 0
            for sample in tqdm(ds['train']):
                raw_depth = np.array(sample['raw_depth'], dtype=np.float32)
                depth_in_meters = (raw_depth[:, :, 0] + raw_depth[:, :, 1] * 256 + raw_depth[:, :, 2] * 256 * 256) / (256 * 256 * 256 - 1) * 1000
                
                img, depth = self.crop_and_resize(
                    np.array(sample['rgb'], dtype=np.uint8),
                    depth_in_meters
                )
                images[idx] = img
                depths[idx] = 1.0 / (depth + 1e-6)
                idx += 1
            
            for sample in tqdm(ds['test']):
                raw_depth = np.array(sample['raw_depth'], dtype=np.float32)
                depth_in_meters = (raw_depth[:, :, 0] + raw_depth[:, :, 1] * 256 + raw_depth[:, :, 2] * 256 * 256) / (256 * 256 * 256 - 1) * 1000
                
                img, depth = self.crop_and_resize(
                    np.array(sample['rgb'], dtype=np.uint8),
                    depth_in_meters
                )
                images[idx] = img
                depths[idx] = 1.0 / (depth + 1e-6)
                idx += 1
                
            for sample in tqdm(ds['validation']):
                raw_depth = np.array(sample['raw_depth'], dtype=np.float32)
                depth_in_meters = (raw_depth[:, :, 0] + raw_depth[:, :, 1] * 256 + raw_depth[:, :, 2] * 256 * 256) / (256 * 256 * 256 - 1) * 1000
                
                img, depth = self.crop_and_resize(
                    np.array(sample['rgb'], dtype=np.uint8),
                    depth_in_meters
                )
                images[idx] = img
                depths[idx] = 1.0 / (depth + 1e-6)
                idx += 1
                
        print(f"Data saved to {self.outfile}.")

class UrbanSynProcessor(DatasetProcessor):
    
    def __init__(self):
        super(UrbanSynProcessor, self).__init__()
        self.outfile = "D:\\processed_urban_syn.h5"
        self.cache_dir = "D:\\"
        
    def download_data(self):
        snapshot_download(repo_id="UrbanSyn/UrbanSyn", repo_type="dataset", cache_dir=self.cache_dir)
        
    def process(self):
        img_file = "D:\\datasets--UrbanSyn--UrbanSyn\\snapshots\\2be11cba462f5f67d8a28f07cdff58c8bab646f4\\rgb\\rgb_{:>04}.png"
        depth_file = "D:\\datasets--UrbanSyn--UrbanSyn\\snapshots\\2be11cba462f5f67d8a28f07cdff58c8bab646f4\\depth\\depth_{:>04}.exr"
        with h5py.File(self.outfile, "w") as h5f:
            n_images = 7487
            images = h5f.create_dataset("images", (n_images, 480, 480, 3), dtype="uint8")
            depths = h5f.create_dataset("depths", (n_images, 480, 480), dtype="float32")
            idx = 0 
            for i in tqdm(range(1,7540)):
                if 6972 <= i <= 7028: # broken depths
                    continue
                img, depth = self.crop_and_resize(
                    cv2.cvtColor(cv2.imread(img_file.format(i)), cv2.COLOR_BGR2RGB),
                    cv2.imread(depth_file.format(i),cv2.IMREAD_ANYDEPTH) * 1.0e5
                )
                images[idx] = img
                depths[idx] = 1.0 / (depth + 1e-6)
                idx += 1
                
        print(f"Data saved to {self.outfile}.")
        
class DDOSProcessor(DatasetProcessor):
    
    def __init__(self):
        super(DDOSProcessor, self).__init__()
        self.outfile = "D:\\processed_ddos.h5"
        self.cache_dir = "D:\\"
        
    def download_data(self):
        snapshot_download(repo_id="benediktkol/DDOS", repo_type="dataset", cache_dir=self.cache_dir)
        
    def process(self):
        depth_file = "D:\\datasets--benediktkol--DDOS\\snapshots\\1ed1314d32ef3a5a7e1434000783a8433517bd0e\\data\\{}\\{}\\{}\\depth\\{}.png"
        img_file = "D:\\datasets--benediktkol--DDOS\\snapshots\\1ed1314d32ef3a5a7e1434000783a8433517bd0e\\data\\{}\\{}\\{}\\image\\{}.png"
        structure = {
            "train": {
                "neighbourhood": 250 ,
                "park": 50
            },
            "test": {
                "neighbourhood": 15,
                "park": 5
            },
            "validation": {
                "neighbourhood": 15,
                "park": 5
            },
        }
        with h5py.File(self.outfile, "w") as h5f:
            n_images = 3400
            images = h5f.create_dataset("images", (n_images, 480, 480, 3), dtype="uint8")
            depths = h5f.create_dataset("depths", (n_images, 480, 480), dtype="float32")
            
            idx = 0
            for split, folders in structure.items():
                for place, n_scans in folders.items():
                    for scan in tqdm(range(n_scans)):
                        for num in range(0,100,10):
                            raw_depth = cv2.imread(depth_file.format(split,place,scan,num),cv2.IMREAD_ANYDEPTH)
                            depth_in_meters = (raw_depth / 65535.0) * 100
                            img, depth = self.crop_and_resize(
                                cv2.cvtColor(cv2.imread(img_file.format(split,place,scan,num)), cv2.COLOR_BGR2RGB),
                                depth_in_meters
                            )
                            images[idx] = img
                            depths[idx] = 1.0 / (depth + 1e-6)
                            idx += 1
                
        print(f"Data saved to {self.outfile}.")

class TartanAirProcessor(DatasetProcessor):
    
    def __init__(self):
        super(TartanAirProcessor, self).__init__()
        self.outfile = "D:\\processed_tartan_air.h5"
        
    def download_data(self):
        raise NotImplementedError("Download and unzip sample data from https://theairlab.org/tartanair-dataset/ into D:\\TartanAir")
        
    def process(self):
        directories = [f.path for f in os.scandir("D:\\TartanAir") if f.is_dir()]
        with h5py.File(self.outfile, "w") as h5f:
            n_images = 3335
            images = h5f.create_dataset("images", (n_images, 480, 480, 3), dtype="uint8")
            depths = h5f.create_dataset("depths", (n_images, 480, 480), dtype="float32")
            
            left_camera = True
            idx = 0
            for dir in directories:
                depth_left = os.listdir(os.path.join(dir,"depth_left"))
                image_left = os.listdir(os.path.join(dir,"image_left"))
                depth_right = os.listdir(os.path.join(dir,"depth_right"))
                image_right = os.listdir(os.path.join(dir,"image_right"))
                for i in tqdm(range(0,len(image_left),2)):
                    if left_camera:
                        img_path = os.path.join(dir,"image_left",image_left[i])
                        depth_path = os.path.join(dir,"depth_left",depth_left[i])
                    else:
                        img_path = os.path.join(dir,"image_right",image_right[i])
                        depth_path = os.path.join(dir,"depth_right",depth_right[i])
                    left_camera = not left_camera
                    img, depth = self.crop_and_resize(
                        cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB),
                        np.load(depth_path)
                    )
                    images[idx] = img
                    depths[idx] = 1.0 / (depth + 1e-6)
                    idx += 1
                    
            print(idx)
                
        print(f"Data saved to {self.outfile}.")
