import torch
import torch.nn.functional as F
import cv2
from torch.utils.data import DataLoader
from data_loading import H5Dataset, create_random_dataloaders, EqualSampler, create_equal_dataloaders
import matplotlib.pyplot as plt
from tqdm import tqdm
from training import train_model
from depth_net import DepthNet
from losses import SSIMSELoss
from metrics import compute_metrics


im = cv2.imread("weights/lab_02.png")
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
im_tensor = torch.tensor(im)

depth = cv2.imread("weights/lab_02_depth.png",cv2.IMREAD_ANYDEPTH)
depth = torch.tensor(depth)*65535.0/50.0
depth_tensor = (depth - depth.min()) / (depth.max() - depth.min())

model = DepthNet()
model.load_state_dict(torch.load("weights/best_depth_model.pth",weights_only=True,map_location='cpu'))
model.eval()

# print(compute_metrics(model))

output = model.infer_image(im_tensor)
output = (output - output.min()) / (output.max() - output.min())

fig, (ax1,ax2,ax3) = plt.subplots(1,3,)
ax1.imshow(im)
ax1.axis('off')
ax2.imshow(1/depth_tensor,'Spectral_r')
ax2.axis('off')
ax3.imshow(output,'Spectral')
ax3.axis('off')
plt.tight_layout()
plt.savefig("model_output.png",bbox_inches='tight',dpi=600)
plt.show()

# nyu = H5Dataset("D:\\processed_nyu_depth_v2.h5")

# image,depth = nyu[500]
# output = model(image.unsqueeze(0))

# fig, (ax1,ax2,ax3,ax4) = plt.subplots(1,4)
# ax1.imshow(image.permute(1,2,0))
# ax1.axis('off')
# ax2.imshow(depth)
# ax2.axis('off')
# ax4.imshow(output.squeeze(0).detach())
# ax4.axis('off')
# ax3.imshow(1/output.squeeze(0).detach())
# ax3.axis('off')
# plt.show()
# train,test,val = create_equal_dataloaders([urbanSyn, nyu, ddos, carla, tartanAir])

# for features,labels in tqdm(train):
#     img1 = features[0].permute(1,2,0)
#     label1 = labels[0][0]
#     img2 = features[-1].permute(1,2,0)
#     label2 = labels[-1][0]

#     fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
#     ax1.imshow(img1)
#     ax1.axis('off')
#     ax1.set_title(f"{features.shape}")
#     ax2.imshow(label1)
#     ax2.axis('off')
#     ax2.set_title(f"{labels.shape}")
#     ax3.imshow(img2)
#     ax3.axis('off')
#     ax4.imshow(label2)
#     ax4.axis('off')

#     plt.show()
#     break

# for features,labels in tqdm(test):
#     img1 = features[0]
#     label1 = labels[0]
#     img2 = features[-1]
#     label2 = labels[-1]

#     # fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
#     # ax1.imshow(img1)
#     # ax1.axis('off')
#     # ax2.imshow(label1)
#     # ax2.axis('off')
#     # ax3.imshow(img2)
#     # ax3.axis('off')
#     # ax4.imshow(label2)
#     # ax4.axis('off')

#     # plt.show()
#     # break

# for features,labels in tqdm(val):
#     img1 = features[0]
#     label1 = labels[0]
#     img2 = features[-1]
#     label2 = labels[-1]

#     # fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
#     # ax1.imshow(img1)
#     # ax1.axis('off')
#     # ax2.imshow(label1)
#     # ax2.axis('off')
#     # ax3.imshow(img2)
#     # ax3.axis('off')
#     # ax4.imshow(label2)
#     # ax4.axis('off')

#     # plt.show()
#     # break