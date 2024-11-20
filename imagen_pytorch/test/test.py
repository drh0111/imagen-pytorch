import os

os.environ['TRANSFORMERS_OFFLINE'] = '1'

from imagen_pytorch.trainer import ImagenTrainer
from imagen_pytorch.configs import ImagenConfig
from imagen_pytorch import Unet1d, Imagen
import torch.nn.functional as F
import time
from tqdm import tqdm
from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR,StepLR,ReduceLROnPlateau
from sklearn.metrics import cohen_kappa_score,f1_score
from sklearn.model_selection import KFold, train_test_split

device = torch.device("cuda")

class ION_Dataset(Dataset):
    def __init__(self, train_input, train_output,mode='train'):
        self.train_input = train_input
        self.train_output = train_output
        self.mode = mode
        
    def __len__(self):
        return len(self.train_input)
    
    def _augmentations(self,input_data, target_data):
        #flip
        if np.random.rand()<0.5:    
            input_data = input_data[::-1]
            target_data = target_data[::-1]
        return input_data, target_data
    
    def __getitem__(self, idx):
        x = self.train_input[idx]
        y = self.train_output[idx]
        if self.mode =='train':
            x,y = self._augmentations(x,y)
        out_x = torch.tensor(np.transpose(x.copy(),(1,0)), dtype=torch.float) # This makes [b, c, l] format
        out_y = torch.tensor(np.transpose(y.copy(),(1,0)), dtype=torch.float)
        return out_x, out_y
    
### DEFINE DATALOADER ###
df_train = pd.read_csv("./Data/liverpool-ion-switching/train.csv")
df_test = pd.read_csv("./Data/liverpool-ion-switching/test.csv")

# I don't use "time" feature
train_input = df_train["signal"].values.reshape(-1,4000,1)#number_of_data:1250 x time_step:4000, this shape is not in [b, c, l] format yet
train_input_mean = train_input.mean()
train_input_sigma = train_input.std()
train_input = (train_input-train_input_mean)/train_input_sigma
test_input = df_test["signal"].values.reshape(-1,10000,1)
test_input = (test_input-train_input_mean)/train_input_sigma

train_target = pd.get_dummies(df_train["open_channels"]).values.reshape(-1,4000,11)#classification

idx = np.arange(train_input.shape[0])
train_idx, val_idx = train_test_split(idx, random_state = 111,test_size = 0.2)

val_input = train_input[val_idx]
train_input = train_input[train_idx] 
val_target = train_target[val_idx]
train_target = train_target[train_idx] 

print("train_input:{}, val_input:{}, train_target:{}, val_target:{}".format(train_input.shape, val_input.shape, train_target.shape, val_target.shape))
    
unet1 = Unet1d(
        dim=128,
        text_embed_dim = None,
        num_resnet_blocks = 1,
        cond_dim = None,
        num_image_tokens = None,
        num_time_tokens = None,
        learned_sinu_pos_emb_dim = 0,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        cond_images_channels = 0,
        channels = 1,
        channels_out = 11,
        attn_dim_head = 0,
        attn_heads = 2, # But should not be used at all
        ff_mult = 2.,
        lowres_cond = False,                # for cascading diffusion - https://cascaded-diffusion.github.io/
        layer_attns = False,
        layer_attns_depth = 0,
        layer_mid_attns_depth = 0,
        layer_attns_add_text_cond = False,   # whether to condition the self-attention blocks with the text embeddings, as described in Appendix D.3.1
        attend_at_middle = False,            # whether to have a layer of attention at the bottleneck (can turn off for higher resolution in cascading DDPM, before bringing in efficient attention)
        layer_cross_attns = False,
        use_linear_attn = False,
        use_linear_cross_attn = False,
        cond_on_text = False,
        max_text_len = 0,
        init_dim = None,
        init_conv_kernel_size = 7,          # kernel size of initial conv, if not using cross embed
        init_cross_embed = True,
        init_cross_embed_kernel_sizes = (3, 7, 15),
        cross_embed_downsample = False,
        cross_embed_downsample_kernel_sizes = (2, 4),
        attn_pool_text = False,
        attn_pool_num_latents = 0,
        dropout = 0.,
        memory_efficient = False,
        init_conv_to_final_conv_residual = False,
        use_global_context_attn = False,
        scale_skip_connection = True,
        final_resnet_block = True,
        final_conv_kernel_size = 3,
        self_cond = False,
        resize_mode = 'nearest',
        combine_upsample_fmaps = False,      # combine feature maps from all upsample blocks, used in unet squared successfully
        pixel_shuffle_upsample = False,       # may address checkboard artifacts, have problem with 1D case
    )

batch_size = 8
train = ION_Dataset(train_input, train_target,mode='train')
valid = ION_Dataset(val_input, val_target,mode='valid')

x_test = torch.tensor(np.transpose(test_input,(0,2,1)), dtype=torch.float).cuda()
test = torch.utils.data.TensorDataset(x_test)

train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)


#### TRAINING ####
## Hyperparameter
n_epochs = 100
lr = 0.001

## Build tensor data for torch
train_preds = np.zeros((int(train_input.shape[0]*train_input.shape[1])))
val_preds = np.zeros((int(val_input.shape[0]*val_input.shape[1])))

train_targets = np.zeros((int(train_input.shape[0]*train_input.shape[1])))

avg_losses_f = []
avg_val_losses_f = []

##Loss function
loss_fn = torch.nn.BCEWithLogitsLoss()

#Build model, initial weight and optimizer
model = unet1 
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = lr,weight_decay=1e-5) # Using Adam optimizer
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.8, min_lr=1e-8) # Using ReduceLROnPlateau schedule
temp_val_loss = 9999999999


for epoch in range(n_epochs):
    
    start_time = time.time()
    model.train()
    avg_loss = 0.
    for i, (x_batch, y_batch) in enumerate(train_loader):
        y_pred = model(x_batch.cuda(), time=0) # Time here is a dummy variable
        
        loss = loss_fn(y_pred.cpu(), y_batch)
        
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        avg_loss += loss.item()/len(train_loader)

        pred = F.softmax(y_pred, 1).detach().cpu().numpy().argmax(axis=1)
        train_preds[i * batch_size*train_input.shape[1]:(i+1) * batch_size*train_input.shape[1]] = pred.reshape((-1))
        train_targets[i * batch_size*train_input.shape[1]:(i+1) * batch_size*train_input.shape[1]] = y_batch.detach().cpu().argmax(axis=1).reshape((-1))
        del y_pred, loss, x_batch, y_batch, pred
        
        
    model.eval()

    avg_val_loss = 0.
    for i, (x_batch, y_batch) in enumerate(valid_loader):
        y_pred = model(x_batch.cuda()).detach()

        avg_val_loss += loss_fn(y_pred.cpu(), y_batch).item() / len(valid_loader)
        pred = F.softmax(y_pred, 1).detach().cpu().numpy().argmax(axis=1)
        val_preds[i * batch_size*val_input.shape[1]:(i+1) * batch_size*val_input.shape[1]] = pred.reshape((-1))
        del y_pred, x_batch, y_batch, pred
        
    if avg_val_loss<temp_val_loss:
        #print ('checkpoint_save')
        temp_val_loss = avg_val_loss
        torch.save(model.state_dict(), './Model/liverpool-ion-switching/ION_train_checkpoint.pt')
        
    train_score = f1_score(train_targets,train_preds,average = 'macro')
    val_score = f1_score(val_target.argmax(axis=2).reshape((-1)),val_preds,average = 'macro')
    
    elapsed_time = time.time() - start_time 
    scheduler.step(avg_val_loss)
    
    print('Epoch {}/{} \t loss={:.4f} \t train_f1={:.4f} \t val_loss={:.4f} \t val_f1={:.4f} \t time={:.2f}s'.format(
        epoch + 1, n_epochs, avg_loss,train_score, avg_val_loss,val_score, elapsed_time))