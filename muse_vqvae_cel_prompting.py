'''
--- Program to prompt a pre-trained Muse model to generate images.
'''

import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision 
import matplotlib.pyplot as plt 
import cv2 
import math 
from copy import deepcopy
from tqdm import tqdm 
import json 
import os 

# import T5 
from transformers import T5Tokenizer, T5ForConditionalGeneration
# import VQVAE for loading the pretrained weights
from vqvae_residual_moreFilters import VQVAE 
from muse_vqvae_residual_cel import Muse_vqvae


# utility function to load model weights from checkpoint - loads to the device passed as 'device' argument
def load_ckpt(checkpoint_path, model, optimizer=None, scheduler=None, device=torch.device('cpu'), mode='eval'):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    if mode == 'eval':
        model.eval() 
        return model
    else:
        model.train()
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if scheduler is not None:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            return model, optimizer, scheduler
        else:
            return model, optimizer

# utility function to save a checkpoint (model_state, optimizer_state, scheduler_state) - saves on cpu (to save gpu memory)
def save_ckpt(device, checkpoint_path, model, optimizer, scheduler=None):
    # transfer model to cpu
    model = model.to('cpu')
    # prepare dicts for saving
    save_dict = {'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}
    if scheduler is not None:
        save_dict['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(save_dict, checkpoint_path)
    # load model back on original device 
    model = model.to(device)

# convert tensor to img
def to_img(x):
    x = x.clamp(-1, 1) # clamp img to be strictly in [-1, 1]
    x = 0.5 * x + 0.5 # transform img from range [-1, 1] -> [0, 1]
    x = x.permute(0,2,3,1)
    return x

# function to save a generated img
def save_img_generated(x_g, save_path):
    gen_img = x_g.detach().cpu().numpy()
    gen_img = np.uint8( gen_img * 255 )
    # bgr to rgb 
    # concat_img = concat_img[:, :, ::-1]
    cv2.imwrite(save_path, gen_img)


### main ###
if __name__ == '__main__':

    # hyperparams for vqvae 
    num_embed = 8192 # codebook vocab size 
    embed_dim = 16 # embedding dimension of codebook vectors
    num_blocks = 2 # number of residual blocks in a residual stack
    img_size = 128 
    img_channels = 3
    img_shape = torch.tensor([img_channels, img_size, img_size])
    resize_shape = (img_size, img_size)
    img_latent_dim = embed_dim # as used in the pretrained VQVAE 
    img_latent_seqlen = 64 # as used in the pretrained VQVAE 

    # hyperparams for T5 
    d_model = 768 # d_model for T5 (required for image latents projection)
    max_seq_len = 512 # required to init T5 Tokenizer

    num_epochs = 100
    epochs_done = 0
    random_seed = 10

    vqvae_ckpt_path = './ckpts/VQVAE_residual_moreFilters_voc_val_batchSize256.pth' # path to pretrained vqvae 
    muse_ckpt_path = './ckpts/muse_vqvae_residual_cel.pth' # path to save the trained muse model (pretrained T5 with custom trained decoder)
    save_img_folder = './out_imgs_muse_vqvae_residual_cel_prompted'
    if not os.path.exists(save_img_folder):
        os.makedirs(save_img_folder)

    t5_model_name = 't5-base'
    
    # set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # cuda
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load pretrained VQVAE in eval mode 
    vqvae_model = VQVAE(device, num_blocks, num_embed, embed_dim).to(device)
    vqvae_model = load_ckpt(vqvae_ckpt_path, vqvae_model, device=device, mode='eval')

    # init T5 tokenizer and transformer model
    t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name, model_max_length=max_seq_len)
    t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name).to(device)

    # get start token embedding from T5 embedder (used for decoder input and generation)
    start_token_id = t5_tokenizer(t5_tokenizer.pad_token, return_tensors='pt', padding=False, truncation=True).input_ids
    start_token_id = start_token_id[:, 0].to(device) # trim end token appended by the tokenizer
    start_token_emb = t5_model.shared(start_token_id) # shape: [d_model]
    start_token_emb = start_token_emb.detach() # since we don't want to train the T5 embedder

    # init muse model 
    muse_model = Muse_vqvae(t5_model.decoder, img_latent_dim, img_latent_seqlen, num_embed, d_model, device).to(device)

    # load muse model from checkpoint in eval mode
    muse_model = load_ckpt(muse_ckpt_path, muse_model, device=device, mode='eval')

    # bookeeping results for plotting 
    results_loss, results_batch_accuracy = [], []

    # train loop
    for ep in tqdm(range(num_epochs)):
        ep += epochs_done 

        # input prompt 
        prompt = input('Enter prompt: ')

        # convert prompt to tokens 
        cap_tokens_dict = t5_tokenizer([prompt], return_tensors='pt', padding=True, truncation=True)
        cap_tokens_dict = cap_tokens_dict.to(device)

        with torch.no_grad():

            # get a generated img
            cap_tokens, cap_attn_mask = cap_tokens_dict.input_ids, cap_tokens_dict.attention_mask
            cap_tokens, cap_attn_mask = cap_tokens[0].unsqueeze(0), cap_attn_mask[0].unsqueeze(0)
            enc_out = t5_model.encoder(input_ids=cap_tokens, attention_mask=cap_attn_mask).last_hidden_state # enc_out.shape: [b=1, cap_seqlen, d_model]
            z_gen = muse_model.generate_greedy(enc_out, start_token_emb, vqvae_model) # out.shape: [b, img_latent_seqlen, img_latent_dim]
            z_gen = z_gen.flatten(start_dim=0, end_dim=1)
            gen_img = vqvae_model.decode(z_gen)
            x_g = to_img(gen_img.data)[0]
            save_img_generated(x_g, save_img_folder + '/{}_{}_generated.png'.format(ep, prompt))

