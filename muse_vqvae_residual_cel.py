'''
--- Program implementing Muse, but using pretrained VQVAE instead of VQGAN to obtain discrete image tokens / latents.

-- Features:
1. The encoder of a pretrained VQVAE is used to obtain discrete image tokens / latents.
2. The image latents are feed to the T5 decoder (by-passing the T5 decoder's embedding layer) for autoregressive modelling via teacher forcing.
3. Image caption tokens are fed to the T5 encoder to go through the T5 encoder's embedding layer to convert text tokens (obtained from T5 tokenizer) to embeddings. 
4. The T5 encoder's output is used to condition the T5 decoder via cross-attention. The T5 decoder predicts the output / target text in a teacher-forcing regime (thanks to causal masking in the T5 decoder self-attn layers).
5. Loss is MSEloss between the predicted and target image tokens / latents (NOTE that when training transformers, this is typically a cross-entropy loss between predicted and target tokens, but in this case the tokens are image latent vectors). Only the T5 decoder weights are trained. The T5 encoder and the VQVAE are frozen.

-- Todos / Questions:
1. Do we need to re-initialize the weights of T5 decoder (e.g. Xavier initialization) ?
2. Do we need to append start token embedding to the image latents before feeding them to T5 decoder ? [Guess: Yes, it will be necessary for generation]
3. Replacing causal masking in T5 decoder with arbitrary / random masking as originally done in Muse.
4. Do we need to add positional encoding to img latents before feeding them to T5 decoder ? [Guess: Yes, since a particular codebook vector has different meaning when its present at different positions in the image latent vector]
5. [Done in this implementation] Replace MSELoss with Cross-entropy loss where the targets are codebook vector indices instead of the vectors themselves.
6. Add ignore_index in cross-entropy loss to avoid incurring loss on pad tokens 
7. Provide encoder attention_mask in decoder input
8. Do we need to add the LLM positional embedding or T5 adds it internally
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


# class implementing Muse_vqvae (the T5 decoder with linear projection layers and positional encoding layers)
class Muse_vqvae(nn.Module):
    def __init__(self, t5_decoder, img_latent_dim, img_latent_seqlen, num_embed, d_model, device):
        super().__init__()
        self.decoder = t5_decoder 
        self.input_proj = nn.Linear(img_latent_dim, d_model)
        self.pos_emb = nn.Parameter(torch.rand(img_latent_seqlen+1, d_model)) # +1 for prepended start token 
        self.output_proj = nn.Linear(d_model, num_embed)
        self.img_latent_seqlen = img_latent_seqlen 
        self.img_latent_dim = img_latent_dim 
        self.d_model = d_model 
        self.num_embed = num_embed 
        self.device = device 

    def init_decoder(self): 
        # TODO: re-init T5 decoder params 
        pass 

    def forward(self, encoder_out, img_latents, start_emb): 
        '''
        encoder_out: output from the T5 encoder after feeding caption tokens (shape: [b, caption_seq_len, d_model])
        img_latents: codebook vectors obtained from the pretrained VQVAE, appended with start token and pad token embeddings (shape: [b, img_latent_seqlen, img_latent_dim])
        start_emb: embedding for the start token obtained from T5 embedder (shape: [d_model])
        pad_emb: embedding for the  token obtained from T5 embedder (shape: [d_model])
        '''
        batch_size = encoder_out.shape[0]
        # project img_latents to d_model
        z_emb = self.input_proj(img_latents) # shape: [b, img_latent_seqlen, d_model]
        # append start token embedding
        start_emb = start_emb.expand(batch_size, -1).unsqueeze(1) # shape: [b, 1, d_model] 
        z_emb = torch.cat([start_emb, z_emb], dim=1) # shape: [b, img_latent_seqlen + 1, d_model]
        # add positional embedding 
        pos_emb = self.pos_emb.unsqueeze(0).expand(batch_size, -1, -1) # shape: [b, img_latent_seqlen + 1, d_model]
        # TODO: check if we need to add positional embedding or T5 adds it internally
        z_emb += pos_emb 
        # feed to decoder 
        # NOTE that when we don't provide an attention_mask, T5 (correctly) assumes that there are no pad tokens in the input. So we don't need to provide a decoder attention_mask here as there are no pad tokens in the decoder input.
        # TODO but do we need to provide encoder attention_mask, as there are pad tokens in the encoder input?
        out = self.decoder(inputs_embeds=z_emb, encoder_hidden_states=encoder_out).last_hidden_state # out.shape: [b, img_latent_seqlen + 1, d_model]
        # trim last prediction since we have no target for that 
        out = out[:, :-1] # shape: [b, img_latent_seqlen, d_model]
        # project to num_embed to get score
        scores = self.output_proj(out) # shape: [b, img_latent_seqlen, num_embed]
        # get predicted codebook indices 
        pred_idx = torch.argmax(scores, dim=-1) # shape: [b, img_latent_seqlen]
        return scores, pred_idx 
    
    def generate_greedy(self, encoder_out, start_emb, vqvae_model):
        batch_size = encoder_out.shape[0]
        z_placeholder = torch.rand(batch_size, self.img_latent_seqlen, self.img_latent_dim).to(self.device)
        # prepare start embedding and pos_embedding
        start_emb = start_emb.expand(batch_size, -1).unsqueeze(1) # shape: [b=1, 1, d_model]
        pos_emb = self.pos_emb.unsqueeze(0).expand(batch_size, -1, -1) # shape: [b=1, img_latent_seqlen + 1, d_model]
        # start generation 
        idx = 0
        while idx < self.img_latent_seqlen:
            # project z_placholder from img_latent_dim to d_model
            z_emb = self.input_proj(z_placeholder) # shape: [b, img_latent_seqlen, d_model]
            # prepend start_emb to z_emb
            z_emb = torch.cat([start_emb, z_emb], dim=1) # shape: [b, img_latent_seqlen + 1, d_model]
            # add positional embedding
            z_emb += pos_emb 
            # forward prop through decoder
            # NOTE that we don't need to provide attention_mask (pad_mask) as T5 decoder internally applies the causal attention mask (This will change when using arbitrary masking as origianlly used in Muse)
            out = self.decoder(inputs_embeds=z_emb, encoder_hidden_states=encoder_out).last_hidden_state # out.shape: [b, img_latent_seqlen, d_model]
            z_out = out[:, idx] # out.shape: [b, d_model]
            # project to num_embed to get score 
            z_score = self.output_proj(z_out) # shape: [b, num_embed]
            z_pred_idx = torch.argmax(z_score, dim=-1) # shape: [b=1]
            z_pred = vqvae_model.codebook[z_pred_idx] # shape: [b=1, img_latent_dim]
            z_placeholder[:, idx] = z_pred 
            idx += 1
        return z_placeholder


# function to forward prop through the vqvae, T5_encoder and Muse models to calculate loss
def calculate_loss(vqvae_model, t5_model, muse_model, start_token_emb, imgs, cap_tokens_dict, device):
    batch_size = imgs.shape[0]
    with torch.no_grad():
        # obtain img embeddings
        z_e = vqvae_model.encode(imgs) # z_e.shape: [b * img_latent_seqlen,  img_latent_dim]
        img_latents, _, _, _, _, target_idx = vqvae_model.quantize(z_e) # img_latents.shape: [b * img_latent_seqlen, img_latent_dim]
        img_latents = img_latents.view(batch_size, -1, z_e.shape[-1]) # img_latents.shape: [b, img_latent_seqlen, img_latent_dim]

        # extract cap tokens and attn_mask from cap_tokens_dict
        cap_tokens, cap_attn_mask = cap_tokens_dict.input_ids, cap_tokens_dict.attention_mask
        # feed cap_tokens to t5 encoder to get encoder output
        enc_out = t5_model.encoder(input_ids=cap_tokens, attention_mask=cap_attn_mask).last_hidden_state # enc_out.shape: [batch_size, cap_seqlen, d_model]

    # forward prop through Muse model 
    scores, pred_idx = muse_model(enc_out, img_latents, start_token_emb) # scores.shape: [b, img_latent_seqlen, num_embed]

    # calculate loss  
    criterion = nn.CrossEntropyLoss(reduction='mean')
    scores_flat = scores.flatten(start_dim=0, end_dim=1) # shape: [b * img_latent_seqlen, num_embed]
    loss = criterion(scores_flat, target_idx)
    # calculate batch accuracy 
    batch_accuracy = (pred_idx.flatten() == target_idx).float().mean() * 100
    return loss, batch_accuracy, pred_idx  


# utility function to load img and captions data 
def load_data():
    imgs_folder = 'dataset_coco_val2017/images/'
    captions_file_path = 'dataset_coco_val2017/annotations/captions_val2017.json'
    captions_file = open(captions_file_path)
    captions = json.load(captions_file)
    img_dict, img_cap_pairs = {}, []
    print('Loading Data...')
    num_iters = len(captions['images']) + len(captions['annotations'])
    pbar = tqdm(total=num_iters)

    for img in captions['images']:
        id, file_name = img['id'], img['file_name']
        img_dict[id] = file_name
    for cap in captions['annotations']:
        id, caption = cap['image_id'], cap['caption']
        # use img_name as key for img_cap_dict
        img_filename = img_dict[id]

        # load image from img path 
        img_path = imgs_folder + img_filename
        img = cv2.imread(img_path, 1)
        resize_shape = (img_size, img_size)
        img = cv2.resize(img, resize_shape, interpolation=cv2.INTER_LINEAR)
        img = np.float32(img) / 255
        img = torch.tensor(img)
        img = img.permute(2, 0, 1) # [w,h,c] -> [c,h,w]
        transforms = torchvision.transforms.Compose([
            # NOTE: no random resizing as its asking the model to learn to predict different img tokens for the same caption
            # torchvision.transforms.Resize( int(1.25*img_size) ),  # image_size + 1/4 * image_size
            # torchvision.transforms.RandomResizedCrop(resize_shape, scale=(0.8, 1.0)),
            # torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # imagenet stats for normalization
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # zero mean, unit std
        ])
        img = transforms(img)

        # append prefix text to caption : "An example of"
        # caption = 'An example of ' + caption + ': '
        img_cap_pairs.append([img, caption])
        pbar.update(1)
    pbar.close()
    return img_cap_pairs


# utility function to process minibatch - convert img_filenames to augmented_imgs and convert caption_text to tokenized_captions - then obtain embeddings for them
def process_batch(minibatch, tokenizer, img_size, device):
    # augmented_imgs = []
    augmented_imgs, captions = list(map(list, zip(*minibatch)))
    # tokenize captions 
    caption_tokens_dict = tokenizer(captions, return_tensors='pt', padding=True, truncation=True)

    # # get augmented imgs
    # imgs_folder = 'dataset_coco_val2017/images/'
    # for img_filename in img_files:
    #     img_path = imgs_folder + img_filename
    #     img = cv2.imread(img_path, 1)
    #     resize_shape = (img_size, img_size)
    #     img = cv2.resize(img, resize_shape, interpolation=cv2.INTER_LINEAR)
    #     img = np.float32(img) / 255
    #     img = torch.tensor(img)
    #     img = img.permute(2, 0, 1) # [w,h,c] -> [c,h,w]
    #     transforms = torchvision.transforms.Compose([
    #         # NOTE: no random resizing as its asking the model to learn to predict different img tokens for the same caption
    #         # torchvision.transforms.Resize( int(1.25*img_size) ),  # image_size + 1/4 * image_size
    #         # torchvision.transforms.RandomResizedCrop(resize_shape, scale=(0.8, 1.0)),
    #         # torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # imagenet stats for normalization
    #         torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # zero mean, unit std
    #     ])
    #     img = transforms(img)
    #     augmented_imgs.append(img)

    augmented_imgs = torch.stack(augmented_imgs, dim=0).to(device)
    caption_tokens_dict = caption_tokens_dict.to(device)
    return augmented_imgs, caption_tokens_dict


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

# utility function to freeze model
def freeze(model):
    for p in model.params:
        p.requires_grad = False 
    return model 

# convert tensor to img
def to_img(x):
    x = x.clamp(-1, 1) # clamp img to be strictly in [-1, 1]
    x = 0.5 * x + 0.5 # transform img from range [-1, 1] -> [0, 1]
    x = x.permute(0,2,3,1)
    return x

# function to save a test img and its reconstructed img 
def save_img_reconstructed(x, x_r, save_path):
    concat_img = torch.cat([x, x_r], dim=1)
    concat_img = concat_img.detach().cpu().numpy()
    concat_img = np.uint8( concat_img * 255 )
    # bgr to rgb 
    # concat_img = concat_img[:, :, ::-1]
    cv2.imwrite(save_path, concat_img)

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

    num_epochs = 2500 * 7 * 11
    epochs_done = 2500
    batch_size = 32
    lr = 3e-4
    random_seed = 1010
    resume_training_from_ckpt = True   

    vqvae_ckpt_path = './ckpts/VQVAE_residual_moreFilters_voc_val_batchSize256.pth' # path to pretrained vqvae 
    muse_ckpt_path = './ckpts/muse_vqvae_residual_cel.pth' # path to save the trained muse model (pretrained T5 with custom trained decoder)
    save_img_folder = './out_imgs_muse_vqvae_residual_cel'
    if not os.path.exists(save_img_folder):
        os.makedirs(save_img_folder)
       

    t5_model_name = 't5-base'
    
    # set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # cuda
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # create dataset from img_cap_dict
    dataset = load_data()
    dataset_len = len(dataset)

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

    # init optimizer 
    # NOTE that we don't freeze the T5 model except for the decoder, instead we just register the T5 decoder params with the optimezer for training / gradient update (and not register any other T5 params)
    optimizer = torch.optim.AdamW(params=muse_model.parameters(), lr=lr)

    # load checkpoint to resume training from
    if resume_training_from_ckpt:
        muse_model, optimizer = load_ckpt(muse_ckpt_path, muse_model, optimizer=optimizer, device=device, mode='train')

    # bookeeping results for plotting 
    results_loss, results_batch_accuracy = [], []

    # train loop
    for ep in tqdm(range(num_epochs)):
        ep += epochs_done 

        # fetch minibatch
        idx = np.arange(dataset_len)
        np.random.shuffle(idx)
        idx = idx[:batch_size]
        minibatch = [dataset[i] for i in idx]

        # process minibatch - convert img_filenames to augmented_imgs and convert caption_text to tokenized_captions
        # note that we don't create embeddings yet, since that's done by the image_encoder and T5 model
        imgs, cap_tokens_dict = process_batch(minibatch, t5_tokenizer, img_size, device) # imgs.shape:[batch_size, 3, 32, 32], captions.shape:[batch_size, max_seq_len]

        # calculate loss
        loss, batch_accuracy, pred_idx = calculate_loss(vqvae_model, t5_model, muse_model, start_token_emb, imgs, cap_tokens_dict, device)

        # update params
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        results_loss.append(loss.item())
        results_batch_accuracy.append(batch_accuracy.item())


        if (ep+1) % 500 == 0:
            # print metrics
            print('ep:{} \t loss:{:.3f} \t batch_accuracy:{:.3f}'.format(ep, loss.item(), batch_accuracy.item()))

            with torch.no_grad():
            
                # get reconstructed img from predicted img latents
                pred_idx = pred_idx.flatten() # [b, img_latent_seqlen] -> [b * img_latent_seqlen]
                pred_img_latents = vqvae_model.codebook[pred_idx] # [b * img_latent_seqlen, img_latent_dim]
                recon_imgs = vqvae_model.decode(pred_img_latents)
                x_r = to_img(recon_imgs.data)[0]
                x = to_img(imgs.data)[0]
                x_caption = minibatch[0][1]
                save_img_reconstructed(x, x_r, save_img_folder + '/{}_{}_reconstructed.png'.format(ep, x_caption))

                # get a generated img
                cap_tokens, cap_attn_mask = cap_tokens_dict.input_ids, cap_tokens_dict.attention_mask
                cap_tokens, cap_attn_mask = cap_tokens[0].unsqueeze(0), cap_attn_mask[0].unsqueeze(0)
                enc_out = t5_model.encoder(input_ids=cap_tokens, attention_mask=cap_attn_mask).last_hidden_state # enc_out.shape: [b=1, cap_seqlen, d_model]
                z_gen = muse_model.generate_greedy(enc_out, start_token_emb, vqvae_model) # out.shape: [b, img_latent_seqlen, img_latent_dim]
                z_gen = z_gen.flatten(start_dim=0, end_dim=1)
                gen_img = vqvae_model.decode(z_gen)
                x_g = to_img(gen_img.data)[0]
                save_img_generated(x_g, save_img_folder + '/{}_{}_generated.png'.format(ep, x_caption))


        if (ep+1) % 2500 == 0:
            # save model checkpoint 
            save_ckpt(device, muse_ckpt_path, muse_model, optimizer)

            # plot results
            fig, ax = plt.subplots(1,2, figsize=(15,5))

            ax[0].plot(results_loss, label='train_loss')
            ax[0].legend()
            ax[0].set(xlabel='iters')
            ax[1].plot(results_batch_accuracy, label='batch_accuracy')
            ax[1].legend()
            ax[1].set(xlabel='iters')

            plt.suptitle('final_loss: ' + str(results_loss[-1]))
            plt.savefig(save_img_folder + '/plot_' + str(ep) + '.png')
