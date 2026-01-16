import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms as T
from models import mobilev2_Match_hash, res50_cls, res50_cls_hash, res50_spatial
from hash import CrossAttentionModel, max_min_hash, ImprovedTransformerDecoder_v2, ImprovedTransformerDecoder
import matplotlib.pyplot as plt
import numpy as np

dataset_name = "scutv1"

num_steps = 500
alpha = 1.0       
beta = 0.001      
x1_image_path = "/home/xtm/dataset/palm/tongji_16/session1/00013.jpg"
x2_image_path = "/home/xtm/py/guided-diffusion/guided-diffusion-main/datasets/leaf/0043_02.jpg"

xn_image_path = "/home/xtm/py/guided-diffusion/guided-diffusion-main/datasets/scutsd_550/0004_04.jpg"

def main():

    x1 = load_image(x1_image_path).cuda()
    x2 = load_image(x2_image_path).cuda()
    xn = load_image(xn_image_path).cuda()
    x2_adv = x2.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([x2_adv], lr=0.01)
            
    model = res50_cls_hash(256)
    #state_dict = torch.load("/home/xtm/py/palm_q/output/backbone_scut_034.pth")
    #model.load_state_dict(state_dict)
    model = model.cuda()

    ca = ImprovedTransformerDecoder_v2(60, 2048, 256, 8)
    #state_dict = torch.load("/home/xtm/py/palm_q/output/ca_scut_034.pth")
    #ca.load_state_dict(state_dict)
    ca = ca.cuda()
    model.eval()
    ca.eval()

    cosine_similarities = [] 
    cosine_similarities_ca = []
    plot_steps = np.arange(0, num_steps, 1)

    for step in range(num_steps):
        optimizer.zero_grad()
        emb1 = model(x1, False)         
        emb2 = model(x2_adv, False)
        
        cosine_sim = F.cosine_similarity(emb1, emb2).item()
        cosine_similarities.append(cosine_sim)
        
               
        loss_embedding = F.mse_loss(emb1, emb2)
        loss_perturb = F.mse_loss(x2_adv, x2)   
        loss = alpha * loss_embedding + beta * loss_perturb

        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            x2_adv.clamp_(0, 1)

        if step % 50 == 0:
            print(f"[Step {step}] Loss: {loss.item():.6f}")
        
    x2_adv = x2.clone().detach().requires_grad_(True)
    optimizer_v2 = optim.Adam([x2_adv], lr=0.01)
        
    for step in range(num_steps):
        optimizer.zero_grad()
        mid1 = model(x1, False)  
        emb1, _, _ = ca(mid1)          
        mid2 = model(x2_adv, False)
        emb2, _, _ = ca(mid2)  
        
        cosine_sim_ca = F.cosine_similarity(emb1, emb2).item()
        cosine_similarities_ca.append(cosine_sim_ca)
               
        loss_embedding = F.mse_loss(emb1, emb2)
        loss_perturb = F.mse_loss(x2_adv, x2)   
        loss = alpha * loss_embedding + beta * loss_perturb

        loss.backward()
        optimizer_v2.step()
        
        

   
        with torch.no_grad():
            x2_adv.clamp_(0, 1)

        if step % 50 == 0:
            print(f"[Step {step}] Loss: {loss.item():.6f}")
            
    with torch.no_grad():
        emb1 = model(x1, False)  
        #emb1, _, _ = ca(mid1)   
        emb2 = model(x2, False)  
        #emb2, _, _ = ca(mid2)        
        emb2_adv = model(x2_adv, False) 
        #emb2_adv, _, _ = ca(mid2_adv)    
        
        midn = model(xn, False)  
        embn, _, _ = ca(midn)   
        
        

        dist_before = F.mse_loss(emb1, emb2).item()
        dist_after = F.mse_loss(emb1, emb2_adv).item()
        l2_perturb = torch.norm((x2_adv - x2).view(-1)).item()
        cosine_sim = F.cosine_similarity(emb1, emb2_adv).item()
        #cosine_simn = F.cosine_similarity(emb1, embn).item()

    print("\n--- result---")
    print(f"dis before attack: {dist_before:.6f}")
    print(f"dis after attack: {dist_after:.6f}")
    print(f"Adversarial disturbance size (L2): {l2_perturb:.6f}")
    print(f"Cosine similarity after attack: {cosine_sim:.6f}")
    
    plot_cosine_similarity(plot_steps, cosine_similarities, cosine_similarities_ca)
    #print(f"Cosine similarity before attack: {cosine_simn:.6f}")


    
    
    
def load_image(image_path):
    transform_test = T.Compose([T.Resize([224, 224]), T.ToTensor(), T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])])
    image = Image.open(image_path).convert('RGB') 
    return transform_test(image).unsqueeze(0) 
   
def plot_cosine_similarity(steps, cosine_similarities, cosine_similarities_ca):
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, cosine_similarities, label='w/o Hybrid-key Set protection method', color='b')
    plt.plot(steps, cosine_similarities_ca, label='w/ proposed Hybrid-key Set protection method', color='r')
    #plt.title('Cosine Similarity vs Step')
    plt.xlabel('Steps', fontsize=14)
    plt.ylabel('Cosine Similarity', fontsize=14)
    plt.grid(True)
    plt.legend(loc='upper left')
    
    plt.tick_params(axis='x', labelsize=12)  
    plt.tick_params(axis='y', labelsize=12)  
    
    plt.savefig("/home/xtm/py/palm_q/duikang.png")
    plt.show() 
    
if __name__ == '__main__':
    main()
    