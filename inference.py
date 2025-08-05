import os
from model.openllama import OpenLLAMAPEFTModel
import torch
from torchvision import transforms
import numpy as np
import json,random
import cv2

# init the model
args = {
    'model': 'openllama_peft',
    'ckpt_path': '../checkpoint/ckpt.pth',
    'max_tgt_len': 128,
    'lora_r': 32,
    'lora_alpha': 32,
    'lora_dropout': 0.1,
}
model = OpenLLAMAPEFTModel(**args)
delta_ckpt = torch.load(args['ckpt_path'], map_location=torch.device('cpu'))
model.load_state_dict(delta_ckpt, strict=False)
model = model.cuda().eval().bfloat16()
torch.autograd.set_grad_enabled(False)


norm_transform = transforms.Compose(
                            [
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    mean=(0.48145466, 0.4578275, 0.40821073),
                                    std=(0.26862954, 0.26130258, 0.27577711),
                                ),
                            ]
                        )
print(f'[!] init the 7b model over ...')

"""Override Chatbot.postprocess"""

def set_random_seed(seed):
    if seed is not None and seed > 0:
        print('seed')
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_random_seed(42)

def predict(
    input, 
    image_path, 
    normal_img_path, 
    max_length, 
    top_p, 
    temperature,
    history,
    modality_cache,  
):
    prompt_text = ''
    for idx, (q, a) in enumerate(history):
        if idx == 0:
            prompt_text += f'{q}\n### Assistant: {a}\n###'
        else:
            prompt_text += f' Human: {q}\n### c: {a}\n###'
    if len(history) == 0:
        prompt_text += f'{input}'
    else:
        prompt_text += f' Human: {input}'
   
    response, pixel_output, prob = model.generate({
        'prompt': prompt_text,
        'image_paths': [image_path] if image_path else [],
        'audio_paths': [],
        'video_paths': [],
        'thermal_paths': [],
        'normal_img_paths': normal_img_path if normal_img_path else [],
        'top_p': top_p,
        'temperature': temperature,
        'max_tgt_len': max_length,
        'modality_embeds': modality_cache
    })

    return pixel_output, response, prob[0][1].item()

input = "Is this a deepfake image?"


CLASS_NAMES = ['person face']
caption='This is a facial image designed for deepfake detection, and it should not exhibit any localized color discrepancies or evident signs of splicing. '


index=0
for file_path in os.listdir('./input'):
    file_path= os.path.join('./input', file_path)
    index+=1
    with torch.no_grad():
        anomaly_map, response, prob = predict(caption+ input, file_path, [], 512, 0.1, 1, [], [])

    if 'pristine' in response:
        pred=0
    if 'not' in response and 'pristine' in response:
        pred=1 
    if 'deepfake' in response:
        pred=1
    if 'not' in response and 'deepfake' in response:
        pred=0
    if 'Yes' in response:
        pred=1
    if 'No' in response:
        pred=0
    print(f'[{index}] {file_path} | {response} | fakeprob: {prob} | LLM Judge: {pred}')
    anomaly_map = anomaly_map.reshape(224, 224).detach().cpu().numpy()
    anomaly_map = np.array(anomaly_map*255,dtype=np.uint8)
    cv2.imwrite(f'./output/{index}.png', anomaly_map)
        
