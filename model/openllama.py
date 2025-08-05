import torch.nn.functional as F
from .ImageBind import *
from .ImageBind import data
from transformers import LlamaForCausalLM, AutoTokenizer
from .AnomalyGPT_models import *
from transformers import StoppingCriteria, StoppingCriteriaList,LlamaTokenizer
import kornia as K
from peft import LoraConfig,TaskType,get_peft_model
import torch
from torch.nn.utils import rnn
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoModel

CLASS_NAMES=['face','object']
normal_sentences_all = []
abnormal_sentences_all = []
normal_sentences_all.append(data.load_and_transform_text('real', torch.cuda.current_device()))
abnormal_sentences_all.append(data.load_and_transform_text('fake', torch.cuda.current_device()))

def encode_text_with_prompt_ensemble(model, obj, device, ctx):
    global normal_sentences_all,abnormal_sentences_all
    normal_sentences = torch.cat(normal_sentences_all).to(device)
    abnormal_sentences = torch.cat(abnormal_sentences_all).to(device)
    with torch.no_grad():
        class_embeddings_normal = model({ModalityType.TEXT: normal_sentences})[ModalityType.TEXT][0]
        class_embeddings_abnormal = model({ModalityType.TEXT: abnormal_sentences})[ModalityType.TEXT][0]

    class_embeddings_normal=torch.cat((ctx.cuda(),class_embeddings_normal),dim=0)
    class_embeddings_abnormal=torch.cat((ctx.cuda(),class_embeddings_abnormal),dim=0)

    class_embeddings_normal = class_embeddings_normal.unsqueeze(0)
    class_embeddings_normal = class_embeddings_normal.mean(dim=1, keepdim=True)
    class_embeddings_normal = class_embeddings_normal / class_embeddings_normal.norm(dim=-1, keepdim=True)

    class_embeddings_abnormal = class_embeddings_abnormal.unsqueeze(0)
    class_embeddings_abnormal = class_embeddings_abnormal.mean(dim=1, keepdim=True)
    class_embeddings_abnormal = class_embeddings_abnormal / class_embeddings_abnormal.norm(dim=-1, keepdim=True)
    
    text_features = torch.cat([class_embeddings_normal, class_embeddings_abnormal], dim=1)

    return text_features


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops = [], encounters=1):
        super().__init__()
        self.stops = stops
        self.ENCOUNTERS = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        stop_count = 0
        for stop in self.stops:
            stop_count = (stop == input_ids[0]).sum().item()
        if stop_count >= self.ENCOUNTERS:
            return True
        return False

def build_one_instance(tokenizer, conversation):
    text_list = []
    turn_num = len(conversation)
    input_ids, target_ids = [], []
    for i in range(turn_num):
        turn = conversation[i]
        role = turn['from']
        if i == 0: # the first human turn
            assert role == 'human'
            text = turn['value'] + '\n### Assistant:'
            one_input_id = tokenizer(text, add_special_tokens=False).input_ids
            input_ids += one_input_id
            target_ids += [-100]*len(one_input_id) # do not perform loss regression on human prompt
        else:
            if role == 'human':
                text = 'Human: ' + turn['value'] + '\n### Assistant:'
                one_input_id = tokenizer(text, add_special_tokens=False).input_ids
                input_ids += one_input_id
                target_ids += [-100]*len(one_input_id)
            elif role == 'gpt':
                text = turn['value'] + '\n###'
                one_input_id = tokenizer(text, add_special_tokens=False).input_ids
                input_ids += one_input_id
                target_ids += one_input_id
            else:
                raise Exception('Wrong Role!!!')
        text_list.append(text)
        assert len(input_ids) == len(target_ids)
    return text_list, input_ids, target_ids

def process_batch_instance(tokenizer, batch_of_conversations, max_tgt_len):
    batch_input_ids, batch_target_ids = [], []
    for conversation in batch_of_conversations:
        _, one_input_ids, one_target_ids = build_one_instance(tokenizer, conversation)
        batch_input_ids.append(torch.LongTensor(one_input_ids))
        batch_target_ids.append(torch.LongTensor(one_target_ids))

    input_ids = rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    target_ids = rnn.pad_sequence(batch_target_ids, batch_first=True, padding_value=-100)
    assert input_ids.size() == target_ids.size()
    input_ids = input_ids[:,:max_tgt_len]
    target_ids = target_ids[:,:max_tgt_len]
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    assert attention_mask.size() == input_ids.size()
    return input_ids, target_ids, attention_mask.long()



PROMPT_START = '### Human: <Img>'
class OpenLLAMAPEFTModel(nn.Module):

    '''LoRA for LLaMa model'''

    def __init__(self,imagesize=224, **args):
        super(OpenLLAMAPEFTModel, self).__init__()
        self.args = args
        max_tgt_len = args['max_tgt_len']
        self.visual_encoder, self.visual_hidden_size = imagebind_model.imagebind_huge(args)
        self.visual_encoder=self.visual_encoder.bfloat16()
        self.iter = 0
        self.locator=ConvLayer(2,128,3).bfloat16()
        self.classifier = Classifier(2,3).bfloat16()
        self.image_decoder = LinearLayer(1280, 1024, 3).bfloat16()
        self.prompt_learner = ForgeryPromptLearner(6, 4096).bfloat16()

        ctx_vectors = torch.empty(16,1024)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors).bfloat16()
        self.imagesize=imagesize

        # add the lora module
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=True, 
            r=self.args['lora_r'], 
            lora_alpha=self.args['lora_alpha'], 
            lora_dropout=self.args['lora_dropout'],
            target_modules=['q_proj','k_proj','v_proj','o_proj']
        )
        self.llama_tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5-16k", use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.llama_tokenizer.padding_side = "right"
        
        self.llama_model = LlamaForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5-16k")
        self.llama_model = get_peft_model(self.llama_model, peft_config).bfloat16()

        print ('Language decoder initialized.')

        self.llama_proj = nn.Linear(
            self.visual_hidden_size, 4096
        ).bfloat16()

        self.max_tgt_len = max_tgt_len
        self.device = torch.cuda.current_device()


    def encode_image(self, image_paths):
        inputs = {ModalityType.VISION: data.load_and_transform_vision_data(image_paths, self.device)}
        # convert into visual dtype
        inputs = {key: inputs[key].bfloat16() for key in inputs}
        # with torch.no_grad():
        embeddings = self.visual_encoder(inputs)
        image_embeds = embeddings['vision'][0] # bsz x 1024
        patch_features = embeddings['vision'][1] # bsz x h*w x 1280
        patch_tokens = self.image_decoder(patch_features) # bsz x h*w x 1024

        inputs_llama = self.llama_proj(image_embeds).unsqueeze(1) # bsz x 1 x llama_size
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(self.device) # bsz x 1
        return inputs_llama, atts_llama, patch_tokens
    
    def encode_image_for_web_demo(self, image_paths,img_tensor=None):
        if img_tensor is not None:
            inputs = {ModalityType.VISION: img_tensor.to(self.device)}
        else:
            inputs = {ModalityType.VISION: data.load_and_transform_vision_data_for_web_demo(image_paths, self.device)}
        # convert into visual dtype
        inputs = {key: inputs[key].bfloat16() for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            image_embeds = embeddings['vision'][0] # bsz x 1024
            patch_features = embeddings['vision'][1] # bsz x h*w x 1280
        patch_tokens = self.image_decoder(patch_features) # bsz x h*w x 1024

        inputs_llama = self.llama_proj(image_embeds).unsqueeze(1) # bsz x 1 x llama_size
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(self.device) # bsz x 1
        return inputs_llama, atts_llama, patch_tokens

    
    def encode_image_from_tensor(self, image_tensors):
        if not isinstance(image_tensors, list):
            image_tensors = [image_tensors]
        inputs = {ModalityType.VISION: torch.stack(image_tensors, dim=0).to(self.device)}
        # convert into visual dtype
        inputs = {key: inputs[key].bfloat16() for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            image_embeds = embeddings['vision'][0] # bsz x 1024
            patch_features = embeddings['vision'][1] # bsz x h*w x 1024
            patch_tokens = self.image_decoder(patch_features)

        if self.llama_proj.weight.dtype==torch.float32:
            image_embeds=image_embeds.float()
            inputs_llama = self.llama_proj(image_embeds).unsqueeze(1).bfloat16() # bsz x 1 x llama_size
        else:
            inputs_llama = self.llama_proj(image_embeds).unsqueeze(1).bfloat16() # bsz x 1 x llama_size
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(self.device) # bsz x 1
        return inputs_llama, atts_llama, patch_tokens
    

    def prompt_wrap(self, img_embeds, input_ids, target_ids, attention_mask, anomaly_embedding = None, prediction=None):
        '''
            input_ids, target_ids, attention_mask: bsz x s2
        '''
        input_ids = input_ids.to(self.device) # bsz x s2
        target_ids = target_ids.to(self.device) # bsz x s2
        attention_mask = attention_mask.to(self.device) # bsz x s2

        batch_size = img_embeds.shape[0]
        p_before = PROMPT_START
        p_before_tokens = self.llama_tokenizer(p_before, 
            return_tensors="pt", add_special_tokens=False).to(self.device)
        # peft model need deeper call
        p_before_embeds = self.llama_model.model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1) # bsz x s1 x embed_dim
        if prediction.shape[0]==2:
            p_middle1 = '</Img> According to KFD prediction, the forgery score is '+str(int(prediction[0,1].item()*100))[:5]+'%. '
            p_middle2 = '</Img> According to KFD prediction, the forgery score is '+str(int(prediction[1,1].item()*100))[:5]+'%. '
            p_middle_tokens1 = self.llama_tokenizer(p_middle1, 
                return_tensors="pt", add_special_tokens=False).to(self.device)
            # peft model need deeper call
            p_middle_embeds1 = self.llama_model.model.model.embed_tokens(p_middle_tokens1.input_ids) # bsz x s1 x embed_dim
            p_middle_tokens2 = self.llama_tokenizer(p_middle2, 
                return_tensors="pt", add_special_tokens=False).to(self.device)
            # peft model need deeper call
            p_middle_embeds2 = self.llama_model.model.model.embed_tokens(p_middle_tokens2.input_ids) # bsz x s1 x embed_dim
            min_len=min(p_middle_embeds1.shape[1],p_middle_embeds2.shape[1])
            p_middle_embeds=torch.cat((p_middle_embeds1[:,:min_len,:],p_middle_embeds2[:,:min_len,:]),dim=0)
        if prediction.shape[0]==1:
            p_middle1 = '</Img> According to deepfake prediction, the forgery score is '+str(int(prediction[0,1].item()*100))[:5]+'%. '
            p_middle_tokens1 = self.llama_tokenizer(p_middle1, 
                return_tensors="pt", add_special_tokens=False).to(self.device)
            # peft model need deeper call
            p_middle_embeds = self.llama_model.model.model.embed_tokens(p_middle_tokens1.input_ids) # bsz x s1 x embed_dim

        p_after_embeds = self.llama_model.model.model.embed_tokens(input_ids).expand(batch_size, -1, -1) # bsz x s2 x embed_dim
        bos = torch.ones([batch_size, 1],
                         dtype=p_before_tokens.input_ids.dtype,
                         device=p_before_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id # bsz x 1
        bos_embeds = self.llama_model.model.model.embed_tokens(bos) # bsz x 1 x embed_dim

        inputs_embeds = torch.cat([bos_embeds, p_before_embeds, img_embeds, p_middle_embeds, anomaly_embedding, p_after_embeds], dim=1) # bsz x (1+s1+1+s2) x embed_dim
        if prediction != None:
            prediction=prediction.unsqueeze(2).repeat(1,1,4096)
            inputs_embeds = torch.cat([bos_embeds, p_before_embeds, img_embeds, p_middle_embeds, anomaly_embedding, prediction, p_after_embeds], dim=1)
        # create targets
        empty_targets = (
            torch.ones([batch_size, 3+p_before_embeds.size()[1]+1+p_middle_embeds.size()[1] + anomaly_embedding.size()[1]], dtype=torch.long).to(self.device).fill_(-100)  
        ) # bsz x (1 + s1 + 1)
        targets = torch.cat([empty_targets, target_ids], dim=1) # bsz x (1 + s1 + 1 + s2)
        assert inputs_embeds.size()[1] == targets.size()[1]

        atts_prefix = torch.ones([batch_size, 3+p_before_embeds.size()[1]+1+p_middle_embeds.size()[1] + anomaly_embedding.size()[1]], dtype=torch.long).to(self.device) # bsz x (1 + s1 +1)
        attention_mask = torch.cat([atts_prefix, attention_mask], dim=1)
        assert attention_mask.size() == targets.size() # bsz x (1 + s1 + 1 + s2)
        return inputs_embeds, targets, attention_mask 



    def forward(self, inputs):
        image_paths = inputs['images']
        img_embeds, _, patch_tokens = self.encode_image_from_tensor(image_paths)
        class_name = inputs['class_names']
        feats_text_tensor = encode_text_with_prompt_ensemble(self.visual_encoder, class_name, self.device,self.ctx)

        anomaly_maps = []
        for layer in range(len(patch_tokens)):
            patch_tokens[layer] = patch_tokens[layer] / patch_tokens[layer].norm(dim=-1, keepdim=True)
            anomaly_map = (100.0 * patch_tokens[layer].bfloat16() @ feats_text_tensor.transpose(-2,-1))
            B, L, C = anomaly_map.shape
            H = int(np.sqrt(L))
            anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H).float(),
                                        size=self.imagesize, mode='bilinear', align_corners=True)
            anomaly_maps.append(anomaly_map)  
        anomaly_map_all = self.locator(anomaly_maps)
        prediction=self.classifier(anomaly_maps)
        label=torch.stack(inputs['labels'], dim=0).to(self.device)
        gt = inputs['masks']
        gt = torch.stack(gt, dim=0).to(self.device)
        f_loss = self.loss_focal(anomaly_map_all, gt)
        d_loss = self.loss_dice(anomaly_map_all[:,1:,:,:], gt)
        loss_pixel =  d_loss+f_loss
        anomaly_map_all = anomaly_map_all[:,1:,:,:]
        
        clsloss=nn.BCELoss()(prediction,label.bfloat16())
        out = torch.argmax(prediction.data, 1)
        label1 = torch.argmax(label,dim=1).cuda()
        cls_acc = torch.sum(out == label1).item() / len(out)

        anomaly_maps_cat=torch.cat(anomaly_maps,dim=1)
        anomaly_map_prompts = self.prompt_learner(anomaly_maps_cat,anomaly_map_all,prediction)
        output_texts = inputs['texts']
        input_ids, target_ids, attention_mask = process_batch_instance(self.llama_tokenizer, output_texts, self.max_tgt_len)
        inputs_embeds, targets, attention_mask = self.prompt_wrap(img_embeds, input_ids, target_ids, attention_mask, anomaly_map_prompts, prediction)
        outputs = self.llama_model(
            inputs_embeds=inputs_embeds.bfloat16(),
            attention_mask=attention_mask.bfloat16(),
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss
        chosen_tokens = torch.max(outputs.logits, dim=-1)[1][:, 1:-1]    # [B, S-1]
        labels = targets[:, 2:]
        gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(torch.long)    # [B*S]
        valid_mask = (labels != -100).reshape(-1)
        valid_tokens = gen_acc & valid_mask    # [B*S]
        gen_acc = valid_tokens.sum().item() / valid_mask.sum().item()
        return anomaly_map_all
    
        

    def extract_multimodal_feature(self, inputs, img_tensor=None):
        features = []    
        prompt = inputs['prompt']
        c_name = 'face'
        for name in CLASS_NAMES:
            if name in prompt:
                c_name = name
                break
        image_embeds, _, patch_tokens = self.encode_image_for_web_demo(inputs['image_paths'],img_tensor)
        feats_text_tensor = encode_text_with_prompt_ensemble(self.visual_encoder, [c_name], self.device,self.ctx)
        
        anomaly_maps = []
        if img_tensor is None:
            image_paths=data.load_and_transform_vision_data(inputs['image_paths'], self.device)
            img_tensor=torch.cat([i.unsqueeze(0) for i in image_paths],dim=0)
        for layer in range(len(patch_tokens)):
            patch_tokens[layer] = patch_tokens[layer] / patch_tokens[layer].norm(dim=-1, keepdim=True)
            anomaly_map = (100.0 * patch_tokens[layer] @ feats_text_tensor.transpose(-2,-1))
            B, L, C = anomaly_map.shape
            H = int(np.sqrt(L))
            anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H).float(),
                                        size=self.imagesize, mode='bilinear', align_corners=True)
            anomaly_maps.append(anomaly_map.bfloat16())
        anomaly_map_all = self.locator(anomaly_maps)
        features.append(image_embeds)
    

        feature_embeds = torch.cat(features).sum(dim=0).unsqueeze(0)
        return feature_embeds, anomaly_map_all,anomaly_maps

    def prepare_generation_embedding(self, inputs, img_tensor=None):
        prompt = inputs['prompt']
        feature_embeds, anomaly_map,maps = self.extract_multimodal_feature(inputs, img_tensor=img_tensor)

        prediction=self.classifier(maps)
        pred=prediction
        inputs['modality_embeds'].append(feature_embeds)

        batch_size = feature_embeds.shape[0]
        p_before = PROMPT_START
        p_before_tokens = self.llama_tokenizer(p_before, 
            return_tensors="pt", add_special_tokens=False).to(self.device)
        p_before_embeds = self.llama_model.model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1) # bsz x s1 x embed_dim
        
        if prediction != None:
            if prediction.shape[0]==2:
                p_middle1 = '</Img> According to KFD prediction, the forgery score is '+str(int(prediction[0,1].item()*100))[:5]+'%. '
                p_middle2 = '</Img> According to KFD prediction, the forgery score is '+str(int(prediction[1,1].item()*100))[:5]+'%. '
                p_middle_tokens1 = self.llama_tokenizer(p_middle1, 
                    return_tensors="pt", add_special_tokens=False).to(self.device)
                p_middle_embeds1 = self.llama_model.model.model.embed_tokens(p_middle_tokens1.input_ids) # bsz x s1 x embed_dim
                p_middle_tokens2 = self.llama_tokenizer(p_middle2, 
                    return_tensors="pt", add_special_tokens=False).to(self.device)
                p_middle_embeds2 = self.llama_model.model.model.embed_tokens(p_middle_tokens2.input_ids) # bsz x s1 x embed_dim
                min_len=min(p_middle_embeds1.shape[1],p_middle_embeds2.shape[1])
                p_middle_embeds=torch.cat((p_middle_embeds1[:,:min_len,:],p_middle_embeds2[:,:min_len,:]),dim=0)
            if prediction.shape[0]==1:
                p_middle1 = '</Img> According to deepfake prediction, the forgery score is '+str(int(prediction[0,1].item()*100))[:5]+'%. '
                p_middle_tokens1 = self.llama_tokenizer(p_middle1, 
                    return_tensors="pt", add_special_tokens=False).to(self.device)
                p_middle_embeds = self.llama_model.model.model.embed_tokens(p_middle_tokens1.input_ids) # bsz x s1 x embed_dim
        else:
            p_middle = '</Img> '
            p_middle_tokens = self.llama_tokenizer(p_middle, 
                return_tensors="pt", add_special_tokens=False).to(self.device)
            p_middle_embeds = self.llama_model.model.model.embed_tokens(p_middle_tokens.input_ids).expand(batch_size, -1, -1) # bsz x s1 x embed_dim
        anomaly_map_cat=torch.cat(maps,dim=1)
        anomaly_map_prompts = self.prompt_learner(anomaly_map_cat,anomaly_map[:,1:,:,:],prediction)


        text = prompt + '\n### Assistant:'
        p_after_tokens = self.llama_tokenizer(text, add_special_tokens=False, return_tensors='pt').to(self.device)
        p_after_embeds = self.llama_model.model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1) # bsz x s2 x embed_dim
        bos = torch.ones([batch_size, 1],
                         dtype=p_before_tokens.input_ids.dtype,
                         device=p_before_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id # bsz x 1
        bos_embeds = self.llama_model.model.model.embed_tokens(bos) # bsz x 1 x embed_dim
        prediction=prediction.unsqueeze(1).repeat(1,1,2048)
        if img_tensor is not None:
            inputs_embeds=bos_embeds
        else:
            inputs_embeds = torch.cat([bos_embeds, p_before_embeds, feature_embeds, p_middle_embeds, anomaly_map_prompts, prediction, p_after_embeds], dim=1) # bsz x (1+s1+1+s2) x embed_dim
        anomaly_map=F.interpolate(anomaly_map.float(),(self.imagesize,self.imagesize))

        return inputs_embeds, anomaly_map[:,1:,:,:], pred

    def generate(self, inputs, img_tensor=None):
        '''
            inputs = {
                'image_paths': optional,
                'audio_paths': optional
                'video_paths': optional
                'thermal_paths': optional
                'mode': generation mode,
                'prompt': human input prompt,
                'max_tgt_len': generation length,
                'top_p': top_p,
                'temperature': temperature
                'modality_embeds': None or torch.tensor
                'modality_cache': save the image cache
            }
        '''

        input_embeds, pixel_output, prob = self.prepare_generation_embedding(inputs, img_tensor=img_tensor)
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=[2277], encounters=1)])
        outputs = self.llama_model.generate(
            inputs_embeds=input_embeds.bfloat16(),
            max_new_tokens=inputs['max_tgt_len'],
            top_p=inputs['top_p'],
            temperature=inputs['temperature'],
            do_sample=True,
            use_cache=True,
            stopping_criteria=stopping_criteria,
            attention_mask=torch.ones((input_embeds.shape[0],input_embeds.shape[1])).cuda(),
            pad_token_id=self.llama_tokenizer.eos_token_id
        )
        output_text = self.llama_tokenizer.decode(outputs[0][:-2], skip_special_tokens=True)
        return output_text, pixel_output, prob