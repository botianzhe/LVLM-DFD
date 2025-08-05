import torch
from torch import nn
import numpy as np
# from datas.dataset_3d import  *
from torch.nn import functional as F
from model.clip import clip
from model.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

from collections import OrderedDict

class Normalize(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.nn.functional.normalize(x, dim=self.dim, p=2)

    
class LinearLayer(nn.Module):
    def __init__(self, dim_in, dim_out, k):
        super(LinearLayer, self).__init__()
        self.fc = nn.ModuleList([nn.Linear(dim_in, dim_out) for i in range(k)])

    def forward(self, tokens):
        for i in range(len(tokens)):
            if len(tokens[i].shape) == 3:
                # print('AAAAAA',tokens[i].dtype,tokens[i].shape)
                tokens[i] = tokens[i].transpose(0,1)
                # print('AAAAAB',tokens[i].dtype,tokens[i].shape)
                # print('AAAAAC', self.fc[i].weight.dtype)
                if self.fc[i].weight.dtype==torch.float32:
                    tokens[i] = self.fc[i](tokens[i][:, 1:, :].float())
                else:
                    tokens[i] = self.fc[i](tokens[i][:, 1:, :])
                
            else:
                B, C, H, W = tokens[i].shape
                tokens[i] = self.fc[i](tokens[i].view(B, C, -1).permute(0, 2, 1).contiguous())
        return tokens

class ConvLayer(nn.Module):
    def __init__(self, dim_in, dim_out, k):
        super(ConvLayer, self).__init__()
        self.conv = nn.ModuleList([nn.Conv2d(dim_in, dim_out,3,padding=2) for i in range(k)])
        self.batchnorm=nn.ModuleList([nn.BatchNorm2d(dim_out) for i in range(k)])
        self.finalconv=nn.Conv2d(dim_out*k,2,3)

    def forward(self, maps):
        newmap=[]
        for i in range(len(maps)):
            B, C, H, W = maps[i].shape
            # print(self.conv[i].weight.dtype)
            if self.conv[i].weight.dtype==torch.float32:
                maps[i]=maps[i].float()
            else:
                maps[i]=maps[i].bfloat16()
            t = self.conv[i](maps[i])
            t = nn.ReLU()(t)
            t = self.batchnorm[i](t)
            newmap.append(t)
        newmap=torch.cat(newmap,dim=1)
        res=self.finalconv(newmap)
        res=nn.Softmax(dim=1)(res)
        return res

class SimUpsample(nn.Module):
    def __init__(self):
        super(SimUpsample, self).__init__()
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False)
        self.conv1=nn.Conv2d(1,16,3,padding=2)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False)
        self.conv2=nn.Conv2d(16,1,3,padding=2)
        self.up3 = nn.Upsample(size=(224,224), mode='bilinear',align_corners=False)
        # self.conv3=nn.Conv2d(32,1,3,padding=2)

    def forward(self, x):
        x= F.interpolate(x,size=56, mode='bilinear', align_corners=False)
        x=nn.ReLU()(self.conv1(x))
        x= F.interpolate(x,size=112, mode='bilinear', align_corners=False)
        x=nn.ReLU()(self.conv2(x))
        x= F.interpolate(x,size=224, mode='bilinear', align_corners=False)
        
        return x

class ForgeryPromptLearner(nn.Module):
    def __init__(self, dim_in, dim_out) -> None:
        super().__init__()
        # self.conv = nn.ModuleList([nn.Conv2d(dim_in, dim_out,3,padding=2) for i in range(k)])
        # self.batchnorm=nn.ModuleList([nn.BatchNorm2d(dim_out) for i in range(k)])
        
        self.meta_net1 = nn.Sequential(
            nn.Conv2d(dim_in, 4 * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4 * 4),
            nn.MaxPool2d(2), # 112 * 112

            nn.Conv2d(4 * 4, 4 * 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4 * 8),
            nn.MaxPool2d(2), # 56 * 56

            nn.Conv2d(4 * 8, 4 * 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4 * 16),
            nn.MaxPool2d(2), # 28 * 28

            nn.Conv2d(4 * 16, 4 * 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4 * 32),
            nn.MaxPool2d(2), # 14 * 14

            nn.Conv2d(4 * 32, 4 * 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4 * 64),
            nn.MaxPool2d(2), # 7 * 7

            nn.Conv2d(4 * 64, 4 * 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4 * 128),
            # nn.MaxPool2d(2), # 7 * 7

            nn.Conv2d(4 * 128, dim_out, kernel_size=5, padding=0),
            # nn.BatchNorm2d(dim_out),
            # nn.ReLU(inplace=True),
        )

        self.meta_net2 = nn.Sequential(
            nn.Conv2d(1, 4 * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4 * 4),
            nn.MaxPool2d(2), # 112 * 112

            nn.Conv2d(4 * 4, 4 * 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4 * 8),
            nn.MaxPool2d(2), # 56 * 56

            nn.Conv2d(4 * 8, 4 * 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4 * 16),
            nn.MaxPool2d(2), # 28 * 28

            nn.Conv2d(4 * 16, 4 * 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4 * 32),
            nn.MaxPool2d(2), # 14 * 14

            nn.Conv2d(4 * 32, 4 * 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4 * 64),
            nn.MaxPool2d(2), # 7 * 7

            nn.Conv2d(4 * 64, 4 * 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4 * 128),
            # nn.MaxPool2d(2), # 7 * 7

            nn.Conv2d(4 * 128, dim_out, kernel_size=5, padding=0),
            # nn.BatchNorm2d(dim_out),
            # nn.ReLU(inplace=True),
        )
        self.fc=nn.Linear(2,4096)
        self.finalconv=nn.Conv1d(28,9,3,1,1)
        self.base_prompts = nn.Parameter(torch.randn((9, dim_out)),requires_grad=True)

    def forward(self, cons,loc,clss):
        B,C,H,W = cons.shape
        t1 = self.meta_net1(cons)
        # print('t1.shape',t1.shape)
        t2 = self.meta_net2(loc)
        # print('t2.shape',t2.shape)
        t3 = self.fc(clss)
        # print('t3.shape',t3.shape)
        t1 = t1.reshape(B,4096,9).transpose(-2,-1)
        t2 = t2.reshape(B,4096,9).transpose(-2,-1)
        
        output = torch.cat([self.base_prompts.expand(B,-1,-1), t1,t2,t3.unsqueeze(1)], dim=1)
        # print(t1.shape,t2.shape,output.shape)
        output = self.finalconv(output)
        # print(output.shape)
        # newmap=torch.cat((t1,t2,t3),dim=1)
        # res=self.finalconv(newmap)
        return output

class PromptLearner(nn.Module):
    def __init__(self, dim_in, dim_out) -> None:
        super().__init__()
        self.meta_net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(dim_in * 4),
            nn.MaxPool2d(2), # 112 * 112

            nn.Conv2d(dim_in * 4, dim_in * 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(dim_in * 16),
            nn.MaxPool2d(2), # 56 * 56

            nn.Conv2d(dim_in * 16, dim_in * 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(dim_in * 64),
            nn.MaxPool2d(2), # 28 * 28

            nn.Conv2d(dim_in * 64, dim_in * 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(dim_in * 256),
            nn.MaxPool2d(2), # 14 * 14

            nn.Conv2d(dim_in * 256, dim_in * 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(dim_in * 512),
            nn.MaxPool2d(2), # 7 * 7

            nn.Conv2d(dim_in * 512, dim_in * 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(dim_in * 1024),
            # nn.MaxPool2d(2), # 7 * 7

            nn.Conv2d(dim_in * 1024, dim_out, kernel_size=5, padding=0),
            # nn.BatchNorm2d(dim_out),
            # nn.ReLU(inplace=True),
        )
        self.base_prompts = nn.Parameter(torch.randn((9, dim_out)),requires_grad=True)

    def forward(self, input):
        B,C,H,W = input.shape
        img_prompts = self.meta_net(input)
        # print(input.shape, img_prompts.shape)
        img_prompts = img_prompts.reshape(B,4096,9).transpose(-2,-1)
        output = torch.cat([self.base_prompts.expand(B,-1,-1), img_prompts], dim=1)
        return output

class LinearClassifier(nn.Module):
    def __init__(self):
        super(LinearClassifier, self).__init__()
        self.fc1=nn.Linear(4096,512)
        self.fc2=nn.Linear(512,128)
        self.fc3=nn.Linear(128,2)
        # self.conv3=nn.Conv2d(32,1,3,padding=2)

    def forward(self, x):
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = nn.ReLU()(self.fc3(x))
        x = nn.Softmax(dim=1)(x)
        
        
        return x

# class Classifier(nn.Module):
#     def __init__(self, dim_in) -> None:
#         super().__init__()
#         self.meta_net = nn.Sequential(
#             nn.Conv2d(dim_in, dim_in * 8, kernel_size=3, padding=1),
#             # nn.BatchNorm2d(dim_in * 4),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2), # 112 * 112
#             nn.BatchNorm2d(dim_in * 8),

#             nn.Conv2d(dim_in * 8, dim_in * 32, kernel_size=3, padding=1),
#             # nn.BatchNorm2d(dim_in * 16),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2), # 56 * 56
#             nn.BatchNorm2d(dim_in * 32),

#             nn.Conv2d(dim_in * 32, dim_in * 64, kernel_size=3, padding=1),
#             # nn.BatchNorm2d(dim_in * 64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2), # 28 * 28
#             nn.BatchNorm2d(dim_in * 64),

#             nn.Conv2d(dim_in * 64, dim_in * 128, kernel_size=3, padding=1),
#             # nn.BatchNorm2d(dim_in * 64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2), # 28 * 28
#             nn.BatchNorm2d(dim_in * 128),

#             nn.AdaptiveAvgPool2d((1,1))
#         )
#         self.fc1=nn.Linear(dim_in * 128, 128)
#         self.fc2=nn.Linear(128, 2)

#     def forward(self, input):
#         B,C,H,W = input.shape
#         if input.dtype==torch.float32:
#             input=input.bfloat16()
                
#         feature = self.meta_net(input).view(B,-1)
#         # print("feature",feature.shape)
#         x=nn.ReLU()(self.fc1(feature))
#         x=nn.Softmax(dim=1)(self.fc2(x))
#         return x

class Classifier(nn.Module):
    def __init__(self, dim_in, k) -> None:
        super().__init__()
        self.meta_net = nn.ModuleList([nn.Sequential(
            nn.Conv2d(dim_in, dim_in * 8, kernel_size=3, padding=1),
            # nn.BatchNorm2d(dim_in * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 112 * 112
            nn.BatchNorm2d(dim_in * 8),

            nn.Conv2d(dim_in * 8, dim_in * 32, kernel_size=3, padding=1),
            # nn.BatchNorm2d(dim_in * 16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 56 * 56
            nn.BatchNorm2d(dim_in * 32),

            nn.Conv2d(dim_in * 32, dim_in * 64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(dim_in * 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 28 * 28
            nn.BatchNorm2d(dim_in * 64),

            nn.Conv2d(dim_in * 64, dim_in * 128, kernel_size=3, padding=1),
            # nn.BatchNorm2d(dim_in * 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 28 * 28
            nn.BatchNorm2d(dim_in * 128),

            nn.AdaptiveAvgPool2d((1,1))
        ) for i in range(k)])
        self.fc1=nn.Linear(768, 128)
        self.fc2=nn.Linear(128, 2)

    def forward(self, maps):
        newmap=[]
        for i in range(len(maps)):
            # print(maps[i].shape)
            B, C, H, W = maps[i].shape
            # print(self.conv[i].weight.dtype)
            t = self.meta_net[i](maps[i])
            newmap.append(t)
        newmap=torch.cat(newmap,dim=1)
        # print(newmap.shape)        
        # feature = self.meta_net(input).view(B,-1)
        # print("feature",feature.shape)
        x=nn.ReLU()(self.fc1(newmap.squeeze(-1).squeeze(-1)))
        x=nn.Softmax(dim=1)(self.fc2(x))
        return x

def load_clip_to_cpu():

    backbone_name = "ViT-B/16"
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextPromptLearner(nn.Module):
    def __init__(self, classnames,clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 4
        ctx_init = "a photo of a"
        # clip_model=load_clip_to_cpu()
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)
        self.image_encoder = clip_model.visual

        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ]))
        
        # if cfg.TRAINER.COCOOP.PREC == "fp16":
        # self.meta_net.half()

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
    
    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,     # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, images):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx                     # (n_ctx, ctx_dim)
        im_features=self.image_encoder(images)
        bias = self.meta_net(im_features)  # (batch, ctx_dim)
        bias = bias.unsqueeze(1)           # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)             # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias           # (batch, n_ctx, ctx_dim)
        
        # Use instance-conditioned context tokens for all classes
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)
        
        return prompts

# class CustomCLIP(nn.Module):
#     def __init__(self, cfg, classnames, clip_model):
#         super().__init__()
#         self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
#         self.tokenized_prompts = self.prompt_learner.tokenized_prompts
#         clip_model=load_clip_to_cpu()
#         self.image_encoder = clip_model.visual
#         self.text_encoder = TextEncoder(clip_model)
#         self.logit_scale = clip_model.logit_scale
#         self.dtype = clip_model.dtype

#     def forward(self, image):
#         image_features = self.image_encoder(image.type(self.dtype))

#         prompts = self.prompt_learner()
#         tokenized_prompts = self.tokenized_prompts
#         text_features = self.text_encoder(prompts, tokenized_prompts)

#         image_features = image_features / image_features.norm(dim=-1, keepdim=True)
#         text_features = text_features / text_features.norm(dim=-1, keepdim=True)

#         logit_scale = self.logit_scale.exp()
#         logits = logit_scale * image_features @ text_features.t()

#         return logits

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class CPromptLearner(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 4
        ctx_init = "a photo of a"
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ]))
        
        # if cfg.TRAINER.COCOOP.PREC == "fp16":
        # self.meta_net.half()

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
    
    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,     # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, im_features):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx                     # (n_ctx, ctx_dim)
        bias = self.meta_net(im_features)  # (batch, ctx_dim)
        bias = bias.unsqueeze(1)           # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)             # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias           # (batch, n_ctx, ctx_dim)
        
        # Use instance-conditioned context tokens for all classes
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)
        
        return prompts


class CustomCLIP(nn.Module):
    def __init__(self,  classnames):
        super().__init__()
        self.clip_model=load_clip_to_cpu()
        # lora.mark_only_lora_as_trainable(self.clip_model)
        self.prompt_learner = CPromptLearner(classnames, self.clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = self.clip_model.visual
        self.text_encoder = TextEncoder(self.clip_model)
        self.logit_scale = self.clip_model.logit_scale
        self.visual_hidden_size=self.image_encoder.output_dim
        self.dtype = self.clip_model.dtype

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        image_features = self.image_encoder(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        B,H=image_features.shape
        prompts = self.prompt_learner(image_features)
        
        logits = []
        for pts_i, imf_i in zip(prompts, image_features):
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            # print(imf_i.shape,text_features.t().shape)
            # l_i = logit_scale * imf_i @ text_features.t()
            l_i = logit_scale * imf_i.unsqueeze(1).repeat(1,2) @ text_features
            logits.append(l_i)
        logits = torch.stack(logits)
        anomaly_map = F.interpolate(logits.view(B, 1, H, H), size=224, mode='bilinear', align_corners=True)
        anomaly_map = nn.Sigmoid()(anomaly_map)
        # print('anomaly_map',anomaly_map.shape)
        # if self.prompt_learner.training:
        #     return F.cross_entropy(logits, label)
        
        return image_features,anomaly_map
