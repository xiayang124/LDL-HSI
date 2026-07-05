try:
    import extract_text.clip as clip
    import extract_text.text_encoder as encoder
except:
    import clip
    import text_encoder as encoder
import os
import torch


def get_text_embedding(lable_value: dict, args):
    device = args.TrainingSetting.get('device', 'cuda')
    dataset = args.BaseSetting.get('dataset', 'Indian')
    
    if (os.path.exists(f"./model_save/{dataset}_text_logit.pt") and not args.BaseSetting.get("change_text", False)):
        text_token = torch.load(f"./model_save/{dataset}_text_token.pt").to(device)
        text_logit = torch.load(f"./model_save/{dataset}_text_logit.pt").to(device)
        embed_dim = text_logit.shape[-1]
        return text_token, text_logit, embed_dim
    else:
        text_token_dict = {k: None for k in lable_value.keys()}
        text_logit_dict = {k: None for k in lable_value.keys()}
        pretrain_loc = args.TrainingSetting.get('pretrain_loc')

        text_encoder, embed_dim = load_text_encoder(pretrain_loc)
        text_encoder.eval()

        with torch.no_grad():
            for key, value in lable_value.items():
                text_token = clip.tokenize(value).to(device)
                text_tokens, text_logits = text_encoder(text_token)
                text_tokens, text_logits = text_tokens.detach(), text_logits.detach()
                text_token_dict[key] = text_tokens
                text_logit_dict[key] = text_logits
        text_token = prepare_text_embedding_tensor(text_token_dict)
        text_logit = prepare_text_embedding_tensor(text_logit_dict)
        torch.save(text_token, f"./model_save/{dataset}_text_token.pt")
        torch.save(text_logit, f"./model_save/{dataset}_text_logit.pt")
        return text_token, text_logit, embed_dim


def get_text_token(lable_value: dict, args):
    device = args.TrainingSetting.get('device', 'cuda')
    dataset = args.BaseSetting.get('dataset', 'Indian')
    
    if (os.path.exists(f"./model_save/{dataset}_text_token.pt") and not args.BaseSetting.get("change_text", False)):
        text_token = torch.load(f"./model_save/{dataset}_text_token.pt").to(device)
        return text_token
    else:
        text_token_dict = {k: None for k in lable_value.keys()}

        with torch.no_grad():
            for key, value in lable_value.items():
                text_token = clip.tokenize(value).to(device)
                text_token_dict[key] = text_token
        text_token = prepare_text_embedding_tensor(text_token_dict)
        torch.save(text_token, f"./model_save/{dataset}_text_token.pt")
        return text_token


def prepare_text_embedding_tensor(embedding_dict):
        sample_tensor = next(iter(embedding_dict.values())).squeeze()
        num_classes = len(embedding_dict)
        embed_dim = sample_tensor.shape[-1]
        if len(sample_tensor.shape) == 1:
            embeddings = torch.zeros(num_classes, embed_dim, dtype=sample_tensor.dtype, device=sample_tensor.device)
        else:
            embeddings = torch.zeros(num_classes, sample_tensor.shape[0], embed_dim, dtype=sample_tensor.dtype, device=sample_tensor.device)
        for label, embedding in embedding_dict.items():
            idx = int(label) - 1
            embeddings[idx] = embedding.detach().to(sample_tensor.device)
        return embeddings
    

def load_text_encoder(pt_loc):
    pretrained_dict = torch.jit.load(pt_loc, map_location="cuda").state_dict()
    embed_dim = pretrained_dict["text_projection"].shape[1]
    context_length = pretrained_dict["positional_embedding"].shape[0]
    vocab_size = pretrained_dict["token_embedding.weight"].shape[0]
    transformer_width = pretrained_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = 12

    text_encoder = encoder.Text_encoder(
        embed_dim,
        context_length,
        vocab_size,
        transformer_width,
        transformer_heads,
        transformer_layers
    )

    model_state = text_encoder.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_state and 'visual' not in k.split('.')}

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in pretrained_dict:
            del pretrained_dict[key]

    model_state.update(pretrained_dict)
    text_encoder.load_state_dict(model_state)
    return text_encoder.cuda(), embed_dim