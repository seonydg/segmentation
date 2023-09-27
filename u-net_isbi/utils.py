import os
import torch

def save(ckpt_dir, model, optimizer, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    
    torch.save({'model':model.state_dict(), 'optimizer':optimizer.state_dict()}, f'./{ckpt_dir}/model_epoch_{epoch}.pth')


def load(ckpt_dir, model, optimizer):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return model, optimizer, epoch
    
    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load(f'./{ckpt_dir}/{ckpt_lst[-1]}')

    model.load_state_dict(dict_model['model'])
    optimizer.load_state_dict(dict_model['optimizer'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return model, optimizer, epoch