import torch, os, argparse
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt',  default='pretrained/co3d.ckpt')
parser.add_argument('--out',   default='pretrained/autoencoder/kl_f8.pt')
args = parser.parse_args()

pl_ckpt = torch.load(args.ckpt, map_location='cpu')
# latentSplat 把 VAE 放在 autoencoder 字段
vae_state = pl_ckpt['state_dict']
# 只保留 autoencoder 键
vae_state = {k.replace('autoencoder.', ''): v for k in vae_state if k.startswith('autoencoder.')}
os.makedirs(os.path.dirname(args.out), exist_ok=True)
torch.save(vae_state, args.out)
print('✅ 已写入', args.out, '- 大小 %.1f MB' % (os.path.getsize(args.out)/1024/1024))
