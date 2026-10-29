"""CPU regression seam for the GPU training proxy; no hardware dependencies."""
import argparse
import torch
from clip_model_official import VisionTransformer
from qat_visual_core import QuantVisual, ActivationQuant

parser=argparse.ArgumentParser()
parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
args=parser.parse_args()
torch.set_num_threads(2)
torch.manual_seed(20261010)
observer = ActivationQuant({},'regression_observer')
observer.enabled = True
observer.calibrating = True
observer(torch.tensor([-1.,2.],device=args.device))
observer(torch.tensor([-3.,4.],device=args.device))
observer.finish_calibration()
assert observer.ready and abs(observer.scale-7/255)<1e-10
frozen_scale = observer.scale
observer(torch.tensor([10.],device=args.device))
assert observer.scale == frozen_scale, 'validation must not recalibrate activation ranges'
original=VisionTransformer(8,4,16,2,4,8).float().eval().to(args.device)
expected_keys=set(original.state_dict())
x=torch.randn(2,3,8,8,device=args.device)
with torch.no_grad():
    expected=original(x)
model=QuantVisual(original,{})
with torch.no_grad():
    actual=model(x)
torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
model.requires_grad_(False)
for name,parameter in model.named_parameters():
    if 'ln_' in name:
        parameter.requires_grad_(True)
model.cache_frozen_weights()
model.enable_quantization(True)
model.checkpoint_blocks=True
model.train()
actual=model(x)
loss=(actual-expected).square().mean()
if not torch.isfinite(loss):
    raise RuntimeError('nonfinite fake-quant loss')
loss.backward()
gradients=[p.grad for p in model.parameters() if p.requires_grad]
if not gradients or any(g is None or not torch.isfinite(g).all() for g in gradients):
    raise RuntimeError('missing/nonfinite normalization gradients')
if sum(float(g.abs().sum()) for g in gradients)==0:
    raise RuntimeError('zero fake-quant training signal')
keys=set(model.canonical_state())
if keys!=expected_keys:
    raise RuntimeError('canonical checkpoint changes original architecture keys')
# Trainable final projection must use STE weights, never a detached stale cache.
model.zero_grad(set_to_none=True)
model.proj.requires_grad_(True)
model.cache_frozen_weights()
assert model.cached_proj.numel()==0
projection_loss=(model(x)-expected).square().mean()
projection_loss.backward()
assert model.proj.grad is not None and torch.isfinite(model.proj.grad).all()
assert float(model.proj.grad.abs().sum())>0
print('PASS: float identity, finite fake-quant forward/backward, checkpointing, unchanged architecture keys')
