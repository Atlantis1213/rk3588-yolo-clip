"""Objective regression: same teacher => zero; wrong ranking => finite corrective gradient."""
import torch
from torch.nn import functional as F
from train_qat import distillation_loss

torch.manual_seed(20261010)
text=F.normalize(torch.randn(6,8),dim=-1)
teacher=F.normalize(torch.randn(2,8),dim=-1)
zero,*_=distillation_loss(teacher,teacher,text,5.,1.)
torch.testing.assert_close(zero,torch.zeros_like(zero),atol=2e-6,rtol=0.)
student=(-teacher).clone().requires_grad_(True)
loss,cosine,kl,rank=distillation_loss(student,teacher,text,5.,1.)
assert float(rank)>0 and torch.isfinite(loss)
loss.backward()
assert student.grad is not None and torch.isfinite(student.grad).all() and float(student.grad.abs().sum())>0
old=distillation_loss(student.detach(),teacher,text,.1,0.)[0]
torch.testing.assert_close(old,cosine+.1*kl)
print('PASS: identical teacher, corrective ranking gradient, original objective preserved')
