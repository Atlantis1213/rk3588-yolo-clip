"""Bounded LN/optional-projection QAT pilot; a training proxy, NOT RKNN validation."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import time
import traceback
import zipfile

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from clip_model_official import VisionTransformer
from qat_visual_core import ActivationQuant, QuantVisual

CHECKPOINT_SHA = 'b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836'
DATA_SHA = '0bf472790f8b20e5c950d5b5012a9d38af0d3392efd65f8ce171334fc16b07c2'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def prepare_inputs(root):
    provenance = json.loads((root/'dataset_provenance.json').read_text())
    validation = json.loads((root/'eval_labels.json').read_text())
    test = json.loads((root/'final_test_labels.json').read_text())
    train_names = provenance['calibration_sources']
    val_names = [row['source'] for row in validation['samples']]
    test_names = [row['source'] for row in test['samples']]
    sets = [set(train_names), set(val_names), set(test_names)]
    if any(len(s) != len(n) for s, n in zip(sets, [train_names,val_names,test_names])):
        raise RuntimeError('duplicate dataset members')
    if any(sets[i] & sets[j] for i,j in [(0,1),(0,2),(1,2)]):
        raise RuntimeError('training/validation/test leakage')
    mean = np.array([.48145466,.4578275,.40821073],dtype=np.float32)
    std = np.array([.26862954,.26130258,.27577711],dtype=np.float32)
    def read(names, archive):
        values = []
        for name in names:
            image = cv2.imdecode(np.frombuffer(archive.read(name),dtype=np.uint8),cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError('cannot decode '+name)
            image = cv2.cvtColor(cv2.resize(image,(224,224)),cv2.COLOR_BGR2RGB)
            # Match existing deployment evaluation's IEEE-FP16-rounded inputs.
            value = ((image.astype(np.float32)/255-mean)/std).astype(np.float16).astype(np.float32)
            values.append(torch.from_numpy(value.transpose(2,0,1).copy()))
        return torch.stack(values)
    with zipfile.ZipFile(root/'assets/dataset-resized.zip') as archive:
        train = read(train_names,archive)
        val = read(val_names,archive)
    labels = torch.tensor([validation['classes'].index(row['class']) for row in validation['samples']])
    return train,val,labels,{'train':train_names,'validation':val_names,'held_out_test':test_names}


def make_visual(root, device):
    checkpoint = torch.jit.load(str(root/'assets/ViT-L-14.pt'), map_location='cpu').eval()
    state = {k[7:]:v for k,v in checkpoint.state_dict().items() if k.startswith('visual.')}
    model = VisionTransformer(224,14,1024,24,16,768).float().eval()
    model.load_state_dict(state,strict=True)
    del state, checkpoint
    gc.collect()
    return model.to(device)


@torch.no_grad()
def embeddings(model, inputs, device):
    result = []
    model.eval()
    for i in range(len(inputs)):
        result.append(F.normalize(model(inputs[i:i+1].to(device)),dim=-1).cpu())
        if (i+1)%20 == 0:
            print(json.dumps({'phase':'embedding_cache','done':i+1,'total':len(inputs)}),flush=True)
    return torch.cat(result)


def metrics(model, inputs, teacher, text, labels, device):
    outputs = embeddings(model,inputs,device)
    cosine = (outputs*teacher).sum(-1)
    prediction = (outputs @ text.T).argmax(-1)
    return {'mean_cosine_to_teacher':cosine.mean().item(),
            'min_cosine_to_teacher':cosine.min().item(),
            'six_class_accuracy':(prediction==labels).float().mean().item(),
            'teacher_top1_agreement':(prediction==(teacher@text.T).argmax(-1)).float().mean().item()}


def distillation_loss(prediction,target,text,kl_weight=.1,rank_weight=0.):
    cosine_loss = 1-(prediction*target).sum(-1).mean()
    student_logits,teacher_logits = prediction@text.T/.07,target@text.T/.07
    kl_loss = F.kl_div(F.log_softmax(student_logits,dim=-1),
                      F.softmax(teacher_logits,dim=-1),reduction='batchmean')
    top = teacher_logits.argmax(-1,keepdim=True)
    teacher_margin = teacher_logits.gather(1,top)-teacher_logits
    student_margin = student_logits.gather(1,top)-student_logits
    # Preserve teacher top1-to-other margins, not validation labels. Ignore top-to-self.
    mask = torch.ones_like(teacher_logits).scatter_(1,top,0.)
    rank_loss = (F.relu(teacher_margin.detach()-student_margin)*mask).sum()/mask.sum()
    return cosine_loss+kl_weight*kl_loss+rank_weight*rank_loss,cosine_loss,kl_loss,rank_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps',type=int,default=300)
    parser.add_argument('--eval-every',type=int,default=50)
    parser.add_argument('--lr',type=float,default=1e-5)
    parser.add_argument('--run',default='qat_ln_pilot_20261010')
    parser.add_argument('--kl-weight',type=float,default=.1)
    parser.add_argument('--rank-weight',type=float,default=0.)
    parser.add_argument('--train-projection',action='store_true')
    args = parser.parse_args()
    if args.steps < 1 or args.eval_every < 1 or args.lr <= 0 or args.kl_weight < 0 or args.rank_weight < 0:
        parser.error('positive steps, eval-every and lr required')
    root = Path(__file__).resolve().parent
    output = root/'runs'/args.run
    output.mkdir(parents=True,exist_ok=True)
    # Never silently overwrite or duplicate a previously started pilot.
    lock = output/'run.lock'
    with lock.open('x') as stream:
        stream.write(str(os.getpid()))
    status = {'phase':'starting','pid':os.getpid(),'steps_requested':args.steps,
              'scope':'official-reference '+('LN+projection' if args.train_projection else 'LN-only')+' W8A8 training proxy; not deployed historical weights',
              'rknn_acceptance':'not evaluated'}
    def update(**values):
        status.update(values)
        status['updated_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
        write_json(output/'status.json',status)
        print(json.dumps(status),flush=True)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; refuse silent CPU training')
        torch.set_num_threads(4)
        torch.manual_seed(20261010)
        np.random.seed(20261010)
        random.seed(20261010)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        for filename,expected in [('assets/ViT-L-14.pt',CHECKPOINT_SHA),('assets/dataset-resized.zip',DATA_SHA)]:
            if sha(root/filename) != expected:
                raise RuntimeError('source SHA mismatch: '+filename)
        train,val,labels,splits = prepare_inputs(root)
        text = F.normalize(torch.from_numpy(np.load(root/'public_text_features.npy')).float(),dim=-1)
        if text.shape != (6,768):
            raise RuntimeError('unexpected fixed text bank dimensions')
        write_json(output/'provenance.json',{'checkpoint_sha':CHECKPOINT_SHA,'dataset_sha':DATA_SHA,
            'splits':splits,'activation_cfg_sha':sha(root/'activation_parameters.json'),
            'text_bank_sha':sha(root/'public_text_features.npy'),
            'preprocessing':'RGB direct resize224, normalize, fp16 round, fp32',
            'training':'LayerNorm gamma/beta'+(' + final projection' if args.train_projection else ' only')+'; cosine + text-distribution KL + optional teacher margin',
            'trainer_sha':sha(Path(__file__)), 'qat_core_sha':sha(root/'qat_visual_core.py'),
            'args':vars(args),'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0)})
        update(phase='teacher_cache')
        teacher = make_visual(root,'cuda')
        teacher_train = embeddings(teacher,train,'cuda')
        teacher_val = embeddings(teacher,val,'cuda')
        torch.save({'train':teacher_train,'val':teacher_val},output/'teacher_features.pt')
        teacher_accuracy = ((teacher_val@text.T).argmax(-1)==labels).float().mean().item()
        # Transfer the same tower into the student; do not hold two full towers on GPU.
        student = QuantVisual(teacher,json.loads((root/'activation_parameters.json').read_text()))
        del teacher
        student.requires_grad_(False)
        for module in student.modules():
            if isinstance(module,nn.LayerNorm):
                module.requires_grad_(True)
        if args.train_projection:
            student.proj.requires_grad_(True)
        student.cache_frozen_weights()
        student.enable_quantization(True)
        missing = [m for m in student.modules() if isinstance(m,ActivationQuant) and not m.ready]
        for module in missing:
            module.calibrating = True
        update(phase='missing_activation_calibration',missing_proxy_sites=len(missing),train_count=len(train))
        with torch.no_grad():
            student.eval()
            for i in range(len(train)):
                student(train[i:i+1].cuda())
        for module in missing:
            module.finish_calibration()
        activations = {m.name:{'scale':[m.scale],'zero_point':[m.zero]} for m in student.modules() if isinstance(m,ActivationQuant)}
        write_json(output/'proxy_activation_parameters.json',activations)
        student.checkpoint_blocks = True
        baseline = metrics(student,val,teacher_val,text,labels,'cuda')
        write_json(output/'baseline.json',{'teacher_accuracy':teacher_accuracy,'proxy_before':baseline})
        names = [name for name,p in student.named_parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],lr=args.lr,weight_decay=0.)
        text_gpu = text.cuda()
        best_cosine = baseline['mean_cosine_to_teacher']
        best_accuracy = baseline['six_class_accuracy']
        def save_checkpoint(filename, step, score):
            path = output/filename
            temporary = path.with_suffix('.tmp')
            torch.save({'delta':{n:p.detach().cpu() for n,p in student.named_parameters() if n in names},
                'source_checkpoint_sha':CHECKPOINT_SHA,'step':step,'proxy_validation':score,
                'activation_parameters':activations,'architecture':'OpenAI ViT-L/14 unchanged',
                'trainable_names':names},temporary)
            os.replace(temporary,path)
        save_checkpoint('best_cosine_delta.pt',0,baseline)
        save_checkpoint('best_accuracy_delta.pt',0,baseline)
        order = []
        started = time.monotonic()
        update(phase='training',step=0,baseline=baseline,teacher_accuracy=teacher_accuracy,
               trainable_parameters=sum(p.numel() for p in student.parameters() if p.requires_grad))
        with (output/'metrics.jsonl').open('a',encoding='utf-8') as log:
            for step in range(1,args.steps+1):
                if not order:
                    order = torch.randperm(len(train)).tolist()
                index = order.pop()
                student.train()
                optimizer.zero_grad(set_to_none=True)
                prediction = F.normalize(student(train[index:index+1].cuda()),dim=-1)
                target = teacher_train[index:index+1].cuda()
                loss,cosine_loss,kl_loss,rank_loss = distillation_loss(
                    prediction,target,text_gpu,args.kl_weight,args.rank_weight)
                if not torch.isfinite(loss):
                    raise RuntimeError('nonfinite training loss')
                loss.backward()
                gradients = [p.grad for p in student.parameters() if p.requires_grad]
                if any(g is None or not torch.isfinite(g).all() for g in gradients):
                    raise RuntimeError('invalid normalization gradients')
                norm = torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad],1.)
                optimizer.step()
                row = {'step':step,'loss':loss.item(),'cosine_loss':cosine_loss.item(),
                    'kl_loss':kl_loss.item(),'rank_loss':rank_loss.item(),'grad_norm':float(norm),'elapsed_seconds':time.monotonic()-started,
                    'peak_cuda_mib':torch.cuda.max_memory_allocated()/2**20}
                if step%args.eval_every==0 or step==args.steps:
                    row['validation'] = metrics(student,val,teacher_val,text,labels,'cuda')
                    score = row['validation']
                    save_checkpoint('latest_delta.pt',step,score)
                    if score['mean_cosine_to_teacher'] > best_cosine:
                        best_cosine = score['mean_cosine_to_teacher']
                        save_checkpoint('best_cosine_delta.pt',step,score)
                    if score['six_class_accuracy'] > best_accuracy:
                        best_accuracy = score['six_class_accuracy']
                        save_checkpoint('best_accuracy_delta.pt',step,score)
                log.write(json.dumps(row)+'\n'); log.flush()
                if step%10==0 or step==1:
                    update(phase='training',**row)
            update(phase='completed',step=args.steps,best_proxy_cosine=best_cosine,
                   best_proxy_accuracy=best_accuracy,final_proxy_validation=row['validation'])
    except Exception as error:
        update(phase='failed',error=repr(error),traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
