import torch

from base_model import Loss
from train import dboxes300_coco
from opt_loss import OptLoss

# In:
#  ploc : N x 8732 x 4
#  plabel : N x 8732
#  gloc : N x 8732 x 4
#  glabel : N x 8732

data = torch.load('loss.pth')
ploc = data['ploc'].cuda()
plabel = data['plabel'].cuda()
gloc = data['gloc'].cuda()
glabel = data['glabel'].cuda()

dboxes = dboxes300_coco()
# loss = Loss(dboxes).cuda()
loss = OptLoss(dboxes).cuda()

loss = torch.jit.trace(loss, (ploc, plabel, gloc, glabel))
# print(traced_loss.graph)

# timing
timing_iterations = 1000

import time

# Dry run to eliminate JIT compile overhead
dl = torch.tensor([1.], device="cuda")
l = loss(ploc, plabel, gloc, glabel)
l.backward(dl)

# fprop
torch.cuda.synchronize()
start = time.time()
with torch.no_grad():
    for _ in range(timing_iterations):
        l = loss(ploc, plabel, gloc, glabel)

print('loss: {}'.format(l))
torch.cuda.synchronize()
end = time.time()

time_per_fprop = (end - start) / timing_iterations

print('took {} seconds per iteration (fprop)'.format(time_per_fprop))

# fprop + bprop
torch.cuda.synchronize()
start = time.time()
for _ in range(timing_iterations):
    l = loss(ploc, plabel, gloc, glabel)
    l.backward(dl)

torch.cuda.synchronize()
end = time.time()

time_per_fprop_bprop = (end - start) / timing_iterations

print('took {} seconds per iteration (fprop + bprop)'.format(time_per_fprop_bprop))

print(loss.graph_for(ploc, plabel, gloc, glabel))
