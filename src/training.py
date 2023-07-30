import torch
import numpy as np # linear algebra

# cv = np.array([[0,0,0,0,1],[1,0,0,0,0]])
# rv = np.array([[.5, .5, .5, .5, 1],
#                [.5, .5, .5, .5, .5],
#                [.5, .5, .5, .5, .5],
#                [.5, .5, .5, .5, .5],
#                [.5, 1, .5, .5, .5]])
# x = cv @ rv
# print(x)
# print(x.shape)
names = open('./data/names.txt', 'r').read().splitlines()

#Create dictionary of available letters
setLetters = list(set(list(''.join(names))))
setLetters.sort()

# {'.': 0, 'a': 1, ... , 'z': 27}
stoi = {'.': 0}; itos = {0: '.'}
for i,s in enumerate(setLetters):
  i = i+1
  stoi[s] = i
  itos[i] = s

# Create inputs array (. , E, M, M, A), shape [n, 27] each row is (27,)
# Create outputs array (E, M, M, A, .), shape [n, 27] each row is (27,)
inp = []
outp = []
for name in names:
  name = '.' + name + '.'
  bigram = list(zip(name, name[1:]))
  for i,o in bigram:
    inp.append(stoi[i])
    outp.append(stoi[o])

inp = torch.tensor(inp)
outp = torch.tensor(outp)
num = inp.nelement()
#Create weights tensor of shape [27, 27]
W = torch.randn((27,27), requires_grad=True)

epochs = 100
while epochs > 0:
  epochs = epochs - 1
  xenc = torch.nn.functional.one_hot(inp, num_classes=27).float()
  #train
  logits = xenc @ W
  #next two lines are softmax function
  counts = logits.exp()
  # probs = how likely 1-27 characters are next
  probs = counts / counts.sum(1, keepdims=True)

  # We are doing CLASSIFICATION not REGRESSION 
  # so we are using negative log likeliness
  loss = -probs[torch.arange(num), outp].log().mean()
  print(loss.item())
  W.grad = None # set the gradient to zero
  loss.backward()
  W.data += -50 * W.grad
  #use some loss function
  #back-propogate from loss
  # report loss

  

