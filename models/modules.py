import torch
from torch import nn
from pathlib import Path
from copy import deepcopy

class Mish(nn.Module):
  def forward(self, x):
    return x * torch.nn.functional.softplus(x).tanh()
            
class Conv(nn.Module):
  # Standard convolution
  def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
    super(Conv, self).__init__()
    self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
    self.bn = nn.BatchNorm2d(c2)
    self.act = Mish() if act else nn.Identity()

  def forward(self, x):
    return self.act(self.bn(self.conv(x)))

  def fuseforward(self, x):
    return self.act(self.conv(x))

class Bottleneck(nn.Module):
  # Standard bottleneck
  def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
    super(Bottleneck, self).__init__()
    c_ = int(c2 * e)  # hidden channels
    self.cv1 = Conv(c1, c_, 1, 1)
    self.cv2 = Conv(c_, c2, 3, 1, g=g)
    self.add = shortcut and c1 == c2

  def forward(self, x):
    return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class BottleneckCSP(nn.Module):
  # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
  def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
    super(BottleneckCSP, self).__init__()
    c_ = int(c2 * e)  # hidden channels
    self.cv1 = Conv(c1, c_, 1, 1)
    self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
    self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
    self.cv4 = Conv(2 * c_, c2, 1, 1)
    self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
    self.act = Mish()
    self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

  def forward(self, x):
    y1 = self.cv3(self.m(self.cv1(x)))
    y2 = self.cv2(x)
    return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class BottleneckCSP2(nn.Module):
  # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
  def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
    super(BottleneckCSP2, self).__init__()
    c_ = int(c2)  # hidden channels
    self.cv1 = Conv(c1, c_, 1, 1)
    self.cv2 = nn.Conv2d(c_, c_, 1, 1, bias=False)
    self.cv3 = Conv(2 * c_, c2, 1, 1)
    self.bn = nn.BatchNorm2d(2 * c_) 
    self.act = Mish()
    self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

  def forward(self, x):
    x1 = self.cv1(x)
    y1 = self.m(x1)
    y2 = self.cv2(x1)
    return self.cv3(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class VoVCSP(nn.Module):
  # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
  def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
    super(VoVCSP, self).__init__()
    c_ = int(c2)  # hidden channels
    self.cv1 = Conv(c1//2, c_//2, 3, 1)
    self.cv2 = Conv(c_//2, c_//2, 3, 1)
    self.cv3 = Conv(c_, c2, 1, 1)

  def forward(self, x):
    _, x1 = x.chunk(2, dim=1)
    x1 = self.cv1(x1)
    x2 = self.cv2(x1)
    return self.cv3(torch.cat((x1,x2), dim=1))


class SPP(nn.Module):
  # Spatial pyramid pooling layer used in YOLOv3-SPP
  def __init__(self, c1, c2, k=(5, 9, 13)):
    super(SPP, self).__init__()
    c_ = c1 // 2  # hidden channels
    self.cv1 = Conv(c1, c_, 1, 1)
    self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
    self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

  def forward(self, x):
    x = self.cv1(x)
    return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPCSP(nn.Module):
  # CSP SPP https://github.com/WongKinYiu/CrossStagePartialNetworks
  def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
    super(SPPCSP, self).__init__()
    c_ = int(2 * c2 * e)  # hidden channels
    self.cv1 = Conv(c1, c_, 1, 1)
    self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
    self.cv3 = Conv(c_, c_, 3, 1)
    self.cv4 = Conv(c_, c_, 1, 1)
    self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
    self.cv5 = Conv(4 * c_, c_, 1, 1)
    self.cv6 = Conv(c_, c_, 3, 1)
    self.bn = nn.BatchNorm2d(2 * c_) 
    self.act = Mish()
    self.cv7 = Conv(2 * c_, c2, 1, 1)

  def forward(self, x):
    x1 = self.cv4(self.cv3(self.cv1(x)))
    y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
    y2 = self.cv2(x)
    return self.cv7(self.act(self.bn(torch.cat((y1, y2), dim=1))))

class Focus(nn.Module):
  # Focus wh information into c-space
  def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
    super(Focus, self).__init__()
    self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

  def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
    return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))

def DWConv(c1, c2, k=1, s=1, act=True):
  # Depthwise convolution
  return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)

def autopad(k, p=None):  # kernel, padding
  # Pad to 'same'
  if p is None:
      p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
  return p

class Concat(nn.Module):
  # Concatenate a list of tensors along dimension
  def __init__(self, dimension=1):
    super(Concat, self).__init__()
    self.d = dimension

  def forward(self, x):
    return torch.cat(x, self.d)

class MixConv2d(nn.Module):
  # Mixed Depthwise Conv https://arxiv.org/abs/1907.09595
  def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):
      super(MixConv2d, self).__init__()
      groups = len(k)
      if equal_ch:  # equal c_ per group
          i = torch.linspace(0, groups - 1E-6, c2).floor()  # c2 indices
          c_ = [(i == g).sum() for g in range(groups)]  # intermediate channels
      else:  # equal weight.numel() per group
          b = [c2] + [0] * groups
          a = np.eye(groups + 1, groups, k=-1)
          a -= np.roll(a, 1, axis=1)
          a *= np.array(k) ** 2
          a[0] = 1
          c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # solve for equal weight indices, ax = b

      self.m = nn.ModuleList([nn.Conv2d(c1, int(c_[g]), k[g], s, k[g] // 2, bias=False) for g in range(groups)])
      self.bn = nn.BatchNorm2d(c2)
      self.act = nn.LeakyReLU(0.1, inplace=True)

  def forward(self, x):
      return x + self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))

class CrossConv(nn.Module):
  # Cross Convolution Downsample
  def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
      # ch_in, ch_out, kernel, stride, groups, expansion, shortcut
      super(CrossConv, self).__init__()
      c_ = int(c2 * e)  # hidden channels
      self.cv1 = Conv(c1, c_, (1, k), (1, s))
      self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
      self.add = shortcut and c1 == c2

  def forward(self, x):
      return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    # Cross Convolution CSP
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))