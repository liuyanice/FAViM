import torch


class XBM1:
    def __init__(self):
        self.K = 20
        self.feats = torch.zeros(self.K, 320, 24, 24).cuda()#384 is 24, 24; 256 is 16 16; 224 is 14 14
      #  self.targets = torch.zeros(self.K, dtype=torch.long).cuda()
        self.ptr = 0

    @property
    def is_full(self):
        return (self.feats[-1:,:,:,:].cpu().numpy() == 0).all()

    def get(self):
        if self.is_full:
            return self.feats[:self.ptr,:,:,:]
        else:
            return self.feats

    def enqueue_dequeue(self, feats):
        q_size = feats.size()[0]
        if self.ptr + q_size > self.K:
            self.feats[-q_size:,:,:,:] = feats
         #   self.targets[-q_size:] = targets
            self.ptr = 0
        else:
            self.feats[self.ptr: self.ptr + q_size,:,:,:] = feats
            self.ptr += q_size
            
class XBM2:
    def __init__(self):
        self.K = 20
        self.feats = torch.zeros(self.K, 1, 24, 24).cuda()#384 is 24, 24; 256 is 16 16
      #  self.targets = torch.zeros(self.K, dtype=torch.long).cuda()
        self.ptr = 0

    @property
    def is_full(self):
        return (self.feats[-1:,:,:,:].cpu().numpy() == 0).all()

    def get(self):
        if self.is_full:
            return self.feats[:self.ptr,:,:,:]
        else:
            return self.feats

    def enqueue_dequeue(self, feats):
        q_size = feats.size()[0]
        if self.ptr + q_size > self.K:
            self.feats[-q_size:,:,:,:] = feats
         #   self.targets[-q_size:] = targets
            self.ptr = 0
        else:
            self.feats[self.ptr: self.ptr + q_size,:,:,:] = feats
            self.ptr += q_size
            
class XBM3:
    def __init__(self):
        self.K = 20
        self.feats = torch.zeros(self.K, 320, 24, 24).cuda()#
      #  self.targets = torch.zeros(self.K, dtype=torch.long).cuda()
        self.ptr = 0

    @property
    def is_full(self):
        return (self.feats[-1:,:,:,:].cpu().numpy() == 0).all()

    def get(self):
        if self.is_full:
            return self.feats[:self.ptr,:,:,:]
        else:
            return self.feats

    def enqueue_dequeue(self, feats):
        q_size = feats.size()[0]
        if self.ptr + q_size > self.K:
            self.feats[-q_size:,:,:,:] = feats
         #   self.targets[-q_size:] = targets
            self.ptr = 0
        else:
            self.feats[self.ptr: self.ptr + q_size,:,:,:] = feats
            self.ptr += q_size
            
class XBM4:
    def __init__(self):
        self.K = 20
        self.feats = torch.zeros(self.K, 1, 24, 24).cuda()
      #  self.targets = torch.zeros(self.K, dtype=torch.long).cuda()
        self.ptr = 0

    @property
    def is_full(self):
        return (self.feats[-1:,:,:,:].cpu().numpy() == 0).all()

    def get(self):
        if self.is_full:
            return self.feats[:self.ptr,:,:,:]
        else:
            return self.feats

    def enqueue_dequeue(self, feats):
        q_size = feats.size()[0]
        if self.ptr + q_size > self.K:
            self.feats[-q_size:,:,:,:] = feats
         #   self.targets[-q_size:] = targets
            self.ptr = 0
        else:
            self.feats[self.ptr: self.ptr + q_size,:,:,:] = feats
            self.ptr += q_size