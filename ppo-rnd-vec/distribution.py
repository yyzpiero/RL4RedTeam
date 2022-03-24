from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence
import torch

class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[], device='cuda'):
        self.masks = masks
        self.device = device
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.masks = masks.type(torch.BoolTensor).to(device)
            logits = torch.where(self.masks, logits, torch.tensor(-1e+8).to(device))
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
    
    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.).to(self.device))
        return -p_log_p.sum(-1)

    def sample(self):
        action_probs = torch.nn.functional.softmax(self.logits, 1).to(self.device)
        return Categorical(action_probs).sample()

class Distributions():
    def sample(self, datas, mask= None, device="cuda"):
        distribution = Categorical(datas)
        #return distribution.sample().float().to(device)
        return distribution.sample().int().to(device)
        
    def entropy(self, datas, mask= None, device="cuda"):
        distribution = Categorical(datas)
        return distribution.entropy().float().to(device)

    def logprob(self, datas, value_data, mask= None, device="cuda"):
        distribution = Categorical(datas)
        return distribution.log_prob(value_data).unsqueeze(1).float().to(device)

    def kl_divergence(self, datas1, datas2, mask= None, device="cuda"):
        distribution1 = Categorical(datas1)
        distribution2 = Categorical(datas2)

        return kl_divergence(distribution1, distribution2).unsqueeze(1).float().to(device)

class DistributionsMasked():
    def sample(self, datas, mask, device="cuda"):
        distribution = CategoricalMasked(logits=datas, masks=mask)
        return distribution.sample().float().to(device)

    def entropy(self, datas, mask, device="cuda"):
        distribution = CategoricalMasked(logits=datas, masks=mask)
        return distribution.entropy().float().to(device)

    def logprob(self, datas, value_data, mask, device="cuda"):
        distribution = CategoricalMasked(logits=datas, masks=mask)
        return distribution.log_prob(value_data).unsqueeze(1).float().to(device)

    def kl_divergence(self, datas1, datas2, mask, device="cuda"):
        distribution1 = CategoricalMasked(logits=datas1, masks=mask)
        distribution2 = CategoricalMasked(logits=datas2, masks=mask)

        return kl_divergence(distribution1, distribution2).unsqueeze(1).float().to(device)