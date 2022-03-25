import torch
import torch.nn.functional as F

class SkipGramModel(torch.nn.Module):
    def __init__(self, v_size, emb_dim):
        super().__init__()
        self.emb = torch.nn.Embedding(v_size, emb_dim)

    def forward(self, x):
        emb = self.emb(x)
        return emb


class NCELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, c_emb, o_emb, negatives):
        '''
        c_emb : torch.tensor (b, h)
        o_emb : torch.tensor (b, h)
        negatives : torch.tensor (b, k, h)
        '''
        pos = F.logsigmoid(torch.bmm(c_emb.unsqueeze(1), o_emb.unsqueeze(2)).squeeze())
        negs = F.logsigmoid(-torch.bmm(negatives, c_emb.unsqueeze(2)).squeeze()).sum(axis=1)
        loss = - (pos + negs).mean()
        return loss
