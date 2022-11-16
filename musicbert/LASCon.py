import torch
from torch import nn
import torch.nn.functional as F


class LASCon(nn.Module):
    def __init__(self, 
        temperature = 0.5,
        label_sim = 'tanh', 
        s_coef = 1,
        out_loss = True, 
        log_prediction = False, 
        threshold = None):
        
        super().__init__()
        self.temperature = temperature
        self.label_sim = label_sim
        self.out_loss = out_loss
        self.log_prediction = log_prediction
        self.threshold = threshold

        if label_sim == 'tanh':
            self.label_sim = LASCon.get_label_similarity_tanh(s_coef)
        elif label_sim == 'linear':
            self.label_sim = LASCon.label_similarity_linear
        elif label_sim == 'supcon':
            self.label_sim = LASCon.label_similarity_supcon
        elif label_sim == 'nt-xent':
            self.label_sim = LASCon.label_similarity_nt_xent

    def forward(self, zis, zjs, yis, yjs=None):
        """Calculates the contrastive loss. 
        Args:
            zis (Tensor): Tensor of embedding, z_i (batch_size x vector_dim).
            zjs (Tensor): Tensor of embedding, z_j (batch_size x vector_dim)
            yis (Tensor) : Tensor of z1's continuous label. 
            yjs (Tensor) : Tensor of z2's continuous label. 
            label_difference : Tensor of label difference between samples (batch_size x batch_size)

        Returns:
            Tensor: Contrastive loss (1)
        """
        if yjs == None: yjs=yis

        ## Calculate Representation Similarity
        z = torch.cat([zis, zjs], dim=0)
        n_sample = z.shape[0]

        z = F.normalize(z, p=2, dim=1)
        sim = torch.mm(z, z.t().contiguous())
        # for numerical stability
        sim_max, _ = torch.max(sim, dim=1, keepdim=True) 
        sim = sim - sim_max
        align = torch.exp(sim / self.temperature)


        # Uniformity
        mask = ~torch.eye(n_sample, device=sim.device).bool()
        uniformity = align.masked_select(mask).view(n_sample, -1).sum(dim=-1)


        ## Calculate Label Similarity: S(i, j)
        l = torch.cat([yis, yjs], dim=0)
        l.requires_grad_(False)
        if not self.threshold == None: l -= self.threshold
        if self.log_prediction: l = torch.log(l)
        s = self.label_sim(l)

        ## Calculate u(i,j)
        u = (align/uniformity).masked_select(mask).view(n_sample, -1)
        masked_s = s.masked_select(mask).view(n_sample, -1)

        ## Calculate Loss
        if self.out_loss:
            logu = torch.log(u)
            numerator = (masked_s * logu).sum(dim=1)
            denominator = masked_s.sum(dim=1)
            loss = - (numerator / denominator).mean()
        else:
            numerator = (masked_s * u).sum(dim=1)
            denominator = masked_s.sum(dim=1)
            loss = - torch.log((numerator / denominator).mean())

        return loss

    @staticmethod
    def label_similarity_linear(l):
        l = l.reshape(-1, 1)
        l = l / torch.max(torch.abs(l)) # Rescale labels to [-1, 1]
        return (1 - torch.abs(l - l.T)/2)

    @staticmethod
    def get_label_similarity_tanh(s_coef):
        def label_similarity_tanh(l):
            l = l.reshape(-1, 1)
            l = l / torch.max(torch.abs(l)) # Rescale labels to [-1, 1]
            return (1 - torch.tanh(s_coef * torch.abs(l - l.T)/2)/2)
        return label_similarity_tanh

    @staticmethod
    def label_similarity_supcon(l):
        n_samples = l.shape[0]
        lo = l.reshape(1, n_samples)
        lt = l.reshape(n_samples, 1)
        r = (lo - lt).abs()
        r[r <= 0.5] = 0
        r[r > 0.5] = 3
        r[r == 0] = 1
        r[r == 3] = 0.01
        return r

    @staticmethod
    def label_similarity_nt_xent(l):
        n_samples = l.shape[0] // 2
        eyes = torch.eye(n_samples)
        zeros = torch.zeros(n_samples, n_samples)
        left = torch.cat([zeros, eyes])
        right = torch.cat([eyes, zeros])
        r = torch.cat([left, right], dim=1)
        r = r.to(device=l.device)
        return r