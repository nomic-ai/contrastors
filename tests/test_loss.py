import torch
from contrastors.loss import clip_loss, gte_loss


def test_clip_loss():
    query = torch.tensor([[1, 2], [2, 3], [3, 4]], dtype=torch.float32)
    query /= torch.norm(query, dim=1, keepdim=True)
    document = torch.tensor([[1, 2], [3, 4], [2, 3]], dtype=torch.float32)
    document /= torch.norm(document, dim=1, keepdim=True)

    loss = clip_loss(query, document)

    sim = torch.exp(query.matmul(document.T))
    softmax = sim / sim.sum(dim=1, keepdim=True)
    naive_loss = -torch.log(softmax[torch.arange(query.shape[0]), torch.arange(query.shape[0])]).mean()

    assert torch.allclose(loss, naive_loss)
