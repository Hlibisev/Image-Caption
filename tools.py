import torch


def save_checkpoint(state, filename="/content/drive/My Drive/Colab Notebooks/my_checkpoint4.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


def get_attention_mask(texts):
    attention_mask = [[float(i > 0) for i in seq] for seq in texts]
    return torch.tensor(attention_mask, device="cuda")


def bert_cos_loss(y, target, bert, loss):
    y = y.permute(1, 0)
    _, embeds_y = bert(y, attention_mask=get_attention_mask(y))

    target = target.permute(1, 0)
    _, embeds_target = bert(target, attention_mask=get_attention_mask(target))

    return loss(embeds_y, embeds_target).mean()
