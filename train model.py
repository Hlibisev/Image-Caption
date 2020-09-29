from model import CNNtoRNN
from downloader import get_coco_loader
from tools import save_checkpoint, load_checkpoint, bert_cos_loss
import torch
from pytorch_pretrained_bert import BertModel
from torch import nn
import torch.optim as optim
from tqdm import tqdm

# creating data_loaders
data_loader, val_loader, dataset = get_coco_loader()

# working with model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_model = True
save_model = True
train_CNN = False

# hyperparameters
embed_size = 768
hidden_size = 768
vocab_size = len(dataset.token_to_id)
num_layers = 1
learning_rate = 0.0002
num_epochs = 30

# bert for loss
bert = BertModel.from_pretrained('bert-base-uncased').cuda()
bert.eval()

for p in bert.parameters():
    p.requires_grad = True

# losses
cos = nn.CosineSimilarity()
mse = nn.MSELoss()

# creating embedding mask
weight_to_token = torch.FloatTensor([dataset.id_to_token]).cuda().T

# initialize model and optimizer
model = CNNtoRNN(embed_size, hidden_size, vocab_size, weight_to_token).to(device)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
model.train()

losses = 0
CLS = (torch.zeros(32, device=device) + 101).unsqueeze(0).T

if load_model:
    load_checkpoint(torch.load("/content/drive/My Drive/Colab Notebooks/my_checkpoint4.pth.tar"), model, optimizer)

for epoch in range(num_epochs):
    # save model
    if save_model:
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": True,
        }
        save_checkpoint(checkpoint)

    for idx, (imgs, captions) in tqdm(
            enumerate(data_loader), total=len(data_loader), leave=False):

        # getting image and caption
        imgs = imgs.to(device)
        captions = captions.to(device)
        outputs = model(imgs)

        # prepearing data for bert
        outputs = torch.cat((CLS, outputs), dim=1)
        loss = bert_cos_loss(outputs.long().permute(1, 0), captions.long(), bert, mse)

        # step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # counting loss
        losses += loss.item()

        if idx % 100 == 0:
            print("Training loss", losses / 100)
            losses = 0

        if idx % 1000 == 0:
            if save_model:
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": True,
                }
                save_checkpoint(checkpoint)
