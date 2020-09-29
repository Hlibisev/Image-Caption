from torch import nn
import torchvision
import torch


class EncoderCNN(nn.Module):
    """
    Encoder.
    """
    def __init__(self, encoded_image_size=3):
        super(EncoderCNN, self).__init__()
        self.enc_image_size = encoded_image_size

        # pretrained ImageNet ResNet-101
        resnet = torchvision.models.resnet101(pretrained=True)

        # Remove linear and pool layers
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.view(-1, 2048 * self.enc_image_size * self.enc_image_size)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False

        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, encoder_dim, weight):
        super(DecoderRNN, self).__init__()
        self.vocab_size = vocab_size

        self.decode_step = nn.LSTMCell(embed_size, hidden_size)
        self.decode_step2 = nn.LSTMCell(hidden_size, hidden_size)

        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=-1)

        self.init_h = nn.Linear(encoder_dim, hidden_size)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, hidden_size)  # linear layer to find initial cell state of LSTMCell

        self.ln_h = nn.LayerNorm(hidden_size)
        self.ln_c = nn.LayerNorm(hidden_size)

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(2048 * 3 * 3, hidden_size)

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.totoken = nn.Embedding.from_pretrained(weight)

    def forward(self, features, max_length=15):

        # Extract features from image
        features = self.relu(self.fc1(features))

        # Init hidden states
        h = self.dropout(self.ln_h(self.relu(self.init_h(features))))
        c = self.dropout(self.ln_c(self.relu(self.init_c(features))))

        # First step for decoder
        h1, c1 = self.decode_step(features, (h, c))
        h2, c2 = self.decode_step2(h1, (h, c))
        h3, c3 = self.decode_step2(h2, (h, c))

        # from h3 to (batch_size, number_vocab)
        prediction = self.fc(self.dropout(h3))

        # init matrix of word
        res_caption = torch.zeros(h.shape[0], max_length).cuda()

        # get more probable word
        res_caption[:, 0] = self.totoken(prediction.argmax(1).long())[0]
        word = self.dropout(self.embed(prediction.argmax(1)))

        for t in range(1, max_length): # Should use better the stop criterion

            h1, c1 = self.decode_step(word, (self.ln_h(h1) + h, self.ln_c(c1) + c))
            h2, c2 = self.decode_step2(h1, (h2, c2))
            h3, c3 = self.decode_step2(h2, (h3, c3))

            prediction = self.fc(self.dropout(h3))
            res_caption[:, t] = self.totoken(prediction.argmax(1).long())[0]

            word = self.dropout(self.embed(prediction.argmax(1)))

        return res_caption


class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, weight_for_token):
        super(CNNtoRNN, self).__init__()
        self.hidden_size = hidden_size
        self.encoderCNN = EncoderCNN()
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, hidden_size, weight_for_token)

    def forward(self, images):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features)
        return outputs
