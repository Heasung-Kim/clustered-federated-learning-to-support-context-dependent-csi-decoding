import torch
import torch.nn as nn
import numpy as np
from model.neural_networks.vqcinet.encoder import Encoder
from model.neural_networks.vqcinet.decoder import Decoder
from model.neural_networks.vqcinet.quantizer import VectorQuantizer


class VQCINet128(nn.Module):
    def __init__(self):
        super(VQCINet128, self).__init__()

        self.embedding_dim = 4 #config["embedding_dim"]
        self.n_embeddings = 16 #config["n_embeddings"]
        self.n_codeword_bits = 128  # config["n_codeword_bits"]
        self.units = int(self.n_codeword_bits / np.log2(self.n_embeddings)) # Bits to Units conversion

        self.img_shape = (2,32,32)
        self.feature_encoder_output_shape = (12,8,8)
        self.feature_encoder_output_size = self.feature_encoder_output_shape[0] *\
                                            self.feature_encoder_output_shape[1] *\
                                            self.feature_encoder_output_shape[2]


        self.feature_encoder = Encoder(img_shape=self.img_shape)
        self.local_encoder_fc = nn.Linear(self.feature_encoder_output_size, self.units * self.embedding_dim)
        self.encoder_vector_quantization = VectorQuantizer(n_e=self.n_embeddings, e_dim=self.embedding_dim, beta=1.0)
        self.local_decoder_fc = nn.Linear(self.units * self.embedding_dim, self.feature_encoder_output_size)
        self.feature_decoder = Decoder(img_shape=self.img_shape)


    def forward(self, x):
        n, c, h, w = x.detach().size()

        z_e = self.feature_encoder(x)
        z_e = self.local_encoder_fc(z_e.view(n, -1))
        z_e = z_e.view(n, self.embedding_dim, self.units, 1)
        embedding_loss, z_q, perplexity, _, _ = self.encoder_vector_quantization(z_e)
        z_q = z_q.view(n,-1)
        #z_q = self.local_decoder_fc(z_q).view(n, c, h, w)
        z_q = self.local_decoder_fc(z_q).view(n, self.feature_encoder_output_shape[0], self.feature_encoder_output_shape[1], self.feature_encoder_output_shape[2])
        x_hat = self.feature_decoder(z_q)

        return embedding_loss, x_hat, z_q #perplexity


class VQCINet64(nn.Module):
    def __init__(self):
        super(VQCINet64, self).__init__()

        self.embedding_dim = 4 #config["embedding_dim"]
        self.n_embeddings = 16 #config["n_embeddings"]
        self.n_codeword_bits = 64  # config["n_codeword_bits"]
        self.units = int(self.n_codeword_bits / np.log2(self.n_embeddings)) # Bits to Units conversion

        self.img_shape = (2,32,32)
        self.feature_encoder_output_shape = (12,8,8)
        self.feature_encoder_output_size = self.feature_encoder_output_shape[0] *\
                                            self.feature_encoder_output_shape[1] *\
                                            self.feature_encoder_output_shape[2]


        self.feature_encoder = Encoder(img_shape=self.img_shape)
        self.local_encoder_fc = nn.Linear(self.feature_encoder_output_size, self.units * self.embedding_dim)
        self.encoder_vector_quantization = VectorQuantizer(n_e=self.n_embeddings, e_dim=self.embedding_dim, beta=1.0)
        self.local_decoder_fc = nn.Linear(self.units * self.embedding_dim, self.feature_encoder_output_size)
        self.feature_decoder = Decoder(img_shape=self.img_shape)


    def forward(self, x):
        n, c, h, w = x.detach().size()

        z_e = self.feature_encoder(x)
        z_e = self.local_encoder_fc(z_e.view(n, -1))
        z_e = z_e.view(n, self.embedding_dim, self.units, 1)
        embedding_loss, z_q, perplexity, _, _ = self.encoder_vector_quantization(z_e)
        z_q = z_q.view(n,-1)
        #z_q = self.local_decoder_fc(z_q).view(n, c, h, w)
        z_q = self.local_decoder_fc(z_q).view(n, self.feature_encoder_output_shape[0], self.feature_encoder_output_shape[1], self.feature_encoder_output_shape[2])
        x_hat = self.feature_decoder(z_q)

        return embedding_loss, x_hat, z_q #perplexity


class VQCINet256(nn.Module):
    def __init__(self):
        super(VQCINet256, self).__init__()

        self.embedding_dim = 4 #config["embedding_dim"]
        self.n_embeddings = 16 #config["n_embeddings"]
        self.n_codeword_bits = 256  # config["n_codeword_bits"]
        self.units = int(self.n_codeword_bits / np.log2(self.n_embeddings)) # Bits to Units conversion

        self.img_shape = (2,32,32)
        self.feature_encoder_output_shape = (12,8,8)
        self.feature_encoder_output_size = self.feature_encoder_output_shape[0] *\
                                            self.feature_encoder_output_shape[1] *\
                                            self.feature_encoder_output_shape[2]


        self.feature_encoder = Encoder(img_shape=self.img_shape)
        self.local_encoder_fc = nn.Linear(self.feature_encoder_output_size, self.units * self.embedding_dim)
        self.encoder_vector_quantization = VectorQuantizer(n_e=self.n_embeddings, e_dim=self.embedding_dim, beta=1.0)
        self.local_decoder_fc = nn.Linear(self.units * self.embedding_dim, self.feature_encoder_output_size)
        self.feature_decoder = Decoder(img_shape=self.img_shape)


    def forward(self, x):
        n, c, h, w = x.detach().size()

        z_e = self.feature_encoder(x)
        z_e = self.local_encoder_fc(z_e.view(n, -1))
        z_e = z_e.view(n, self.embedding_dim, self.units, 1)
        embedding_loss, z_q, perplexity, _, _ = self.encoder_vector_quantization(z_e)
        z_q = z_q.view(n,-1)
        #z_q = self.local_decoder_fc(z_q).view(n, c, h, w)
        z_q = self.local_decoder_fc(z_q).view(n, self.feature_encoder_output_shape[0], self.feature_encoder_output_shape[1], self.feature_encoder_output_shape[2])
        x_hat = self.feature_decoder(z_q)

        return embedding_loss, x_hat, z_q #perplexity

if __name__ == "__main__":
    # random data
    x = np.random.random_sample((200, 2, 32, 32))
    x = torch.tensor(x).float()
    config = {
        "img_shape" : (2,32,32),
        "n_codeword_bits" :32,
        "embedding_dim" :8,
        "n_embeddings":16
    }

    # test encoder
    vqcinet = VQCINet(config=config, device=torch.device("cpu"))
    embedding_loss, x_hat, perplexity = vqcinet(x)
    print('Encoder out shape:', x_hat.shape)
