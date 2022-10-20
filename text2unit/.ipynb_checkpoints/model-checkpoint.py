# transformer encoder decoder from cs447
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoder(nn.Module):
    def __init__(self, src_vocab_size, embedding_dim, num_heads,
        num_layers, dim_feedforward, max_len_src, dropout_rate):
        super(TransformerEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        """
        Args:
            src_vocab_size: int, the size of the source vocabulary
            embedding_dim: the dimension of the embedding (also the number of expected features for the input of the Transformer)
            num_heads: The number of features in the GRU hidden state
            num_layers: the number of Transformer Encoder layers
            dim_feedforward: the dimension of the feedforward network models in the Transformer
            max_len_src: maximum length of the source sentences
            device: the working device (you may need to map your postional embedding to this device)
        """

        ### TODO ###        
        self.position_embedding = self.create_positional_embedding(max_len_src, embedding_dim)
        self.register_buffer('positional_embedding', self.position_embedding)

        # Initialize embedding layer with pretrained_emb
        self.embedding = nn.Embedding(src_vocab_size, embedding_dim, padding_idx=0)

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout_rate)

        # Initialize a nn.TransformerEncoder model (you'll need to use embedding_dim,
        # num_layers, num_heads, & dim_feedforward here)
        self.transformerEncoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(embedding_dim, num_heads, dim_feedforward), num_layers)

    def make_src_mask(self, src):
        src_mask = src.transpose(0, 1) == 0 # padding idx
        return src_mask # (batch_size, max_src_len)
    
    def create_positional_embedding(self, max_len, embed_dim):
        '''
        Args:
            max_len: The maximum length supported for positional embeddings
            embed_dim: The size of your embeddings
        Returns:
            pe: [max_len, 1, embed_dim] computed as in the formulae above
        '''

        ### TODO ###
        pe = torch.zeros(max_len, embed_dim)

        for pos in range(max_len):
            for i in range(embed_dim):
                if (i % 2) == 0:
                    pe[pos][i] = math.sin(pos/(10000**(i/embed_dim)))
                else:
                    pe[pos][i] = math.cos(pos/(10000**((i-1)/embed_dim)))

        return pe.unsqueeze(1)

    def forward(self, x):
        """
        Args:
            x: [max_len, batch_size]
        Returns:
            output: [max_len, batch_size, embed_dim]
        Pseudo-code:
        - Pass x through the word embedding
        - Add positional embedding to the word embedding, then apply dropout
        - Call make_src_mask(x) to compute a mask: this tells us which indexes in x
          are padding, which we want to ignore for the self-attention
        - Call the encoder, with src_key_padding_mask = src_mask
        """

        ### TODO ###
        
        src_mask = self.make_src_mask(x) 
        x = self.embedding(x)
        x = x + (self.position_embedding[:x.shape[0],:].to(x.device))
        x = self.dropout(x)
        output = self.transformerEncoder(src=x, src_key_padding_mask=src_mask)

        return output

class TransformerDecoder(nn.Module):
    def __init__(self, trg_vocab_size, embedding_dim, num_heads,
        num_layers, dim_feedforward, max_len_trg, dropout_rate):
        super(TransformerDecoder, self).__init__()
        self.embedding_dim = embedding_dim
        """
        Args:
            trg_vocab_size: int, the size of the target vocabulary
            embedding_dim: the dimension of the embedding (also the number of expected features for the input of the Transformer)
            num_heads: The number of features in the GRU hidden state
            num_layers: the number of Transformer Decoder layers
            dim_feedforward: the dimension of the feedforward network models in the Transformer
            max_len_trg: maximum length of the target sentences
            device: the working device (you may need to map your postional embedding to this device)
        """

        ### TODO ###
        self.position_embedding = self.create_positional_embedding(max_len_trg, embedding_dim)
        self.register_buffer('positional_embedding', self.position_embedding)
        
        self.embedding = nn.Embedding(trg_vocab_size, embedding_dim, padding_idx=0)
        #self.embedding.weight.data.normal_(0, 0.3)

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout_rate)

        # Initialize a nn.TransformerDecoder model (you'll need to use embedding_dim,
        # num_layers, num_heads, & dim_feedforward here)

        self.transformerDecoder = nn.TransformerDecoder(torch.nn.TransformerDecoderLayer(embedding_dim, num_heads, dim_feedforward, dropout=0.1), num_layers)


        # Final fully connected layer
        self.fc = nn.Linear(embedding_dim, trg_vocab_size)

    def generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def create_positional_embedding(self, max_len, embed_dim):
        '''
        Args:
            max_len: The maximum length supported for positional embeddings
            embed_dim: The size of your embeddings
        Returns:
            pe: [max_len, 1, embed_dim] computed as in the formulae above
        '''

        ### TODO ###
        pe = torch.zeros(max_len, embed_dim)

        for pos in range(max_len):
            for i in range(embed_dim):
                if (i % 2) == 0:
                    pe[pos][i] = math.sin(pos/(10000**(i/embed_dim)))
                else:
                    pe[pos][i] = math.cos(pos/(10000**((i-1)/embed_dim)))

        return pe.unsqueeze(1)

    def forward(self, dec_in, enc_out):
        """
        Args:
            dec_in: [sequence length, batch_size]
            enc_out: [max_len, batch_size, embed_dim]
        Returns:
            output: [sequence length, batch_size, trg_vocab_size]
        Pseudo-code:
        - Compute input word and positional embeddings in similar manner to encoder
        - Call generate_square_subsequent_mask() to compute a mask: this time,
          the mask is to prevent the decoder from attending to tokens in the "future".
          In other words, at time step i, the decoder should only attend to tokens
          1 to i-1.
        - Call the decoder, with trg_mask = trg_mask
        - Run the output through the fully-connected layer and return it
        """

        ### TODO ###
        max_len_trg = dec_in.shape[0]
        
        trg_mask = self.generate_square_subsequent_mask(dec_in.shape[0])
        dec_in = self.embedding(dec_in)
        dec_in = dec_in + (self.position_embedding[:dec_in.shape[0],:].to(dec_in.device))
        dec_in = self.dropout(dec_in)
        output = self.transformerDecoder(dec_in, enc_out, trg_mask.to(dec_in.device))
        output = self.fc(output)

        return output    

