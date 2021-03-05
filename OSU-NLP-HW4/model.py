import torch
import torch.nn as nn
import torch.nn.functional as F


##########################################################################################
# Task 2.1
##########################################################################################

class SingleQueryScaledDotProductAttention(nn.Module):

    # kq_dim  is the  dimension  of keys  and  values. Linear  layers  should  be usedto  project  inputs  to these  dimensions.
    def __init__(self, enc_hid_dim, dec_hid_dim, kq_dim=512):
        super().__init__()
        self.query = nn.Linear(in_features=dec_hid_dim, out_features=kq_dim)
        self.key = nn.Linear(in_features=enc_hid_dim * 2, out_features=kq_dim)
        self.value = nn.Linear(in_features=enc_hid_dim * 2, out_features=kq_dim)

    #hidden  is h_t^{d} from Eq. (11)  and has  dim => [batch_size , dec_hid_dim]
    #encoder_outputs  is the  word  representations  from Eq. (6)
    # and has dim => [src_len , batch_size , enc_hid_dim * 2]
    def forward(self, hidden, encoder_outputs):
        value = self.transpose_for_scores(encoder_outputs) # B x T x Enc_hidden*2
        key = self.transpose_for_scores(self.key(encoder_outputs)) # B x T x Kq_dim
        query = self.transpose_for_scores(self.query(hidden).unsqueeze(0)) # B x 1 x Kq_dim
        alpha = torch.matmul(query, key.transpose(-1, -2)).softmax(dim=-1) # B x 1 x T
        attended_val = torch.matmul(alpha, value).squeeze(1) # B x Enc_hidden*2, cannot squeeze the batch dimension
        alpha = alpha.squeeze(1) # B x T, cannot squeeze the batch dimension
        assert alpha.shape == (hidden.shape[0], encoder_outputs.shape[0]) # Batch x Time
        assert attended_val.shape == (hidden.shape[0], encoder_outputs.shape[2]) # Batch x Enc_hidden*2
        return attended_val, alpha

    def transpose_for_scores(self, x):
        return x.permute(1, 0, 2) # Batch x Time x Kq_dim


##########################################################################################
# Model Definitions
##########################################################################################

class Dummy(nn.Module):

    def __init__(self, dev):
        super().__init__()
        self.dev = dev

    def forward(self, hidden, encoder_outputs):
        zout = torch.zeros( (hidden.shape[0], encoder_outputs.shape[2]) ).to(self.dev)
        zatt = torch.zeros( (hidden.shape[0], encoder_outputs.shape[0]) ).to(self.dev)
        return zout, zatt

class MeanPool(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, hidden, encoder_outputs):

        output = torch.mean(encoder_outputs, dim=0, keepdim=True).squeeze(0)
        alpha = F.softmax(torch.ones(hidden.shape[0], encoder_outputs.shape[0]), dim=0)

        return output, alpha

class BidirectionalEncoder(nn.Module):
    def __init__(self, src_vocab, emb_dim, enc_hid_dim, dec_hid_dim, dropout=0.5):
        super().__init__()

        self.enc_hidden_dim = enc_hid_dim
        self.emb = nn.Embedding(src_vocab, emb_dim)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        """
        :param src: L_en x B
        :return: (enc_hidden_states, sent)
        enc_hidden_states:
              encoded hidden states across all time steps: T x B x enc_hidden_dim*2
        sent:
              taken last hidden state of bidirectional GRU and pass through a fc layer
              used as the initial decoder hidden state: B x dec_hidden_state
        """

        # embed source tokens
        embedded = self.dropout(self.emb(src))

        # process with bidirectional GRU model
        enc_hidden_states, _ = self.rnn(embedded)

        # compute a global sentence representation to feed as the initial hidden state of the decoder
        # concatenate the forward GRU's representation after the last word and
        # the backward GRU's representation after the first word

        last_forward = enc_hidden_states[-1, :, :self.enc_hidden_dim]
        first_backward = enc_hidden_states[0, :, self.enc_hidden_dim:]

        # transform to the size of the decoder hidden state with a fully-connected layer
        sent = F.relu(self.fc(torch.cat((last_forward, first_backward), dim = 1)))

        return enc_hidden_states, sent


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, attention, dropout=0.5,):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.GRU(emb_dim, dec_hid_dim)

        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, encoder_sent_to_be_update, encoder_outputs):
        #Embed input
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))

        #Step decoder model forward
        output, encoder_sent_to_be_update = self.rnn(embedded, encoder_sent_to_be_update.unsqueeze(0))

        #Perform attention operation
        attended_feature, a = self.attention(encoder_sent_to_be_update.squeeze(0), encoder_outputs)

        #Make prediction
        prediction = self.fc_out(torch.cat((output, attended_feature.unsqueeze(0)), dim = 2))

        #Output prediction (scores for each word), the updated hidden state, and the attention map (for visualization)
        return prediction, encoder_sent_to_be_update.squeeze(0), a


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg):
        """
        :param src: L_de x B
        :param trg: L_en x B
        :return:
        """

        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        #encoder_outputs is all hidden states of the input sequence, back and forwards
        #hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, encoder_sent_to_be_update = self.encoder(src)


        for t in range(1, trg_len):

            # Step decoder model forward, getting output prediction, updated hidden, and attention distribution
            output, encoder_sent_to_be_update, a = self.decoder(trg[t-1], encoder_sent_to_be_update, encoder_outputs)

            #place predictions in a tensor holding predictions for each token
            outputs[t] = output

        return outputs