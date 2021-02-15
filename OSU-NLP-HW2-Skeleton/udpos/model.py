import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from udpos.dataset import EOS_VALUE
from udpos.utils import one_hot_vector

class POS_from_WordSeq(torch.nn.Module):
    """
    an encoder-decoder architecture to transform the word sequence to the part-of-speech label
    """

    def __init__(self, args, word_embedding_layer, tag_embedding_layer):
        super().__init__()
        self.encoder = Bidirectional_LSTM_Encoder(
            word_embedding_layer, word_embedding_layer.layer_matrix.size()[1],
            hidden_size=args.hidden_dim, lstm_layers=args.lstm_layers
        )
        self.decoder = Bidirectional_LSTM_Cell_Decoder(
            tag_embedding_layer,
            input_size=tag_embedding_layer.layer_matrix.size()[1],
            hidden_size=self.encoder.hidden_size,
            lstm_layers=self.encoder.lstm_layers
        )

    def forward(self, batch):
        xx_pad, xx_pad_rev, yy_pad, x_lens = batch
        h_t, h_t_rev = self.encoder(xx_pad, xx_pad_rev, x_lens)
        return self.decoder(yy_pad, h_t, h_t_rev, x_lens)

class Bidirectional_LSTM_Encoder(torch.nn.Module):
    """
    input: B x L x ED (a batch of sequence of maximum length of the word embedding(including the reversed sequence))
    output: 2 X B x ED (last hidden state of the 2 lstm(including the one processing reversed sequence))
    """

    def __init__(self, word_embedding_layer, input_size, hidden_size, lstm_layers):
        super().__init__()

        self.word_embedding_layer = word_embedding_layer
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers

        # a single bidirectional LSTM is not worked for a one-sided padded sequence
        self.lstm_1 = torch.nn.LSTM(
            input_size=self.input_size, bidirectional=False,
            hidden_size=self.hidden_size, batch_first=True, num_layers=self.lstm_layers
        )
        self.lstm_2 = torch.nn.LSTM(
            input_size=self.input_size, bidirectional=False,
            hidden_size=self.hidden_size, batch_first=True, num_layers=self.lstm_layers
        )

        self.h_0 = torch.nn.Parameter(torch.zeros(size=(self.lstm_layers, self.hidden_size)))
        self.c_0 = torch.nn.Parameter(torch.zeros(size=(self.lstm_layers, self.hidden_size)))
        self.h_0_rev = torch.nn.Parameter(torch.zeros(size=(self.lstm_layers, self.hidden_size)))
        self.c_0_rev = torch.nn.Parameter(torch.zeros(size=(self.lstm_layers, self.hidden_size)))
        self.sanity_check_done_flag = False

    def forward(self, xx_emb, xx_rev_emb, x_lens):
        xx_emb = self.word_embedding_layer(xx_emb)
        xx_rev_emb = self.word_embedding_layer(xx_rev_emb)
        batch_size = len(x_lens)

        h_0 = self.h_0.unsqueeze(1).expand(-1, batch_size, -1) # add batch dimension and repeat, L x B x HD
        c_0 = self.c_0.unsqueeze(1).expand(-1, batch_size, -1) # add batch dimension and repeat, L x B x HD
        h_0_rev = self.h_0_rev.unsqueeze(1).expand(-1, batch_size, -1) # add batch dimension and repeat, L x B x HD
        c_0_rev = self.c_0_rev.unsqueeze(1).expand(-1, batch_size, -1) # add batch dimension and repeat, L x B x HD

        x_pack = pack_padded_sequence(xx_emb, x_lens.to('cpu'), batch_first=True, enforce_sorted=False)
        # in pack_padded_sequence, the x_lens needs to be on cpu
        x_rev_pack = pack_padded_sequence(xx_rev_emb, x_lens.to('cpu'), batch_first=True, enforce_sorted=False)
        x_out, (h_t, _) = self.lstm_1(x_pack, (h_0, c_0))
        x_out_rev, (h_t_rev, _) = self.lstm_2(x_rev_pack, (h_0_rev, c_0_rev))

        if not self.sanity_check_done_flag:
            # do a sanity check here, make sure the x_lens is used for stop the input to update the hidden state
            x_out, _ = pad_packed_sequence(x_out, batch_first=True)
            x_out_rev, _ = pad_packed_sequence(x_out_rev, batch_first=True)
            x_ret = torch.cat([x_out, x_out_rev], dim=-1)

            time_idx = (x_lens-1)
            last_hidden_state_1 = torch.diagonal(
                torch.index_select(x_ret, dim=1, index=time_idx), dim1=0, dim2=1
            ).transpose(1,0)

            h_t_ret = torch.cat([torch.stack([a_row, b_row]) for a_row, b_row in zip(h_t, h_t_rev)])
            last_hidden_state_0 = h_t_ret[-2:]  # take last 2 layers
            last_hidden_state_0 = last_hidden_state_0.permute(1,0,2)
            last_hidden_state_0 = last_hidden_state_0.reshape(batch_size, -1)

            flag = (last_hidden_state_0 - last_hidden_state_1).to(dtype=torch.uint8)
            assert torch.any(flag) == False
            self.sanity_check_done_flag = True

        return h_t, h_t_rev

class Bidirectional_LSTM_Cell_Decoder(torch.nn.Module):
    """
    input: B x L x H (the output of the bidirectional_lstm_cell_encoder)
    output: B x L x TD (the probability of each tag label, the last dimension sum to 1)
    """

    def __init__(self, tag_embedding_layer, input_size, hidden_size, lstm_layers):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.tag_embedding_layer = tag_embedding_layer

        self.lstm_cell = Bidirectional_Multilayer_LSTM_Cells(
            input_size=self.input_size, hidden_size=self.hidden_size, lstm_layers=self.lstm_layers
        )
        self.tag_output_fc = torch.nn.Linear(in_features=2*self.hidden_size, out_features=self.input_size)
        self.c_0 = torch.nn.Parameter(torch.zeros(size=(self.lstm_layers, self.hidden_size)))
        self.c_0_rev = torch.nn.Parameter(torch.zeros(size=(self.lstm_layers, self.hidden_size)))


    def forward(self, yy_pad, h_t, h_t_rev, x_lens):
        """
        call the cell forward for n times, n = max(x_lens) + 1
        :param yy_pad: used for get the sos token
        :param h_t: the hidden state taken from the encoder
        :param h_t_rev: the hidden state taken from the decoder
        :param x_lens: different length of the sequence in this batch
        :return: a stacking tensor for each step's probability distribution over all speech tag labels B x L x TD
        """
        batch_size = len(x_lens)
        sos = yy_pad[:,[0]]# B x L
        c_0 = self.c_0.unsqueeze(1).expand(-1, batch_size, -1) # add batch dimension and repeat, L x B x HD
        c_0_rev = self.c_0_rev.unsqueeze(1).expand(-1, batch_size, -1) # add batch dimension and repeat, L x B x HD

        feedin_syb = sos
        decode_contents = []
        end_sentence = set()

        for _ in range(max(x_lens) + 1): # expect one more token on the <EOS> at the end
            feedin_emb = self.tag_embedding_layer(feedin_syb)
            h_t, h_t_rev, c_0, c_0_rev = self.lstm_cell(feedin_emb[:,0,:], h_t, h_t_rev, c_0, c_0_rev)
            word_hidden = torch.cat([h_t[-1], h_t_rev[-1]], dim=-1).permute(1, 0).reshape(batch_size, -1)
            word_logits = self.tag_output_fc(word_hidden)

            for i in end_sentence: # use a new vector to stop the gradient from backpropagating here
                word_logits[i, :] = one_hot_vector(self.input_size, EOS_VALUE)

            word_label = torch.argmax(word_logits, dim=1, keepdim=True) # B x (ED x 2) -> B (in categories)
            eq_flag = torch.eq(word_label, 1) # contruct 1 tensor on same device
            if torch.any(eq_flag):
                for i, _ in zip(*torch.where(eq_flag)): # eq_flag is in B x 1
                    end_sentence.add(i.item())

            decode_contents.append(word_logits)
            feedin_syb = word_label

        return torch.stack(decode_contents, dim=1).softmax(dim=-1)



class Bidirectional_Multilayer_LSTM_Cells(torch.nn.Module):
    """
    Get called by the decoder, decoding the hidden state for variable time steps depending on the max length in a sequence
    """

    def __init__(self, input_size, hidden_size, lstm_layers):
        super().__init__()
        self.lstm_cells = torch.nn.ModuleList()
        self.lstm_cells_rev = torch.nn.ModuleList()

        lstm_layer_input_size_sizes = [input_size] + [hidden_size for _ in range(lstm_layers-1)]
        assert len(lstm_layer_input_size_sizes) == lstm_layers

        for lstm_layer_input_size in lstm_layer_input_size_sizes:
            self.lstm_cells.append(torch.nn.LSTMCell(input_size=lstm_layer_input_size, hidden_size=hidden_size))
            self.lstm_cells_rev.append(torch.nn.LSTMCell(input_size=lstm_layer_input_size, hidden_size=hidden_size))

    def forward(self, sos, h_t, h_t_rev, c_t, c_t_rev):
        """
        one time step decoding
        :param sos: the sos embedding taken from the embedding matrix, this is fixed
        :param h_t: previous hidden state
        :param h_t_rev: previous hidden state of the reversed sequence
        :param c_t: previous memory state
        :param c_t_rev: previous memory state of the reversed sequence
        :return: a stack of all hidden states and memory states, depending on how many layers the lstm has
        """
        feedin = sos
        feedin_rev = sos
        h_t_out_collector, h_t_out_rev_collector = [], []
        c_t_out_collector, c_t_out_rev_collector = [], []
        for cell, cell_rev, one_h_t, one_h_t_rev, one_c_t, one_c_t_rev in zip(
                self.lstm_cells, self.lstm_cells_rev, h_t, h_t_rev, c_t, c_t_rev,
        ): # similuate multilayer feed-in for one single step and return the output
            h_t_out, c_t_out = cell(feedin, (one_h_t, one_c_t))
            h_t_out_rev, c_t_out_rev = cell_rev(feedin_rev, (one_h_t_rev, one_c_t_rev))

            h_t_out_collector.append(h_t_out)
            c_t_out_collector.append(c_t_out)
            h_t_out_rev_collector.append(h_t_out_rev)
            c_t_out_rev_collector.append(c_t_out_rev)

            feedin = h_t_out
            feedin_rev = h_t_out_rev

        h_t_out_collector, h_t_out_rev_collector = torch.stack(h_t_out_collector), torch.stack(h_t_out_rev_collector)
        c_t_out_collector, c_t_out_rev_collector = torch.stack(c_t_out_collector), torch.stack(c_t_out_rev_collector)

        return h_t_out_collector, h_t_out_rev_collector, c_t_out_collector, c_t_out_rev_collector