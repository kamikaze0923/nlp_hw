import torch
import spacy
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torchtext.data.metrics import bleu_score

##########################################################################################
# Utility Functions
##########################################################################################
def argmax_decoding(model, hidden, encoder_outputs, trg_field, attentions, max_len):
    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    attentions = attentions.clone()

    for i in range(max_len):

        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(hidden.device)

        with torch.no_grad():
            output, hidden, attention = model.decoder(trg_tensor, hidden, encoder_outputs)

        attentions[i] = attention.squeeze()

        pred_token = output.squeeze().argmax().item()

        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    return trg_tokens[1:], attentions[:len(trg_tokens)-1]


def beam_search_decoding(model, hidden, encoder_outputs, trg_field, attention_buffer, max_len, beam_size):
    all_beam_states = [
        DecodingState(
            init_sentence=[trg_field.init_token],
            init_score=0,
            init_gru_state=hidden,
            trg_field=trg_field,
            attention_buffer=attention_buffer
        )
    ]
    def flatten_list(l):
        if len(l) == 1:
            return l[0]
        else:
            return l[0] + flatten_list(l[1:])
    while not all([dec_state.finished_flag for dec_state in all_beam_states]):
        all_beam_states = [dec_state.expand_and_update(model, encoder_outputs, beam_size) for dec_state in all_beam_states]
        all_beam_states = flatten_list(all_beam_states)
        all_beam_states = sorted(all_beam_states, key=lambda s: getattr(s, 'score'), reverse=True)[:beam_size]
        # There is a better implementation here using priority queue
    trg_tokens = all_beam_states[0].sentence
    attentions = all_beam_states[0].attention_buffer
    return trg_tokens[1:], attentions[:len(trg_tokens)-1]

class DecodingState:

    def __init__(self, init_sentence, init_score, init_gru_state, trg_field, attention_buffer):
        self.sentence = init_sentence
        self.score = init_score
        self.gru_state = init_gru_state
        self.trg_field = trg_field

        self.attention_buffer = attention_buffer.clone()

    def expand_and_update(self, model, encoder_outputs, beam_size):
        """
        :param trg_field: used to map the string to target index
        :param model:
        :param encoder_outputs:
        :return: a list of 'DecodingState' used for sorting for the top candidates, try class method here
        """
        if self.finished_flag:
            return [self]
        else:
            trg_indexes = [self.trg_field.vocab.stoi[self.sentence[-1]]]
            with torch.no_grad():
                trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(self.gru_state.device)
                output, gru_state, attention = model.decoder(trg_tensor, self.gru_state, encoder_outputs)
                self.attention_buffer[len(self.sentence) - 1] = attention.squeeze()
                pred_token_top_k, pred_token_top_k_idx = torch.topk(output.squeeze().softmax(dim=-1), k=beam_size)
                pred_words = [self.trg_field.vocab.itos[token] for token in pred_token_top_k_idx]

            return [
                DecodingState(
                    self.sentence + [w], self.score + torch.log(pred_token_top_k[i]).item(),
                    gru_state, self.trg_field, self.attention_buffer
                ) for i, w in enumerate(pred_words)
            ]

    @property
    def finished_flag(self):
        return self.sentence[-1] == self.trg_field.eos_token or len(self.sentence) == self.attention_buffer.size()[0]
        # self.attention_buffer.size()[0] = max_len


def translate_sentence(sentence, src_field, trg_field, model, device, max_len = 50, beam_size = 0):

    model.eval()

    if isinstance(sentence, str):
        nlp = spacy.load('de')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]

    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)

    src_len = torch.LongTensor([len(src_indexes)])

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)

    attention_buffer = torch.zeros(max_len, 1, len(src_indexes)).to(device)
    if not beam_size:
        return argmax_decoding(model, hidden, encoder_outputs, trg_field, attention_buffer, max_len)
    else:
        return beam_search_decoding(model, hidden, encoder_outputs, trg_field, attention_buffer, max_len, beam_size)


def save_attention_plot(sentence, translation, attention, index):

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)

    attention = attention.squeeze(1).cpu().detach().numpy()

    cax = ax.matshow(attention, cmap='Greys_r', vmin=0, vmax=1)
    fig.colorbar(cax)

    ax.tick_params(labelsize=15)

    x_ticks = [''] + ['<sos>'] + [t.lower() for t in sentence] + ['<eos>']
    y_ticks = [''] + translation

    ax.set_xticklabels(x_ticks, rotation=45)
    ax.set_yticklabels(y_ticks)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.savefig("examples/"+str(index)+'_translation.png')



def calculate_bleu(data, src_field, trg_field, model, device, max_len = 50):

    trgs = []
    pred_trgs = []

    for datum in data:

        src = vars(datum)['src']
        trg = vars(datum)['trg']

        pred_trg, _ = translate_sentence(src, src_field, trg_field, model, device, max_len)

        #cut off <eos> token
        pred_trg = pred_trg[:-1]

        pred_trgs.append(pred_trg)
        trgs.append([trg])

    return bleu_score(pred_trgs, trgs)
