import torch
from typing import Type
from torch import nn
from dataset import TextDataset


class LanguageModel(nn.Module):
    def __init__(self, dataset: TextDataset, embed_size: int = 256, hidden_size: int = 256,
                 rnn_type: Type = nn.RNN, rnn_layers: int = 1):
        """
        Model for text generation
        :param dataset: text data dataset (to extract vocab_size and max_length)
        :param embed_size: dimensionality of embeddings
        :param hidden_size: dimensionality of hidden state
        :param rnn_type: type of RNN layer (nn.RNN or nn.LSTM)
        :param rnn_layers: number of layers in RNN
        """
        super(LanguageModel, self).__init__()
        self.dataset = dataset
        self.vocab_size = dataset.vocab_size
        self.max_length = dataset.max_length

        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Create necessary layers
        """
        self.embedding = nn.Embedding(self.vocab_size, embed_size)
        self.rnn = rnn_type(embed_size, hidden_size, num_layers=rnn_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, self.vocab_size)
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, indices: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute forward pass through the model and
        return logits for the next token probabilities
        :param indices: LongTensor of encoded tokens of size (batch_size, length)
        :param lengths: LongTensor of lengths of size (batch_size, )
        :return: FloatTensor of logits of shape (batch_size, length, vocab_size)
        """
        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Convert indices to embeddings, pass them through recurrent layers
        and apply output linear layer to obtain the logits
        """
        # ИСПОЛЬЗОВАЛ LLM ТАК КАК НЕ ПОНИМАЛ КУДА ВНЕДРИТЬ ТУТ ЭТИ LENGTHS (тут моя идея сделать полностью дз без LLM рухнула :(
        inputs = self.embedding(indices)
        packed_inputs = nn.utils.rnn.pack_padded_sequence(inputs, lengths.cpu(), batch_first=True, enforce_sorted=False)
        if isinstance(self.rnn, nn.LSTM):
            h0 = torch.zeros(self.rnn_layers, indices.size(0), self.hidden_size).to(indices.device)
            c0 = torch.zeros(self.rnn_layers, indices.size(0), self.hidden_size).to(indices.device)
            initial_state = (h0, c0)
        else:
            initial_state = torch.zeros(self.rnn_layers, indices.size(0), self.hidden_size).to(indices.device)

        packed_out, _ = self.rnn(packed_inputs, initial_state)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        logits = self.linear(out)
        return logits

    @torch.inference_mode()
    def inference(self, prefix: str = '', temp: float = 1.) -> str:
        """
        Generate new text with an optional prefix
        :param prefix: prefix to start generation
        :param temp: sampling temperature
        :return: generated text
        """
        self.eval()
        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Encode the prefix (do not forget the BOS token!),
        pass it through the model to accumulate RNN hidden state and
        generate new tokens sequentially, sampling from categorical distribution,
        until EOS token or reaching self.max_length.
        Do not forget to divide predicted logits by temperature before sampling
        """
        prefix_tokens = [self.dataset.bos_id] + self.dataset.text2ids(prefix)
        generated_tokens = prefix_tokens.copy()

        for _ in range(self.max_length - len(generated_tokens)):
             input_tensor = torch.tensor([generated_tokens], dtype=torch.long).to(self.device)
             length_tensor = torch.tensor([len(generated_tokens)], dtype=torch.long).to(self.device)
             logits = self.forward(input_tensor, length_tensor)
             next_token_logits = logits[0, -1] / temp
             next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1).item()
             if next_token == self.dataset.eos_id:
                 break
             generated_tokens.append(next_token)


        if generated_tokens[-1] == self.dataset.eos_id:
            generated_tokens = generated_tokens[1:-1]
        else:
            generated_tokens = generated_tokens[1:]

        result_tokens = generated_tokens
        generated_text = self.dataset.ids2text(result_tokens)
        return generated_text
