import flair
from flair.embeddings import FlairEmbeddings, StackedEmbeddings
import torch
import numpy as np

from konvens2020_summarization import data_classes


def _default_embeddings():
    return StackedEmbeddings([FlairEmbeddings('de-forward'), FlairEmbeddings('de-backward')])


def _get_logprobs(
        text: str,
        lm: flair.models.LanguageModel = FlairEmbeddings('de-forward').lm,
        prefix: str = "\n",
        break_on_suffix=None) -> float:
    """
    Based on LanguageModel.generate_text() in flair.models
    """
    if prefix == "":
        prefix = "\n"
    if not lm.is_forward_lm:
        text = text[::-1]
        assert len(prefix) == 1

    with torch.no_grad():
        characters = []
        # initial hidden state
        hidden = lm.init_hidden(1)
        # pass prefix through language model to get initial hidden state
        if len(prefix) > 1:
            char_tensors = []
            for character in prefix[:-1]:
                char_tensors.append(
                    torch.tensor(lm.dictionary.get_idx_for_item(character))
                        .unsqueeze(0)
                        .unsqueeze(0)
                )
            input = torch.cat(char_tensors).to(flair.device)
            prediction, _, hidden = lm.forward(input, hidden)
        input = (
            torch.tensor(lm.dictionary.get_idx_for_item(prefix[-1])).unsqueeze(0).unsqueeze(0)
        )
        log_prob = 0.0

        for i, char in enumerate(text):
            input = input.to(flair.device)

            # get predicted weights
            prediction, _, hidden = lm.forward(input, hidden)
            prediction = prediction.squeeze().detach()
            word_idx = lm.dictionary.get_idx_for_item(char)
            prob = prediction[word_idx]
            log_prob += prob
            input = torch.tensor(word_idx).type(torch.long).detach().unsqueeze(0).unsqueeze(0)
            characters.append(char)

            if break_on_suffix is not None:
                if text[:i + 1].endswith(break_on_suffix):
                    text = text[:i + 1]
                    break

        log_prob = log_prob.item()
        log_prob /= len(text)
        return log_prob


def predict_flair_grammar_score(corpus: data_classes.Corpus,
                                name: str = 'flair_grammar',
                                embeddings: flair.embeddings.Embeddings = _default_embeddings()) -> data_classes.Corpus:
    def evaluate(text: str, embeddings: flair.embeddings.Embeddings) -> float:
        if isinstance(embeddings, flair.embeddings.StackedEmbeddings):
            scores = []
            for embedding in embeddings.embeddings:
                scores.append(_get_logprobs(text, lm=embedding.lm))
            return np.mean(scores).item()
        return _get_logprobs(text, lm=embeddings.lm)

    for doc in corpus:
        for summary in doc.gen_summaries:
            summary.predicted_scores[name] = evaluate(text=summary.text, embeddings=embeddings)

    return corpus


def main():
    text_correct = 'Das ist ein kleiner Test. Dieser Text sollte eine hohe Wahrscheinlichkeit generieren.'
    text_incorrect = 'Das ist ein kliener Test. dieser Text sollte eine kleines Wahrscheinlichkiet gener.'

    # Test forward language model
    print('Forward')
    lm = flair.embeddings.FlairEmbeddings('de-forward').lm
    print(f'Text1: "{text_correct}"')
    print(f'logprobs: {_get_logprobs(text_correct, lm)}')
    print('\n')
    print(f'Text2: "{text_incorrect}"')
    print(f'logprobs: {_get_logprobs(text_incorrect, lm)}')
    print('\n'*3)

    # Test backward language model
    print('Backward')
    lm = flair.embeddings.FlairEmbeddings('de-backward').lm
    print(f'Text1: "{text_correct}"')
    print(f'logprobs: {_get_logprobs(text_correct, lm)}')
    print(f'Text2: "{text_incorrect}"')
    print(f'logprobs: {_get_logprobs(text_incorrect, lm)}')
    print('\n' * 3)


if __name__ == '__main__':
    main()
