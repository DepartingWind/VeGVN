import torch
from transformers import BertTokenizer, BertModel , RobertaTokenizer,RobertaModel
from sklearn.metrics.pairwise import cosine_similarity


def getSentEmb(sentence, tokenizer, model,device):
    max_size = 512
    marked_sentence = "[CLS] " + sentence + " [SEP]"
    tokenized_sentence = tokenizer.tokenize(marked_sentence)
    tokenized_sentence = tokenized_sentence[:max_size] if len(tokenized_sentence)>max_size else tokenized_sentence

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_sentence)

    segments_ids = [1] * len(tokenized_sentence)

    tokens_tensor = torch.tensor([indexed_tokens]).to(device)
    segments_tensors = torch.tensor([segments_ids]).to(device)
    model.to(device)
    model.eval()

    outputs = model(tokens_tensor, segments_tensors)

    batch_i = 0
    token_embeddings = []
    encoded_layers = outputs.hidden_states
    for token_i in range(len(tokenized_sentence)):
        hidden_layers = []
        for layer_i in range(len(encoded_layers)):
            vec = encoded_layers[layer_i][batch_i][token_i]
            hidden_layers.append(vec)
        token_embeddings.append(hidden_layers)

    sentence_embedding = torch.mean(encoded_layers[11], 1)

    return sentence_embedding


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    model = BertModel.from_pretrained('bert-base-uncased',output_hidden_states=True)

    print(model)
    sentence1 = "I like apple's computers"
    sentence2 = "I like eat apples"

    em1 = getSentEmb(sentence1, tokenizer, model,"cuda:0")
    em2 = getSentEmb(sentence2, tokenizer, model,"cuda:0")
    print("“祖国”有关句子和“天气”有关句子相似度1 =", cosine_similarity(em1.cpu().detach().numpy(), em2.cpu().detach().numpy()))