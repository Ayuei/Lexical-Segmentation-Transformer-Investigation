from bert import modeling, optimization, tokenization
import random

def read_sentencepiece_vocab(filepath):
    voc = []
    with open(filepath, encoding='utf-8') as fi:
        for line in fi:
            voc.append(line.split("\t")[0])
    # skip the first <unk> token
    voc = voc[1:]
    return voc

MODEL_PREFIX="temp/tokenizer"
snt_vocab = read_sentencepiece_vocab("{}.vocab".format(MODEL_PREFIX))
print("Learnt vocab size: {}".format(len(snt_vocab)))
print("Sample tokens: {}".format(random.sample(snt_vocab, 10)))


def parse_sentencepiece_token(token):
    if token.startswith("▁"):
        return token[1:]
    else:
        return "##" + token


bert_vocab = list(map(parse_sentencepiece_token, snt_vocab))
ctrl_symbols = ["[PAD]","[UNK]","[CLS]","[SEP]","[MASK]"]
bert_vocab = ctrl_symbols + bert_vocab
VOC_SIZE=32000

bert_vocab += ["[UNUSED_{}]".format(i) for i in range(VOC_SIZE - len(bert_vocab))]
print(len(bert_vocab))

VOC_FNAME = "medline_vocab.txt" #@param {type:"string"}

with open(VOC_FNAME, "w") as fo:
  for token in bert_vocab:
    fo.write(token+"\n")

testcase="How should I treat polymenorrhea in a 14-year-old girl?"
testcase="Why did they cut out the expanding granuloma which was due to Coccidiomycosis immitis, rather than just treat it medically?"
bert_tokenizer = tokenization.FullTokenizer(VOC_FNAME)
print(bert_tokenizer.tokenize(testcase))

VOC_FNAME="./vocab_additional_exps/models/pretrained/uncased_L-12_H-768_A-12/vocab.txt"
bert_tokenizer = tokenization.FullTokenizer(VOC_FNAME)
print(bert_tokenizer.tokenize(testcase))

VOC_FNAME="./vocab_additional_exps/vocab_builder/new_medical_vocab.txt"
bert_tokenizer = tokenization.FullTokenizer(VOC_FNAME)
print(bert_tokenizer.tokenize(testcase))
