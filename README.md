# German T5 Tokenizer

This repo gives an overview of how to train a custom T5 tokenizer for our German model.

# Training corpus for German T5 tokenizer

We experiment with different corpora from GC4. The corpus used for training a tokenizer has huge impact on the downstream task model performance, as it can be seen in the ["How Good is Your Tokenizer?"](https://arxiv.org/abs/2012.15613) paper.

Thus, we calculate the so called subword fertility rate (number of subtokens / number of total tokens) for three downstream tasks: GermEval 2018 (Classification), GermEval 2014 (NER) and Universal Dependencies (PoS Tagging, Parsing).

To get an overview, we calculated the subword fertility rate for various German (cased) models:

| Model                                                                                      | Vocab Size | Approach  | GermEval 2018 | GermEval 2014 | UD HDT | Average
| ------------------------------------------------------------------------------------------ | ---------- | --------- | ------------- | ------------- | ------ | -------
| [GC4 ELECTRA](https://huggingface.co/stefan-it/electra-base-gc4-64k-0-cased-discriminator) |  64,000    | WordPiece | 1.4749        | 1.2063        | 1.2337 | 1.30
| [German BERT](https://huggingface.co/bert-base-german-cased)                               |  30,000    | WordPiece | 1.6326        | 1.3121        | 1.3852 | 1.44
| [DBMDZ German BERT](https://huggingface.co/bert-base-german-dbmdz-cased)                   |  31,102    | WordPiece | 1.5705        | 1.3004        | 1.35   | 1.41
| [GottBERT](https://huggingface.co/uklfr/gottbert-base)                                     |  52,009    | BPE       | 1.7806        | 1.3934        | 1.4172 | 1.53
| [mT5](https://huggingface.co/google/mt5-base)                                              | 250,112    | SPM       | 1.9149        | 1.7263        | 1.7545 | 1.80
| **Ours**                                                                                   |  32,000    | SPM       | 1.7079        | 1.3824        | 1.4097 | 1.50

We use the following packages from GC4 (filtered) to construct our vocabulary:

| Filename                            | Instances  | Tokens      | Size
| ----------------------------------- | ---------- | ----------- | ----
| `de_head_0000_2015-48_filtered.txt` |  6,823,262 | 230,285,046 | 1.7G
| `de_head_0000_2016-44_filtered.txt` |  1,305,750 |  70,525,146 | 509M
| `de_head_0004_2017-39_filtered.txt` |  1,585,725 |  51,027,153 | 364M
| `de_head_0007_2018-30_filtered.txt` |  1,321,017 |  42,280,783 | 302M
| `de_head_0007_2019-09_filtered.txt` |  2,798,132 |  91,992,353 | 654M
| `de_head_0007_2020-10_filtered.txt` |  1,204,554 |  37,558,448 | 270M
| Total                               | 15,038,440 | 523,668,929 | 3.7G

# De-Constructing original T5 Tokenizer

Before we can start training an own tokenizer, we need to de-construct the original T5 tokenizer.

The original tokenizer is sentencepiece-based as mentioned in the paper. So let's download the `spiece.model` from Hugging Face model hub via:

```bash
$ wget "https://huggingface.co/t5-base/resolve/main/spiece.model"
```

and inspect it:

```python
import sentencepiece as spm

vocab_file = "./spiece.model"
sp_model = spm.SentencePieceProcessor()
sp_model.Load(vocab_file)
```

This will load the original spm-model. Now let's have a look at the first ids and items in the vocab:

```python
for index in range(0,10):
    print(index, "->", sp_model.IdToPiece(index))
```

this outputs:

```bash
0 -> <pad>
1 -> </s>
2 -> <unk>
3 -> ▁
4 -> X
5 -> .
6 -> ,
7 -> s
8 -> ▁the
9 -> a
```

The first three ids are some kind of special symbols: `<pad>, `</s> and `</unk>` and used for padding or denoting an end of sentence. When constructing our own vocab, we need to make sure, that we use the same ids at the beginning.

# SPM training

Now we can train our own spm model. We use the `unigram` approach, because this was also used for building the ALBERT vocab (slightly mentioned in the [documentation](https://github.com/google-research/albert#sentencepiece)). We did not experiment with our algorithms.

Here's the training command:

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="vocab_first_attempt.txt",
    model_prefix="spiece",
    vocab_size=32000,
    unk_id=2,
    bos_id=-1,
    eos_id=1,
    pad_id=0,
    model_type="unigram",
    train_extremely_large_corpus=true,
)
```

We set the the ids `<pad_id>`, `<eos_id`> and `<unk_id>` according to the original T5 spm model and also use a vocab size of 32,000.

**Notice**: training can take ~2 hours. For our 3.7GB corpus it consumes ~170GB of RAM.
