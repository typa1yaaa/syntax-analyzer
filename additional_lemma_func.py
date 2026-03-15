import json
import tensorflow as tf
from tensorflow import keras
import re
import numpy as np

def load_model_and_vocab(NAME_VOCAB, NAME_MODEL):

    with open(NAME_VOCAB, 'r', encoding='utf-8') as f:
        vocabs = json.load(f)

    word2idx = vocabs['word2idx']
    tag2idx  = vocabs['tag2idx']
    rule2idx = vocabs['rule2idx']
    idx2tag  = {int(k): v for k, v in vocabs['idx2tag'].items()}
    idx2rule = {int(k): v for k, v in vocabs['idx2rule'].items()}

    def masked_accuracy(y_true, y_pred):
        pred = tf.cast(tf.argmax(y_pred, axis=-1), y_true.dtype)
        mask = tf.not_equal(y_true, 0)
        correct = tf.logical_and(tf.equal(pred, y_true), mask)
        return tf.reduce_sum(tf.cast(correct, tf.float32)) / (tf.reduce_sum(tf.cast(mask, tf.float32)) + 1e-8)

    model = keras.models.load_model(
        NAME_MODEL,
        custom_objects={'masked_accuracy': masked_accuracy}
    )

    print("модель загружена")
    print(f"словарь: {len(word2idx)} слов, {len(idx2tag)} тегов, {len(idx2rule)} правил")
    
    return model, word2idx, idx2tag, idx2rule


def apply_lemma_rule(word, rule):
    match = re.match(r'-(\d+)\+(.*)', rule)
    strip_len = int(match.group(1))
    append = match.group(2)

    word_lower = word.lower()
    if strip_len > 0:
        stem = word_lower[:-strip_len]
    else:
        stem = word_lower
    return stem + append


def lemmatize(text, model, word2idx, idx2tag, idx2rule, punct_pattern=re.compile(r"[^\w\s-]", re.UNICODE)):
    if isinstance(text, str):
        text = punct_pattern.sub("", text)
        words = text.split()
    else:
        words = [punct_pattern.sub("", w) for w in text]
        words = [w for w in words if w]

    if len(words) == 0:
        return ""
    
    ids = [word2idx.get(w.lower(), word2idx["<UNK>"]) for w in words]
    ids_padded = np.array([ids])

    pos_probs, lemma_probs = model.predict(ids_padded, verbose=0)
    pos_ids  = np.argmax(pos_probs[0],  axis=-1)
    rule_ids = np.argmax(lemma_probs[0], axis=-1)

    parts = []
    for i, word in enumerate(words):
        tag   = idx2tag.get(pos_ids[i], "???")
        rule  = idx2rule.get(rule_ids[i], "-0+")
        lemma = apply_lemma_rule(word, rule)
        parts.append(f"{word}{{{lemma}={tag}}}")

    return " ".join(parts)