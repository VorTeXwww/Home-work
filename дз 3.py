import re
import math
from collections import Counter
from sklearn.datasets import fetch_20newsgroups

def preprocess_text(text, stop_words):
    text = text.lower()
    words = re.findall(r'\b[a-z]+\b', text)
    words = [w for w in words if w not in stop_words]
    words = [w[:-1] if w.endswith('s') and not w.endswith('ss') else w for w in words]
    return words

raw_text = "The news mentioned here is fake. Audience do not encourage fake news. Fake news is false or misleading"
stop_words_list = {'is', 'here', 'do', 'not', 'or', 'the'}

sentences = re.split(r'\.\s*', raw_text.strip())
tokenized_sentences = [preprocess_text(s, stop_words_list) for s in sentences if s]

all_words = []
for s in tokenized_sentences:
    all_words.extend(s)

unique_words = list(set(all_words))
word_freq = Counter(all_words)
sorted_vocab = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
vocab = [w[0] for w in sorted_vocab]

b_bow = []
c_bow = []
for s in tokenized_sentences:
    b_bow.append([1 if w in s else 0 for w in vocab])
    c_bow.append([s.count(w) for w in vocab])

N = len(tokenized_sentences)
idf = {}
for w in vocab:
    df = sum(1 for s in tokenized_sentences if w in s)
    idf[w] = math.log(N / (df + 1))

tfidf = []
for s in tokenized_sentences:
    vec = []
    for w in vocab:
        tf = s.count(w) / len(s) if len(s) > 0 else 0
        vec.append(tf * idf[w])
    tfidf.append(vec)

def vectorize_corpus(docs, method='cbow'):
    corpus_vocab = Counter(w for doc in docs for w in doc)
    valid_vocab = {w for w, c in corpus_vocab.items() if c > 3}
    vectorized = []
    idf_dict = {}
    N_docs = len(docs)

    if method == 'tfidf':
        for w in valid_vocab:
            df = sum(1 for d in docs if w in d)
            idf_dict[w] = math.log(N_docs / (df + 1))

    for doc_id, doc in enumerate(docs):
        doc_dict = {}
        doc_len = len(doc)
        for w in doc:
            if w in valid_vocab:
                if method == 'bbow':
                    doc_dict[w] = 1
                elif method == 'cbow':
                    doc_dict[w] = doc_dict.get(w, 0) + 1
                elif method == 'tfidf':
                    tf = doc.count(w) / doc_len
                    doc_dict[w] = tf * idf_dict[w]
        vectorized.append({f'doc{doc_id}': doc_dict})
    return vectorized

class NaiveBayes:
    def __init__(self):
        self.classes = {}
        self.vocab = set()
        self.word_counts = {}
        self.class_totals = {}
        self.total_docs = 0

    def fit(self, X, y):
        self.total_docs = len(y)
        for c in set(y):
            self.classes[c] = y.count(c) / self.total_docs
            self.word_counts[c] = Counter()
            self.class_totals[c] = 0

        for doc_dict, c in zip(X, y):
            doc_vals = list(doc_dict.values())[0]
            for word, count in doc_vals.items():
                self.vocab.add(word)
                self.word_counts[c][word] += count
                self.class_totals[c] += count

    def predict(self, X):
        preds = []
        v_size = len(self.vocab)
        for doc_dict in X:
            doc_vals = list(doc_dict.values())[0]
            best_c = None
            max_score = -float('inf')
            for c in self.classes:
                score = math.log(self.classes[c])
                for word, count in doc_vals.items():
                    if word in self.vocab:
                        w_count = self.word_counts[c].get(word, 0)
                        prob = (w_count + 1) / (self.class_totals[c] + v_size)
                        score += count * math.log(prob)
                if score > max_score:
                    max_score = score
                    best_c = c
            preds.append(best_c)
        return preds

data = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
X_raw = data.data[:300]
y_train = list(data.target[:300])

X_token = [preprocess_text(t, stop_words_list) for t in X_raw]
X_train_cbow = vectorize_corpus(X_token, method='cbow')

nb = NaiveBayes()
nb.fit(X_train_cbow, y_train)
predictions = nb.predict(X_train_cbow[:5])

regex_letters = r"^[a-zA-Z]+$"
regex_ends_b = r"^[a-z]*b$"
regex_ab_rule = r"^(b|bab)*$"

print(f"(по убыванию частоты): \n{vocab}\n")

print("(B-BoW):")
for tokens, vec in zip(tokenized_sentences, b_bow):
    print(f"{' '.join(tokens)} — {vec}")

print("\n(C-BoW):")
for tokens, vec in zip(tokenized_sentences, c_bow):
    print(f"{' '.join(tokens)} — {vec}")

print("\n(TF-IDF):")
for tokens, vec in zip(tokenized_sentences, tfidf):
    rounded_vec = [round(v, 3) for v in vec]
    print(f"{' '.join(tokens)} — {rounded_vec}")

print("классификация")
first_doc = list(X_train_cbow[0].values())[0]
first_few_items = dict(list(first_doc.items())[:5])
print(f"пример разреженного вектора для 1 (C-BoW): \n{{'doc0': {first_few_items} ...}}\n")

print(f"предсказания Байеса: {predictions}")
print(f"метки классов:                 {y_train[:5]}")

print("\n\nДоп задание:")
print(f"только буквы ('Hello'): {bool(re.match(regex_letters, 'Hello'))}")
print(f"строчные, кончается на 'b' ('superb'): {bool(re.match(regex_ends_b, 'superb'))}")
print(f"правило 'bab' ('babbab'): {bool(re.match(regex_ab_rule, 'babbab'))}")