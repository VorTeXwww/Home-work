import re
from collections import Counter, defaultdict
from nltk.stem.snowball import SnowballStemmer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
try:
    import pymorphy3  # type: ignore
    MORPH = pymorphy3.MorphAnalyzer()
except Exception:
    MORPH = None
CHAT_FILE = "example_chat.txt"
STOP_WORDS = {
    "и","в","во","не","что","он","на","я","с","со","как","а","то","все","она","так","его","но","да",
    "ты","к","у","же","вы","за","бы","по","ее","мне","было","вот","от","меня","еще","нет","о","из",
    "ему","теперь","когда","даже","ну","ли","если","или","ни","быть","был","него","до","вас","нибудь",
    "уж","вам","там","потом","себя","ничего","ей","может","они","тут","где","есть","надо","ней","для",
    "мы","тебя","их","чем","была","сам","чтоб","без","чего","раз","тоже","себе","под","будет","ж","тогда",
    "кто","этот","того","потому","этого","какой","совсем","ним","здесь","этом","один","почти","мой","тем",
    "чтобы","нее","сейчас","были","куда","зачем","всех","никогда","можно","при","наконец","два","об","другой",
    "хоть","после","над","больше","тот","через","эти","нас","про","всего","них","какая","много","разве",
    "три","эту","моя","впрочем","хорошо","свою","этой","перед","иногда","лучше","чуть","том","нельзя","такой",
    "им","более","всегда","конечно","всю","между","это","какое","какие","какую","эта","эти","эту","та","те",
    "только","именно","ладно","доброе","давай","сегодня","вечером","утром","домой","дома","нормально","сначала",
    "кажется","опять","просто","вроде","очень","реально","согласен","будешь","делать","делаешь","сделать","сделал",
    "более","менее","минут","какая","какой","какое","какие","какую"
}

MESSAGE_PATTERN = re.compile(r"^\[(\d{2}:\d{2}, \d{2}\.\d{2}\.\d{4})\]\s([^:]+):\s(.*)$")
RU_STEMMER = SnowballStemmer("russian")
EN_STEMMER = SnowballStemmer("english")
NAME_MAP = {
    "глеб": "Глеб",
    "юра": "Юра",
}
def normalize_name(name: str) -> str:
    return NAME_MAP.get(name.strip().lower(), name.strip())
def normalize_token(token: str) -> str:
    token = token.lower().replace("ё", "е")
    if re.search(r"[a-z]", token):
        return EN_STEMMER.stem(token)
    if MORPH is not None:
        return MORPH.parse(token)[0].normal_form
    return RU_STEMMER.stem(token)
def clean_tokens(text: str):
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^\w\sа-яa-z]", " ", text)
    tokens = []
    for token in text.split():
        if token.isdigit() or len(token) < 3:
            continue
        if token in STOP_WORDS:
            continue
        normalized = normalize_token(token)
        if normalized in STOP_WORDS or len(normalized) < 3:
            continue
        tokens.append(normalized)
    return tokens
def read_chat(path: str):
    messages = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = MESSAGE_PATTERN.match(line)
            if not match:
                continue
            dt, sender, text = match.groups()
            messages.append(
                {
                    "datetime": dt,
                    "sender": normalize_name(sender),
                    "text": text,
                    "tokens": clean_tokens(text),
                }
            )
    return messages
def top_words(messages, n=20):
    counter = Counter()
    for msg in messages:
        counter.update(msg["tokens"])
    return counter.most_common(n)
def top_words_by_user(messages, n=10):
    data = defaultdict(Counter)
    for msg in messages:
        data[msg["sender"]].update(msg["tokens"])
    return {user: counter.most_common(n) for user, counter in data.items()}
def cluster_messages(messages, n_clusters=4):
    docs = [" ".join(msg["tokens"]) for msg in messages if msg["tokens"]]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(docs)

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels = model.fit_predict(X)
    terms = vectorizer.get_feature_names_out()
    order = model.cluster_centers_.argsort(axis=1)[:, ::-1]

    result = []
    for idx in range(n_clusters):
        keywords = [terms[i] for i in order[idx, :8]]
        result.append((idx, keywords))
    return result
def topic_modeling(messages, n_topics=4, n_top_words=8):
    docs = [" ".join(msg["tokens"]) for msg in messages if msg["tokens"]]
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(docs)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)
    terms = vectorizer.get_feature_names_out()
    topics = []
    for idx, topic in enumerate(lda.components_):
        keywords = [terms[i] for i in topic.argsort()[-n_top_words:][::-1]]
        topics.append((idx + 1, keywords))
    return topics
def main():
    messages = read_chat(CHAT_FILE)
    print("Всего сообщений:", len(messages))
    print("\nТоп-20 слов чата:")
    for word, count in top_words(messages, 20):
        print(f"{word}: {count}")
    print("\nТоп слов по пользователям:")
    for user, words in top_words_by_user(messages, 10).items():
        print(f"\n{user}:")
        for word, count in words:
            print(f"  {word}: {count}")
    print("\nКластеры сообщений:")
    for cluster_id, keywords in cluster_messages(messages, 4):
        print(f"Кластер {cluster_id}: {', '.join(keywords)}")
    print("\nТемы LDA:")
    for topic_id, keywords in topic_modeling(messages, 4, 8):
        print(f"Тема {topic_id}: {', '.join(keywords)}")
if __name__ == "__main__":
    main()
