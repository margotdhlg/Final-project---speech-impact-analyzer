import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load only safe resources
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english')) - {"we", "you", "our", "us"}
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z!? ]", " ", text)

    # Regex-based tokenizer instead of nltk.word_tokenize
    tokens = re.findall(r"\b\w+\b", text)

    cleaned = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(cleaned)
