import re
import numpy as np

def rhetorical_features(text):
    text = text.lower()
    # Simple proxy patterns
    logos_words = len(re.findall(r"\b(because|therefore|data|study|fact)\b", text))
    pathos_words = len(re.findall(r"\b(love|fear|anger|hope|fight|cry)\b", text))
    ethos_words = len(re.findall(r"\b(leader|trust|expert|promise|integrity)\b", text))
    
    # Punctuation cues
    exclaims = text.count("!")
    questions = text.count("?")
    
    logos = min(1.0, (logos_words + 0.2) / 10)
    pathos = min(1.0, (pathos_words + exclaims + questions) / 15)
    ethos = min(1.0, (ethos_words + 0.3) / 8)
    
    return np.array([logos, pathos, ethos])
