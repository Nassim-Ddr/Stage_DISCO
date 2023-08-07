import spacy

# Load the language model
nlp = spacy.load("en_core_web_sm")

# Sample text with conjugated words
text = "I am running in the park and dogs are barking loudly."

# Process the text with spaCy
doc = nlp(text)

# Extract lemmatized words
lemmatized_words = [token.lemma_ for token in doc]

print(lemmatized_words)