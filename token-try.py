from keras.preprocessing.text import Tokenizer

from keras.preprocessing.text import text_to_word_sequence

print(text_to_word_sequence("how are you are youu?"))

t = Tokenizer(30, split=' ')
t.fit_on_texts("Greatly cottage thought fortune no mention he. Of mr certainty arranging am smallness by conveying. Him plate you allow built grave. Sigh sang nay sex high yet door game. She dissimilar was favourable unreserved nay expression contrasted saw. Past her find she like bore pain open. Shy lose need eyes son not shot. Jennings removing are his eat dashwood. Middleton as pretended listening he smallness perceived. Now his but two green spoil drift.")
print(t.index_docs)