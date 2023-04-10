
#***************************************NLTK Package********************

from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import matplotlib.pyplot as plt
import nltk
#nltk.download('punkt')

example_string = """
... Muad'Dib learned rapidly because his first training was in how to learn.
... And the first lesson of all was the basic trust that he could learn.
... It's shocking to find how many people do not believe they can learn,
... and how many more believe learning to be difficult."""

#You can use sent_tokenize() to split up example_string into sentences
print (sent_tokenize(example_string))
print()
print()

#tokenizing example_string by word:
print(word_tokenize(example_string))
print()
print()


#********************TOKENS:SEPARAMOS LAS PALABRAS Y CARACTERES**************
#Stop words are words that you want to ignore, 
#so you filter them out of your text when you’re processing it. 
#Very common words like 'in', 'is', and 'an' are often used 
#as stop words since they don’t add a lot of meaning to a text
#in and of themselves.

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#nltk.download("stopwords")

worf_quote = "Sir, I protest. I am not a merry man!"
stop_words = set(stopwords.words("english"))
print(worf_quote)
print()

filtered_sentence = []
for word in worf_quote.split():
    if word.casefold() not in stop_words:
        filtered_sentence.append(word)

#You have a list of the words in worf_quote, 
#so the next step is to create a set of stop words to filter words_in_quote. 
#For this example, you’ll need to focus on stop words in "english":
print(filtered_sentence)
print()

#Este codigo logra separar el "!" como caracter el anterior NO
words_in_quote = word_tokenize(worf_quote)
filtered_list = [ word for word in words_in_quote if word.casefold() not in stop_words]

print(filtered_list)
print()
#************************************Stemming********************************

#Stemming is a text processing task in which you reduce words to their root,
#which is the core part of a word. For example, 
#the words “helping” and “helper” share the root “help.” 
#Stemming allows you to zero in on the basic meaning of a word 
#rather than all the details of how it’s being used. 
#NLTK has more than one stemmer, but you’ll be using the Porter stemmer.

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

stemmer = PorterStemmer()
print("Frase Original")
string_for_stemming = """ The crew of the USS Discovery discovered many 
discoveries. Discovering is what explorers do."""
print(string_for_stemming)
print()
#Before stem the words in that string, separate all the words in it
print("Frase en TOKENS")
words = word_tokenize(string_for_stemming)
print(words)
print()

#PROCESO DE STEMMING
print("Frase con STEMMING-STEEMER PORTER")
stemmed_words = [stemmer.stem(word) for word in words]
print(stemmed_words)
print()
#The Porter stemming algorithm dates from 1979, so it’s a little on the older 
#side. The Snowball stemmer, which is also called Porter2,
#is an improvement on the original and is also available through NLTK, 
#so you can use that one in your own projects. 
#It’s also worth noting that the purpose of the Porter stemmer 
#is not to produce complete words but to find variant forms of a word.

#PROCESO ALTERNATIVO DE STEMMING-SNOWBALL
print("Frase con STEMMING-STEEMER SNOWBALL")
from nltk.stem.snowball import SnowballStemmer
stemmer_sb = SnowballStemmer("english")
stemmed_word_snowball =  [stemmer_sb.stem(word) for word in words]
print(stemmed_word_snowball) 
print()

#****************************Tagging Parts of Speech***********************

#POS tagging, is the task of labeling the words in your text according 
#to their part of speech.
#from nltk.tokenize import word_tokenize

print("Frase de Carl Sagan")
sagan_quote = """If you wish to make an apple pie from scratch,
you must first invent the universe."""
words_in_sagan_quote = word_tokenize(sagan_quote)
print(sagan_quote )
print()
#PROCESO DE IDENTIFICAR PARTES DEL SCRIPT
#import nltk
print("Frase con PROCESO DE POST-TAG")
nltk.download('averaged_perceptron_tagger')
Post_Tag_words = nltk.pos_tag(words_in_sagan_quote)
print(Post_Tag_words)
print()

#VER DESCRIPCIONNES DE PYTHON PARA TAG WORDS*********************
#import nltk
#nltk.download('tagsets')
#nltk.help.upenn_tagset()
#print()
#print()


#****************************LEMMATIZING**********************************

#Now that you’re up to speed on parts of speech, 
#you can circle back to lemmatizing. Like stemming, 
#lemmatizing reduces words to their core meaning,
# but it will give you a complete English word that makes sense
#on its own instead
import nltk
nltk.download('wordnet')
print("LEMMATIZING")
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
string_for_lemmatizing = "The friends of DeSoto love scarves."
print(string_for_lemmatizing)
print()

print("PASO 1 : TOKENS")
words = word_tokenize(string_for_lemmatizing)
print(words)
print()

print("PASO 2 : CON LEMMATIZER")
#list containing all the words in words after they’ve been lemmatized
lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
print(lemmatized_words)
print()
print()
#****************************CHUNKING*********************************
#While tokenizing allows you to identify words and sentences, 
#chunking allows you to identify phrases.

print("CHUNKING")
print()
lotr_quote = "It's a dangerous business, Frodo, going out your door."
print(lotr_quote)
print()

print("PASO 1 : TOKENS")
words_in_lotr_quote = word_tokenize(lotr_quote)
print(words_in_lotr_quote )
print()

print("PASO 2 : PROCESO DE POST-TAG")
nltk.download("averaged_perceptron_tagger")
lotr_pos_tags = nltk.pos_tag(words_in_lotr_quote)
print(lotr_pos_tags)


print("PASO 3 : CHUNKING")
grammar = "NP: {<DT>?<JJ>*<NN>}"
#According to the rule you created, your chunks:
#Start with an optional (?) determiner ('DT')
#Can have any number (*) of adjectives (JJ)
#End with a noun (<NN>)

#Create a chunk parser with this grammar:
chunk_parser = nltk.RegexpParser(grammar)

#Now try it out with your quote:
tree1 = chunk_parser.parse(lotr_pos_tags)
tree1.draw()

grammar = r"""
  NP: {<.*>+}      # chunk everything
      }<JJ>{      # except adjectives
  """
chunk_parser = nltk.RegexpParser(grammar)
tree2 = chunk_parser.parse(lotr_pos_tags)
tree2.draw()

#***************Entity Recognition (NER)*********************************
#Named entities are noun phrases that refer to specific locations, 
#people, organizations, and so on. With named entity recognition, 
#you can find the named entities in your texts and also determine 
#what kind of named entity they are.

nltk.download("maxent_ne_chunker")
nltk.download("words")
tree3 = nltk.ne_chunk(lotr_pos_tags)
tree3.draw()

#also have the option to use the parameter binary=True 
#if you just want to know what the named entities are but 
#not what kind of named entity they are:
#tree = nltk.ne_chunk(lotr_pos_tags, binary=True)
#tree.draw()

#***************EXTRACT ENTITIES FROM TEXT*********************************
#Extract named entities directly from your text. 
#Create a string from which to extract named entities. 
#You can use this quote from The War of the Worlds:

print("PASO 1 : FRASE A TRABAJAR-The War of the Worlds")
quoteWOW = """ Men like Schiaparelli watched the red planet—it is odd,
by-the-bye, that for countless centuries Mars has been the star of war 
but failed to interpret the fluctuating appearances of the markings 
they mapped so well. All that time the Martians must have been getting ready. 
During the opposition of 1894 a great light was seen on the illuminated
part of the disk, first at the Lick Observatory,
then by Perrotin of Nice, and then by other observers. 
English readers heard of it first in the issue of Nature dated August 2.""" 
print(quoteWOW )
print()

#create function to extract named entities:
language = 'english' # define the language variable
def extract_ne(quoteWOW): 
    wordsNER = word_tokenize(quoteWOW , language=language)
    tags = nltk.pos_tag(wordsNER)
    tree = nltk.ne_chunk(tags, binary=True)
    return set(
        " ".join(i[0] for i in t)
       for t in tree
       if hasattr(t, "label") and t.label() == "NE")

#With this function, you gather all named entities, 
#with no repeats. In order to do that, you tokenize by word, 
#apply part of speech tags to those words, and then extract 
#named entities based on those tags. Because you included binary=True, 
#the named entities you’ll get won’t be labeled more specifically. 
#You’ll just know that they’re named entities.

print(extract_ne(quoteWOW))
print()

#********************TEXT ANALYSIS JFK-The City Upon a Hill Speech*********************************
text8 = """ I have welcomed this opportunity to address this historic body, and, through you, the people of Massachusetts to whom I am so deeply indebted for a lifetime of friendship and trust. 

For fourteen years I have placed my confidence in the citizens of Massachusetts--and they have generously responded by placing their confidence in me. 

Now, on the Friday after next, I am to assume new and broader responsibilities. But I am not here to bid farewell to Massachusetts. 

For forty-three years--whether I was in London, Washington, the South Pacific, or elsewhere--this has been my home; and, God willing, wherever I serve this shall remain my home. 

It was here my grandparents were born--it is here I hope my grandchildren will be born. 

I speak neither from false provincial pride nor artful political flattery. For no man about to enter high office in this country can ever be unmindful of the contribution this state has made to our national greatness. 

Its leaders have shaped our destiny long before the great republic was born. Its principles have guided our footsteps in times of crisis as well as in times of calm. Its democratic institutions--including this historic body--have served as beacon lights for other nations as well as our sister states. 

For what Pericles said to the Athenians has long been true of this commonwealth: "We do not imitate--for we are a model to others." 

And so it is that I carry with me from this state to that high and lonely office to which I now succeed more than fond memories of firm friendships. The enduring qualities of Massachusetts--the common threads woven by the Pilgrim and the Puritan, the fisherman and the farmer, the Yankee and the immigrant--will not be and could not be forgotten in this nation's executive mansion. 

They are an indelible part of my life, my convictions, my view of the past, and my hopes for the future. 

Allow me to illustrate: During the last sixty days, I have been at the task of constructing an administration. It has been a long and deliberate process. Some have counseled greater speed. Others have counseled more expedient tests. 

But I have been guided by the standard John Winthrop set before his shipmates on the flagship Arbella three hundred and thirty-one years ago, as they, too, faced the task of building a new government on a perilous frontier. 

"We must always consider," he said, "that we shall be as a city upon a hill--the eyes of all people are upon us." 

Today the eyes of all people are truly upon us--and our governments, in every branch, at every level, national, state and local, must be as a city upon a hill--constructed and inhabited by men aware of their great trust and their great responsibilities. 

For we are setting out upon a voyage in 1961 no less hazardous than that undertaken by the Arabella in 1630. We are committing ourselves to tasks of statecraft no less awesome than that of governing the Massachusetts Bay Colony, beset as it was then by terror without and disorder within. 

History will not judge our endeavors--and a government cannot be selected--merely on the basis of color or creed or even party affiliation. Neither will competence and loyalty and stature, while essential to the utmost, suffice in times such as these. 

For of those to whom much is given, much is required. And when at some future date the high court of history sits in judgment on each one of us--recording whether in our brief span of service we fulfilled our responsibilities to the state--our success or failure, in whatever office we may hold, will be measured by the answers to four questions: 

First, were we truly men of courage--with the courage to stand up to one's enemies--and the courage to stand up, when necessary, to one's associates--the courage to resist public pressure, as well as private greed? 

Secondly, were we truly men of judgment--with perceptive judgment of the future as well as the past--of our own mistakes as well as the mistakes of others--with enough wisdom to know that we did not know, and enough candor to admit it? 

Third, were we truly men of integrity--men who never ran out on either the principles in which they believed or the people who believed in them--men who believed in us--men whom neither financial gain nor political ambition could ever divert from the fulfillment of our sacred trust? 

Finally, were we truly men of dedication--with an honor mortgaged to no single individual or group, and compromised by no private obligation or aim, but devoted solely to serving the public good and the national interest. 

Courage--judgment--integrity--dedication--these are the historic qualities of the Bay Colony and the Bay State--the qualities which this state has consistently sent to this chamber on Beacon Hill here in Boston and to Capitol Hill back in Washington. 

And these are the qualities which, with God's help, this son of Massachusetts hopes will characterize our government's conduct in the four stormy years that lie ahead. 

Humbly I ask His help in that undertaking--but aware that on earth His will is worked by men. I ask for your help and your prayers, as I embark on this new and solemn journey."""


print("****TEXT ANALYSIS****")
print()
from nltk import FreqDist

#text8.dispersion_plot(["woman", "lady", "girl", "gal", "man", 
                       #"gentleman", "boy", "guy"])

#A frequency distribution records the number of times each 
#outcome of an experiment has occurred. 
#For example, a frequency distribution could be used to record 
#the frequency of each word type in a document. 
#Formally, a frequency distribution can be defined as a function 
#mapping from each sample to the number of times 
#that sample occurred as an outcome.


frequency_distribution = FreqDist(text8)
print(frequency_distribution)

print("****MOST COMMON WORDS-without stopwords****")
print(frequency_distribution.most_common(20))
print()

stop_words = set(stopwords.words("english"))

filtered_sentence_txt = []
for word in text8.split():
    if word.casefold() not in stop_words:
        filtered_sentence_txt.append(word)

print(filtered_sentence_txt)
print()

print("****MOST COMMON WORDS****")
frequency_distributiontxt= FreqDist(filtered_sentence_txt)
print(frequency_distributiontxt.most_common(20))
frequency_distributiontxt.tabulate(5)
print()

#**********GRAFICOS DE 20 PALABRAS MAS COMUNES********************************
all_fdist = FreqDist(frequency_distributiontxt).most_common(20)

## Conversion to Pandas series via Python Dictionary for easier plotting
import pandas as pd
import seaborn as sns
all_fdist = pd.Series(dict(all_fdist))

## Setting figure, ax into variables
fig, ax = plt.subplots(figsize=(10,10))

## Seaborn plotting using Pandas attributes + xtick rotation for ease of viewing
all_plot = sns.barplot(x=all_fdist.index, y=all_fdist.values, ax=ax)
plt.xticks(rotation=30);
plt.show()
print()
frequency_distributiontxt.plot(20, cumulative=True)
plt.show()
print()


from nltk import FreqDist, Text
text8_obj = Text(filtered_sentence_txt)
#text8_obj.collocations()

text8_obj.dispersion_plot(["people", "believed", "great", "well","qualities","truly"])

#**********BUSCAR FRASES O GRUPOS DE PALABRAS*******************************
#Bigrams: Frequent two-word combinations
#Trigrams: Frequent three-word combinations
#Quadgrams: Frequent four-word combinations

finder = nltk.collocations.TrigramCollocationFinder.from_words(filtered_sentence_txt)
print(finder.ngram_fd.most_common(3))
print()

#****************************SENTIMENT ANALYSIS*******************************

#NLTK already has a built-in, pretrained sentiment analyzer
#called VADER (Valence Aware Dictionary and sEntiment Reasoner).
#ENTIENDE SOLO INGLES

#You’ll get back a dictionary of different scores. 
#The negative, neutral, and positive scores are related: 
#They all add up to 1 and can’t be negative. 
#The compound score is calculated differently. 
#It’s not just an average, and it can range from -1 to 1.

import nltk
#nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
sent_analysis = sia.polarity_scores("I hate python won't use it again")
print("****ANALISIS WITH VADER****")
print(sent_analysis)




#*******************FOR SPANISH ANALYSIS*************************************+
#from pysentimiento import SentimentAnalyzer
#analyzer = SentimentAnalyzer(lang="es")
#analyzer.predict("Qué gran jugador es Messi")
# returns SentimentOutput(output=POS, probas={POS: 0.998, NEG: 0.002, NEU: 0.000})

#O TAMBIEN
#import indicoio
#indicoio.config.api_key = <YOUR_API_KEY>
#indicoio.sentiment("¡Jamás voy a usar esta maldita aplicación!  No funciona para nada.")
#0.02919392219306888
#indicoio.sentiment("¡Es patrón!  La mejor que he visto.  Punto.")
#0.8860221705630639

















