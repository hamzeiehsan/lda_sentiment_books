# Topic Modelling using LDA (Gensim Library)

## Description
Here, I have implemented several classes to make the process of topic modelling easier:
- Bookshelf: It consists several book object (from class Book). Bookshelf enable managing one or multiple book and transforming their contents into dictionary and pandas dataframe format.
- Book: It consistent of processed information of a book. A book has a name, a reference to text file containing the textual contents, and some additional metadata such as how to trim the unwanted text from start and end of the textual content, how to divide it into parts (based on starting sentence or heading of each part) and finally how identify the paragraphs - here default is just by splitting based on tab '\t', having a certain a minimum threshold of word counts.
- Processor: It's a class that orchestrate a series of preprocessing steps. This can be configured to make a suitable pipline.
- LDAModel: LDA model a wrapper class Gensim that take a pandas dataframe format from bookshelf, perform preprocessing, plot c_v and u_mass measures to help you find optimum number for number of topics and finally create topics and label the dominant topic and the percentage of contribution of the dominant topic for each paragraph
- Sentiment: is a class with several functions that takes a paragraph and derive the sentiment results from flair. 
- BookVis: This class combines the results of LDA model and sentiment model with the hierarchy captured in Bookshelf: shelf > books > parts > paragraphs and create several plots in each level.
## Dependencies
- pandas
- gensim
- matplotlib
- seaborn
- pyLDAvis
- flair
- spacy (run: 'python -m spacy download en_core_web_md' after installation)
- wordcloud
- plotnine (run: pip install 'plotnine[all]')
- itable 


## Sample Books
5 sample books are automatically downloaded from [Project Gutenberg](https://www.gutenberg.org/):
- A Christmas Carol (Charles Dickens)
- Crime and Punishment (Fyodor Dostoyevsky)
- Pride and Prejudice (Jane Austin)
- Romeo and Juliet (William Shakespeare)
- Metamorphosis  (Franz Kafka)


## Running the Code
You can use the implementation (partially or completely) to analyse sentiments in books and identify topics using LDA. 

Check the provided notebook on how you can create bookshelf and analyse the content of all and each of the books.

