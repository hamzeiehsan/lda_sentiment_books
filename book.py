import pandas as pd
import logging
import requests
import time

logging.basicConfig(level=logging.INFO)


class BookShelf:
    @staticmethod
    def read_sample_books():
        end_phrase = "*** END OF THE PROJECT GUTENBERG EBOOK"
        logging.info('{} - start checking the book files'.format('Bookshelf:'))
        # create the first book object
        b1 = Book(book_name='A Christmas Carol', text_address='https://www.gutenberg.org/cache/epub/46/pg46.txt')
        # read and trim the text contents
        b1.read_and_trim(start_phrase="STAVE I:  MARLEY'S GHOST", end_phrase=end_phrase)
        b1_parts_starter = ["MARLEY was dead:",
                            "WHEN Scrooge awoke, it was so dark",
                            "AWAKING in the middle of a prodigiously tough snore",
                            "THE Phantom slowly, gravely, silently, approached.",
                            "YES! and the bedpost was his own",
                            ]
        b1.find_parts(b1_parts_starter)
        b1.find_paragraphs()
        logging.info("\t{} is preprocessed...".format(b1))
        time.sleep(2)

        # create the second book
        b2 = Book(book_name="Crime and Punishment", text_address="https://www.gutenberg.org/files/2554/2554-0.txt")
        # read and trim the text contents
        b2.read_and_trim(start_phrase="PART I",
                         end_phrase=end_phrase)
        b2_parts_starter = ["On an exceptionally hot evening early in July ",
                            "Raskolnikov was not used to crowds,",
                            "He waked up late next day after a broken sleep.",
                            "letter had been a torture to him, but as regards",
                            "The question why he was now going to Razumihin agitated him",
                            "Later on Raskolnikov happened to find out why the huckster",
                            "The door was as before opened a tiny crack, and again",
                            "So he lay a very long while.",
                            "And what if there has been a search already?",
                            "He was not completely unconscious",
                            "Zossimov was a tall, fat man with a puffy",
                            "This was a gentleman no longer young",
                            "But as soon as she went out, he got up",
                            "An elegant carriage stood in the middle of the road ",
                            "Raskolnikov got up, and sat down on the sofa",
                            "Razumihin waked up next morning at eight ",
                            "He is well, quite well!",
                            "At that moment the door was softly opened",
                            "Raskolnikov was already entering the room.",
                            "They were by now approaching Bakaleyev",
                            "He looked carefully and suspiciously at the unexpected visitor.",
                            "that landowner in whose house my sister was",
                            "The fact was that up to the last moment he had never expected",
                            "Raskolnikov went straight to the house on the canal",
                            "When next morning at eleven o",
                            "When he remembered the scene afterwards",
                            "The morning that followed the fateful interview with Dounia",
                            "It would be difficult to explain exactly what",
                            "Katerina Ivanovna remained standing where she was",
                            "Raskolnikov had been a vigorous and active champion",
                            "Lebeziatnikov looked perturbed.",
                            "A strange period began for Raskolnikov",
                            "Porfiry Petrovitch ejaculated at last",
                            "It means that I am not going to lose sight of you now.",
                            "He spent that evening till ten o",
                            "The same day, about seven o"
                            ]
        b2.find_parts(b2_parts_starter)
        b2.find_paragraphs()
        logging.info("\t{} is preprocessed...".format(b2))
        time.sleep(2)

        # create the third book
        b3 = Book(book_name="Alice's Adventures in Wonderland",
                  text_address="https://www.gutenberg.org/cache/epub/11/pg11.txt")
        # read and trim the text contents
        b3.read_and_trim(start_phrase="Alice was beginning to get very tired of sitting by her sister",
                         end_phrase=end_phrase)
        b3_parts_starter = ["Alice was beginning to get very tired of sitting by her sister",
                            "CHAPTER II.",
                            "CHAPTER III.",
                            "CHAPTER IV.",
                            "CHAPTER V.",
                            "CHAPTER VI.",
                            "CHAPTER VII.",
                            "CHAPTER VIII.",
                            "CHAPTER IX.",
                            "CHAPTER X.",
                            "CHAPTER XI.",
                            "CHAPTER XII."
                            ]
        b3.find_parts(b3_parts_starter)
        b3.find_paragraphs()
        logging.info("\t{} is preprocessed...".format(b3))
        time.sleep(2)

        # create the fourth book
        b4 = Book(book_name="Metamorphosis", text_address="https://www.gutenberg.org/files/5200/5200-0.txt")
        # read and trim the text contents
        b4.read_and_trim(start_phrase="Translated by David Wyllie",
                         end_phrase=end_phrase)
        b4_parts_starter = ["I",
                            "II",
                            "III"]
        b4.find_parts(b4_parts_starter)
        b4.find_paragraphs()
        logging.info("\t{} is preprocessed...".format(b4))


        bookshelf = BookShelf()
        bookshelf.add_book(b1)
        bookshelf.add_book(b2)
        bookshelf.add_book(b3)
        bookshelf.add_book(b4)
        logging.info('all books are added to the bookshelf object.\n')
        return bookshelf

    def __init__(self):
        self.books = []
        self.book_dict = {}

    def generate_book_dict(self):
        self.book_dict = {book.book_name: book for book in self.books}

    def add_book(self, book):
        self.books.append(book)
        self.book_dict[book.book_name] = book

    def add_books(self, books):
        self.books.extend(books)
        for book in books:
            self.book_dict[book.book_name] = book

    def create_corpus(self, book='all'):
        if book == 'all':
            books_names = [book.book_name for book in self.books]
        else:
            books_names = [book]
        corpus = []
        corpus_info = {}
        for book_name in books_names:
            corpus_info[book_name] = {}
            b = self.book_dict[book_name]
            for part, paragraphs in b.paragraphs.items():
                corpus_info[book_name][part] = {'start': len(corpus), 'end': len(corpus) + len(paragraphs)}
                corpus.extend(paragraphs)
        return corpus, corpus_info

    def create_corpus_df(self):
        df_dict_list = []
        for b in self.books:
            for part, paragraphs in b.paragraphs.items():
                for idx, p in enumerate(paragraphs):
                    df_dict_list.append({"book": b.book_name,
                                         "chapter": part + 1,
                                         "paragraph_number": idx + 1,
                                         "paragraph": p})

        return pd.DataFrame(df_dict_list)


    def create_corpus_custom_df(self, book, chapters=[]):
        all_df = self.create_corpus_df()
        if len(chapters) == 0:
            chapters = [i+1 for i in range(len(self.book_dict[book].parts))]
        book_df = all_df[all_df['book'] == book]
        chapters_df = book_df[book_df['chapter'].isin(chapters)]
        chapters_df = chapters_df.reset_index()
        return chapters_df


class Book:
    def __init__(self, book_name, text_address):
        self.text_address = text_address
        self.book_name = book_name
        self.whole_text = ''
        self.trimmed_text = ''
        self.parts = []
        self.paragraphs = {}

    def read_and_trim(self, start_phrase, end_phrase, is_file=False):
        if is_file:
            with open(self.text_address, 'r', encoding='latin-1') as fp:
                self.whole_text = fp.read()
        else:
            headers = {
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.75 Safari/537.36'}
            req = requests.get(self.text_address, headers=headers)
            self.whole_text = req.text
        self.trimmed_text = self.whole_text[self.whole_text.index(start_phrase):self.whole_text.index(end_phrase)]
        logging.info('\t\ttext are trimmed - {}'.format(self.book_name))

    def find_parts(self, part_pattern_starter):
        if self.parts is None or len(self.parts) > 0:
            self.parts = []
        for idx, starter in enumerate(part_pattern_starter):
            start = self.trimmed_text.index(starter)
            if idx < len(part_pattern_starter) - 1:
                end = self.trimmed_text.index(part_pattern_starter[idx + 1])
            else:
                end = len(self.trimmed_text) - 1
            part = self.trimmed_text[start:end]
            self.parts.append(part)
        logging.info('\t\t{} parts are identified'.format(len(self.parts)))

    def find_paragraphs(self, paragraph_pattern='\t', length_thresholds=150):
        len_parags = 0
        for idx, part in enumerate(self.parts):
            if idx not in self.paragraphs.keys():
                self.paragraphs[idx] = []
            splitted = part.split(paragraph_pattern)
            current = ''
            for split in splitted:
                current += split.strip()
                if len(split.strip()) > length_thresholds:
                    self.paragraphs[idx].append(current)
                    current = ''
            len_parags += len(self.paragraphs[idx])
        logging.info('\t\tparagraphs are identified, total number of paragraphs in this book: {}\n'.format(len_parags))

    def __str__(self):
        return self.book_name


if __name__ == "__main__":
    print('Creating a bookshelf with default books in the shelf')
    bookshelf = BookShelf.read_sample_books()
