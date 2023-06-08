import pandas as pd


class BookShelf:
    @staticmethod
    def read_menopause_books():
        print('start checking the book files')
        # create the first book object
        b1 = Book(book_name='Hormone Repair Manual', text_address='corpus/Hormone Repair Manual-latin1.txt')
        # read and trim the text contents
        b1.read_and_trim(start_phrase="Hormone revolution:\n", end_phrase="Something to look forward to.")
        b1_parts_starter = ["Hormone revolution:",
                            "Stigma, freedom, grief and everything in between",
                            "Cycle while you can",
                            "The hormonal and physiological changes of second puberty",
                            "General maintenance for perimenopause and beyond",
                            "Menopausal hormone therapy (MHT)",
                            "Rewiring the brain:",
                            "Bodily issues: weight gain",
                            "Estrogen rollercoaster: ",
                            "This is the chapter for once you've achieved menopause"]
        b1.find_parts(b1_parts_starter)
        b1.find_paragraphs()
        print("{} is preprocessed...".format(b1))

        # create the second book
        b2 = Book(book_name="Next Level", text_address="corpus/Next Level-latin1.txt")
        # read and trim the text contents
        b2.read_and_trim(start_phrase="THE STATS. THE STIGMA. THE SILENCE.",
                         end_phrase="Well done!")
        b2_parts_starter = ["THE STATS. THE STIGMA. THE SILENCE.".upper(),
                            "THE SCIENCE OF THE MENOPAUSE TRANSITION".upper(),
                            "Hormones and Symptoms Explained".upper(),
                            "Menopausal Hormone Therapy, Adaptogens, and Other Interventions".upper(),
                            "Kick Up Your Cardio".upper(),
                            "Now's the Time to Lift Heavy Sh*t!".upper(),
                            "Get a Jump on Menopausal Strength Losses".upper(),
                            "Gut Health for Athletic Glory".upper(),
                            "Eat Enough!".upper(),
                            "Fueling for the Menopause Transition".upper(),
                            "Nail Your Nutrition Timing".upper(),
                            "You can't necessarily trust your thirst right now.",
                            "Sleep Well and Recover Right".upper(),
                            "Stability, Mobility, and Core Strength: Keep Your Foundation Strong".upper(),
                            "Motivation and the Mental Game: Your Mind Matters".upper(),
                            "Keep Your Skeleton Strong".upper(),
                            "Strategies for Exercising Through the Transition".upper(),
                            "Supplements: What You Need and What You Don't".upper(),
                            "Pulling It All Together".upper()
                            ]
        b2.find_parts(b2_parts_starter)
        b2.find_paragraphs()
        print("{} is preprocessed...".format(b2))

        # create the third book
        b3 = Book(book_name="Queen Menopause", text_address="corpus/Queen Menopause-latin1.txt")
        # read and trim the text contents
        b3.read_and_trim(start_phrase="I had other names for this book",
                         end_phrase="This book was written over three years and one pandemic")
        b3_parts_starter = ["I had other names for this book"]
        b3_parts_starter.extend(["CHAPTER {}".format(i) for i in range(2, 13)])
        b3.find_parts(b3_parts_starter)
        b3.find_paragraphs()
        print("{} is preprocessed...".format(b3))

        # create the fourth book
        b4 = Book(book_name="The Menopause Manifesto", text_address="corpus/The Menopause Manifesto-latin1.txt")
        # read and trim the text contetns
        b4.read_and_trim(start_phrase="IF MENOPAUSE WERE ON YELP it would have one star.",
                         end_phrase="That's my manifesto.")
        b4_parts_starter = ["IF MENOPAUSE WERE ON YELP it would have one star.",
                            "Reclaiming the Change:",
                            "A Second Coming of Age:",
                            "The History and Language of Menopause:",
                            "The Biology of Menopause:",
                            "The Evolutionary Advantage of Menopause:",
                            "The Timing of Menopause:",
                            "When Periods and Ovulation Stop Before Age Forty:",
                            "Understanding the Change:",
                            "Metamorphoses of Menopause:",
                            "The Heart of the Matter:",
                            "Here or Is It Just Me?",
                            "Menstrual Mayhem:",
                            "Bone Health:",
                            "This Is Your Brain on Menopause:",
                            "The Vagina and Vulva:",
                            "Bladder Health:",
                            "Let's Talk About Sex:",
                            "Will I Ever Feel Rested Again?",
                            "Therapy for the Change:",
                            "The Messy History and Where We Are Today",
                            "The Cinematic Universe of Hormones:",
                            "Phytoestrogens, Food, and Hormones:",
                            "Bioidenticals, Naturals, and Compounding:",
                            "What Makes a Healthy Menopause and Beyond",
                            "Supplements and Menopause",
                            "Contraception and the Menopause Transition:",
                            "Taking Charge of the Change",
                            "Welcome to My Menoparty:",
                            "Final Thoughts:"]
        b4.find_parts(b4_parts_starter)
        b4.find_paragraphs()

        # create the fifth book
        b5 = Book(book_name="Menopausing", text_address="corpus/Menopausing-latin1-with relevant image text.txt")
        # read and trim the text contetns
        b5.read_and_trim(start_phrase="INTRODUCTION:",
                         end_phrase="MENOPAUSE WARRIORS")
        b5_parts_starter = ["INTRODUCTION:"]
        b5_parts_starter.extend(["Chapter {}\n".format(i) for i in range(1, 15)])
        b5.find_parts(b5_parts_starter)
        b5.find_paragraphs()

        bookshelf = BookShelf()
        bookshelf.add_book(b1)
        bookshelf.add_book(b2)
        bookshelf.add_book(b3)
        bookshelf.add_book(b4)
        bookshelf.add_book(b5)
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
        return chapters_df


class Book:
    def __init__(self, book_name, text_address):
        self.text_address = text_address
        self.book_name = book_name
        self.whole_text = ''
        self.trimmed_text = ''
        self.parts = []
        self.paragraphs = {}

    def read_and_trim(self, start_phrase, end_phrase):
        with open(self.text_address, 'r', encoding='latin-1') as fp:
            self.whole_text = fp.read()
        self.trimmed_text = self.whole_text[self.whole_text.index(start_phrase):self.whole_text.index(end_phrase)]
        print('text are trimmed - {}'.format(self.book_name))

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

    def find_paragraphs(self, paragraph_pattern='\t', length_thresholds=150):
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

    def __str__(self):
        return self.book_name


if __name__ == "__main__":
    print('Creating a bookshelf with default books in the shelf')
    bookshelf = BookShelf.read_menopause_books()
