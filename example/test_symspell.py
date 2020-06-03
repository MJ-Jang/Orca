from symspellpy.symspellpy import SymSpell, Verbosity
import pkg_resources

max_edit_distance_dictionary = 2
prefix_length = 7
# create object
sym_spell = SymSpell(max_edit_distance_dictionary, prefix_length)

# 1. create dictionary using corpus.txt
sym_spell.create_dictionary('data/test_data.txt')

a = 0
for key, count in sym_spell.words.items():
    a += 1
    print("{} {}".format(key, count))

a = 0
for key, count in sym_spell.bigrams.items():
    a += 1
    print("{} {}".format(key, count))


# 2. spellcheck
# maximum edit distance per dictionary precalculation
max_edit_distance_dictionary = 2
prefix_length = 7

# create object
sym_spell = SymSpell(max_edit_distance_dictionary, prefix_length)
# load dictionary
dictionary_path = pkg_resources.resource_filename(
        "symspellpy", "frequency_dictionary_en_82_765.txt")

bigram_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_bigramdictionary_en_243_342.txt")
# term_index is the column of the term and count_index is the
# column of the term frequency
if not sym_spell.load_dictionary(dictionary_path, term_index=0,
                                 count_index=1):
    print("Dictionary file not found")

if not sym_spell.load_bigram_dictionary(bigram_path, term_index=0,
                                        count_index=2):
    print("Bigram dictionary file not found")

# lookup suggestions for single-word input strings
input_term = "memebers"  # misspelling of "members"

# max edit distance per lookup
# (max_edit_distance_lookup <= max_edit_distance_dictionary)
max_edit_distance_lookup = 2
suggestion_verbosity = Verbosity.CLOSEST  # TOP, CLOSEST, ALL
suggestions = sym_spell.lookup(input_term, suggestion_verbosity,
                                   max_edit_distance_lookup)

# display suggestion term, term frequency, and edit distance
for suggestion in suggestions:
    print("{}, {}, {}".format(suggestion.term, suggestion.distance,
                                suggestion.count))

# lookup suggestions for multi-word input strings (supports compound
# splitting & merging)
input_term = ("whereis th elove hehad dated forImuch of thepast who "
                  "couqdn'tread in sixtgrade and ins pired him")
# max edit distance per lookup (per single word, not per whole input string)
max_edit_distance_lookup = 2
suggestions = sym_spell.lookup_compound(input_term,
                                        max_edit_distance_lookup)
# display suggestion term, edit distance, and term frequency
for suggestion in suggestions:
    print("{}, {}, {}".format(suggestion.term, suggestion.distance,
                              suggestion.count))