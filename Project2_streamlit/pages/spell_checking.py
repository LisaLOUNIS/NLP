import streamlit as st
from symspellpy import SymSpell, Verbosity
import pandas as pd
from spellchecker import SpellChecker

st.title('Correcteur orthographique')

# Initialisation de SymSpell
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = "fr-100k.txt"  
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

def spell_check(word_list):
    corrected_dict = {}
    for word in word_list:
        suggestions = sym_spell.lookup(word, Verbosity.ALL, max_edit_distance=2)
        if suggestions:
            corrected_dict[word] = [suggestion.term for suggestion in suggestions]
        else:
            corrected_dict[word] = []
    return corrected_dict

# Boîte de saisie
user_input = st.text_input('Saisissez du texe en français :')

st.write("# SymSpellCheck")

# Affichage des suggestions de correction
if user_input:
    spell_dict = spell_check(user_input.split())
    cols = st.columns(len(spell_dict))
    for col, (word, suggestions) in zip(cols, spell_dict.items()):
        df = pd.DataFrame(suggestions[:5], columns=[word])
        col.dataframe(df, hide_index=True)


st.write("# Spell Checker")

spell = SpellChecker(language='fr')  # pour le français

if user_input:
    words = user_input.split()
    misspelled = spell.unknown(words)
    cols = st.columns(len(misspelled))
    for col, word in zip(cols, misspelled):
        suggestions = spell.candidates(word)
        df = pd.DataFrame(list(suggestions)[:5], columns=[word])
        col.dataframe(df, hide_index=True)