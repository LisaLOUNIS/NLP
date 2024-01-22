import streamlit as st
from symspellpy import SymSpell, Verbosity

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

# Affichage des suggestions de correction
if user_input:
    spell_dict = spell_check(user_input.split())
    st.write(spell_dict)