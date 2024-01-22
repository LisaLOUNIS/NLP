import streamlit as st
from symspellpy import SymSpell, Verbosity

st.title('Correcteur orthographique')

# Initialisation de SymSpell
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = "../fr-100k.txt"  # Remplacez par le chemin vers votre dictionnaire français
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

# Boîte de saisie
user_input = st.text_input('Entrez un mot :')

# Affichage des suggestions de correction
if user_input:
    suggestions = sym_spell.lookup(user_input, Verbosity.CLOSEST, max_edit_distance=2)
    st.write(f'Suggestions de correction pour "{user_input}" :')
    for suggestion in suggestions:
        st.write(suggestion.term)