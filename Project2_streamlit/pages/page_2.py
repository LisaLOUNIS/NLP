import streamlit as st
from streamlit_tags import st_tags
from symspellpy import SymSpell, Verbosity

st.title('Correcteur orthographique')

# Initialisation de SymSpell
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = "fr-100k.txt"  # Remplacez par le chemin vers votre dictionnaire français
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

# Boîte de saisie avec suggestions
user_input = st_tags(
    label='Entrez un mot :',
    text='Appuyez sur entrer pour ajouter plus',
    value=[],
    suggestions=list(sym_spell.words.keys()),  # Utilisez votre dictionnaire comme suggestions
    maxtags=1,
    key='1'
)

# Affichage des suggestions de correction
if user_input:
    input_word = user_input[0]
    suggestions = sym_spell.lookup(input_word, Verbosity.CLOSEST, max_edit_distance=2)
    st.write(f'Suggestions de correction pour "{input_word}" :')
    for suggestion in suggestions:
        st.write(suggestion.term)