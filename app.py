import pickle
from collections import defaultdict,Counter
import streamlit as st
from code.model import HMM

st.markdown('''
            ## hello!!! Interested in linguistics and machine learning at the same time 🙀
            ## This is the correct place for you 😃
            ## Welcome To My POS tagging site 😎
            ''')
columns = st.columns(2)

with columns[0]:
    with st.container(border=True):
        st.markdown('''#### Please write the input sentence 🙇''')
        st.session_state.input = st.text_area(label='give your input here....')
with columns[1]:
    with st.container(border=True):
        if st.button(label="click me to generate the POS tags"):
            if 'input' not in st.session_state:
                st.error(body="I cannot find the input string 😢")
            else:
                with st.status(label="loading model"):
                    with open('hmm_model.pkl', 'rb') as f:
                        model = pickle.load(f)
                with st.status(label = "generating sentence",expanded=True):
                    output_sentence = model.forward(st.session_state.input)
                    st.markdown('''##### These are the output POS tags 😌:''')
                    with st.container(border=True):
                        st.markdown(f"`{output_sentence}`")


