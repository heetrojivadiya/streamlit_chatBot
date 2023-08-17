from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
import streamlit as st
import os

baseName = os.path.basename(__file__)
dirName = os.path.dirname(__file__)

# api_key = "sk-API_key"
# v = input("Enter API KEY: ")

st.set_page_config("Dlubal ChatBot")
# api_key = api_key+str(v)
st.sidebar.write('Enter OpenAI API Key 👇')
api_key = st.sidebar.text_input(
    label="### Your OpenAI API key 👇",
    placeholder="Paste your openAI API key, sk-",
    type="password")

os.environ['OPENAI_API_KEY'] = str(api_key)

user_api_key = os.getenv("OPENAI_API_KEY", "")

csv_file_path = dirName + r".\faq_data.csv"

if os.path.isfile(csv_file_path) and api_key:
    loader = CSVLoader(file_path=csv_file_path, encoding="utf-8")
    data = loader.load()

    embeddings = OpenAIEmbeddings(openai_api_key=user_api_key)
    vectors = FAISS.from_documents(data, embeddings)

    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo', openai_api_key=str(user_api_key)),
        retriever=vectors.as_retriever()
    )

    def conversational_chat(query):
        result = chain({"question": query, "chat_history": history})
        history.append((query, result["answer"]))
        return result["answer"]

    history = []
    # print("Hello! Ask me anything about Dlubal Products")

    st.title('Dlubal ChatBot')
    user_input = st.text_input('Question:', placeholder='Ask me about Dlubal Products..!')
    ask = st.button('Ask')
    if user_input or ask:
        answer = conversational_chat(user_input)
        st.empty()
        st.text('')
        st.markdown(str(answer))

        # if answer:
        #     st.text('')
        #     st.text('')
        #     col1, col2, col3 = st.columns([5,2,5])
        #     with col1:
        #         st.write("Do you agree with the responce?")
        #     with col2:
        #         like=st.button('👍')
        #     with col3:
        #         dislike=st.button('👎')
