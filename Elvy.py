import textwrap
import numpy as np
import json
import pandas as pd
import google.generativeai as genai
import google.ai.generativelanguage as glm
# Used to securely store your API key
from dotenv import load_dotenv
import os
from IPython.display import Markdown

class Elvy:
    def __init__(self,query):
        self.query=query
        self.model = 'models/embedding-001'
        load_dotenv()
        self.API_KEY = os.getenv("API_KEY")
        genai.configure(api_key=self.API_KEY)
        self.data_path = open("data/data.json")
        self.data = json.load(self.data_path)
        self.df = pd.DataFrame(self.data)
        self.df.columns = ['Title', 'Text']


    def embed_fn(self,title, text):
        return genai.embed_content(model=self.model,
                             content=text,
                             task_type="retrieval_document",
                             title=title)["embedding"]


    def find_best_passage(self, dataframe):
        self.query_embedding = genai.embed_content(model=self.model,
                                                content=self.query,
                                                task_type="retrieval_query")
        dot_products = np.dot(np.stack(dataframe['Embeddings']), self.query_embedding["embedding"])
        idx = np.argmax(dot_products)
        return dataframe.iloc[idx]['Text'] 

    def make_prompt(self,relevant_passage):
        escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
        prompt = textwrap.dedent("""You are a helpful and informative bot that answers questions using text from the reference passage included below. \
        Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
        However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
        strike a friendly and converstional tone. \
        If the passage is irrelevant to the answer, you may ignore it.
        QUESTION: '{query}'
        PASSAGE: '{relevant_passage}'

            ANSWER:
        """).format(query=self.query, relevant_passage=escaped)

        return prompt

    def answer(self):
        self.df['Embeddings'] = self.df.apply(lambda row: self.embed_fn(row['Title'], row['Text']), axis=1)
        self.passage = self.find_best_passage(self.df)
        self.prompt = self.make_prompt(self.passage)
        self.model = genai.GenerativeModel('gemini-pro')
        self.answer = self.model.generate_content(self.prompt)
        return self.answer.text
    
   


