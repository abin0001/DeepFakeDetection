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
load_dotenv()
API_KEY = os.getenv("API_KEY")
genai.configure(api_key=API_KEY)
data_path = open("data/data.json")

data = json.load(data_path)


model = 'models/embedding-001'

df = pd.DataFrame(data)
df.columns = ['Title', 'Text']


query = "Can you tell me five points the Team LiveWires"

def embed_fn(title, text):
  return genai.embed_content(model=model,
                             content=text,
                             task_type="retrieval_document",
                             title=title)["embedding"]

df['Embeddings'] = df.apply(lambda row: embed_fn(row['Title'], row['Text']), axis=1)

def find_best_passage(query, dataframe):
  query_embedding = genai.embed_content(model=model,
                                        content=query,
                                        task_type="retrieval_query")
  dot_products = np.dot(np.stack(dataframe['Embeddings']), query_embedding["embedding"])
  idx = np.argmax(dot_products)
  return dataframe.iloc[idx]['Text'] 

passage = find_best_passage(query, df)

def make_prompt(query, relevant_passage):
  escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
  prompt = textwrap.dedent("""You are a helpful and informative bot that answers questions using text from the reference passage included below. \
  Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
  However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
  strike a friendly and converstional tone. \
  If the passage is irrelevant to the answer, you may ignore it.
  QUESTION: '{query}'
  PASSAGE: '{relevant_passage}'

    ANSWER:
  """).format(query=query, relevant_passage=escaped)

  return prompt


prompt = make_prompt(query, passage)



model = genai.GenerativeModel('gemini-pro')
answer = model.generate_content(prompt)

print(answer.text)