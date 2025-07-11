import os
import pandas as pd
import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI

last_discussed_course = None

# Loading environment
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Loading and preprocessing data
df = pd.read_csv("udemy_courses.csv")
df = df[['course_title', 'url', 'is_paid', 'price', 'num_subscribers',
         'num_reviews', 'num_lectures', 'level', 'content_duration']]
df['content_duration'] = pd.to_numeric(df['content_duration'], errors='coerce')
df.dropna(subset=['content_duration'], inplace=True)
df['text'] = df['course_title']

# Embedding functions
def get_embedding(text, model="text-embedding-ada-002"):
    text = text.strip().replace("\n", " ")
    if not text:
        raise ValueError("Input text is empty or invalid for embedding.")
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

def get_batch_embeddings(texts, model="text-embedding-ada-002"):
    response = client.embeddings.create(input=texts, model=model)
    return [d.embedding for d in response.data]

# Loading/creating embeddings
if 'embedding' not in df.columns:
    print("\nCreating embeddings... This may take a few minutes.")
    batch_size = 100
    all_embeddings = []
    for i in range(0, len(df), batch_size):
        batch = df['text'].iloc[i:i+batch_size].tolist()
        embeddings = get_batch_embeddings(batch)
        all_embeddings.extend(embeddings)
    df['embedding'] = all_embeddings
    df.to_csv("udemy_courses_with_embeddings.csv", index=False)
else:
    df = pd.read_csv("udemy_courses_with_embeddings.csv", converters={"embedding": eval})

embedding_matrix = np.array(df['embedding'].tolist()).astype('float32')
index = faiss.IndexFlatL2(embedding_matrix.shape[1])
index.add(embedding_matrix)

def get_referenced_course_index(user_input, top_courses):
    titles = [course['course_title'] for course in top_courses]
    joined_titles = "\n".join([f"{i+1}. {title}" for i, title in enumerate(titles)])
    prompt = f"""
You previously recommended these 3 courses:

{joined_titles}

The user now said: "{user_input}"

Which course are they referring to? Answer ONLY with 1, 2, 3, or "None" if it's unclear.
"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content.strip()
    return int(answer) - 1 if answer in ["1", "2", "3"] else None

print("\nWelcome to the Udemy Course Recommender!")

top_courses = None

while True:
    user_input = input("\nWhat do you want to learn today? (or type 'exit'):\n").strip()
    if user_input.lower() == 'exit':
        print("Goodbye!")
        break

    if not top_courses:
        if not user_input.strip():
            print(" Please enter something meaningful.")
            continue

        query_vec = get_embedding(user_input)
        query_vec = np.array([query_vec]).astype('float32')
        D, I = index.search(query_vec, k=3)

        top_courses = df.iloc[I[0]].to_dict(orient='records')

        print("\nTop 3 matching courses:\n")
        for i, course in enumerate(top_courses, start=1):
            print(f"{i}. {course['course_title']}")
            print(f"   {course['content_duration']} hrs |  {'Free' if not course['is_paid'] else '$'+str(course['price'])}")
            print(f"   {course['url']}\n")
        print("You can ask any questions about the course")
        continue

    idx = get_referenced_course_index(user_input, top_courses)
    if idx is not None and 0 <= idx < len(top_courses):
        course = top_courses[idx]
        last_discussed_course = course  #  remembering the course
        system_prompt = "You are a helpful assistant that explains online courses to users."
        user_prompt = (
            f"Explain this course in detail for a beginner:\n"
            f"Title: {course['course_title']}\n"
            f"Level: {course['level']}\n"
            f"Duration: {course['content_duration']} hours\n"
            f"Price: {'Free' if not course['is_paid'] else '$' + str(course['price'])}\n"
            f"Subscribers: {course['num_subscribers']}\n"
            f"Reviews: {course['num_reviews']}\n"
            f"Lectures: {course['num_lectures']}\n"
        )
        chat = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        print("\n", chat.choices[0].message.content)
    else:
        if last_discussed_course:
            # for handling follow-up questions based on the selected course
            system_prompt = "You are a helpful assistant that answers specific questions about a course."
            user_prompt = (
                f"Here are the course details:\n"
                f"Title: {last_discussed_course['course_title']}\n"
                f"Level: {last_discussed_course['level']}\n"
                f"Duration: {last_discussed_course['content_duration']} hours\n"
                f"Price: {'Free' if not last_discussed_course['is_paid'] else '$' + str(last_discussed_course['price'])}\n"
                f"Subscribers: {last_discussed_course['num_subscribers']}\n"
                f"Reviews: {last_discussed_course['num_reviews']}\n"
                f"Lectures: {last_discussed_course['num_lectures']}\n\n"
                f"The user asked: \"{user_input}\"\n"
                f"Answer based on the course above."
            )
            chat = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            print("\n", chat.choices[0].message.content)
        else:
            # No course referenced or remembered — fallback question?
            print("Let me check on that...")
            context = "\n".join([f"{i+1}. {c['course_title']}" for i, c in enumerate(top_courses)])
            fallback_prompt = f"""
Here are the top 3 recommended courses:

{context}

User asked: "{user_input}"

Based on this, respond helpfully, even if they didn’t reference a course directly.
"""
            chat = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": fallback_prompt}]
            )
            print("\n", chat.choices[0].message.content)
