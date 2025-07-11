# AI-course-Recommender

AI-Powered Udemy Course Recommender
This is an intelligent Udemy course recommendation system that uses OpenAI embeddings, vector similarity search (via FAISS), and GPT-3.5-turbo to recommend and describe courses based on natural language queries.

 -> Features
 Semantic search: Understands user queries and finds top matching Udemy courses.

 Embeddings: Uses OpenAI's text-embedding-ada-002 to generate vector representations.

 Fast similarity matching: Employs FAISS for efficient vector search.

 Natural language explanation: Uses GPT-3.5-turbo to explain course details and respond to follow-up questions.

 Conversational: Remembers the last discussed course and supports contextual questions.

-> Project Structure
bash
Copy
Edit

-> AI-Course-Recommender/
├── udemy_courses.csv                 # Raw course data
├── udemy_courses_with_embeddings.csv# (Generated) Courses with embeddings
├── main.py                          # Main recommendation script
├── .env                             # Contains your OpenAI API key
└── README.md                        # Project documentation

-> Setup Instructions
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/yourusername/ai-course-recommender.git
cd ai-course-recommender
2. Install Dependencies
bash
Copy
Edit
pip install openai python-dotenv pandas numpy faiss-cpu
Use faiss-gpu if you're working with GPU acceleration.

3. Prepare Your .env File
Create a .env file and add your OpenAI API key:

ini
Copy
Edit
OPENAI_API_KEY=your_openai_api_key_here
4. Add Course Data
Place a file named udemy_courses.csv in the root directory. It should have the following columns:

course_title

url

is_paid

price

num_subscribers

num_reviews

num_lectures

level

content_duration

You can download this from Udemy or generate sample data if needed.

 -> Running the App
bash
Copy
Edit
python main.py
Then follow the prompt:

rust
Copy
Edit
What do you want to learn today? (or type 'exit'):
> machine learning for beginners

-> How It Works
Embeddings: It generates (or loads) vector embeddings of course titles using OpenAI's embedding model.

Similarity Search: Uses FAISS to find top 3 most relevant courses to the user's input.

GPT Integration:

Explains course details conversationally.

Handles follow-up or clarification questions contextually.

Clarifies user references to previously listed courses.

-> Technologies Used
OpenAI GPT-3.5-turbo

OpenAI Embeddings

FAISS (Facebook AI Similarity Search)

Pandas

NumPy

Python Dotenv

->Disclaimer
This project uses the OpenAI API, which may incur costs. Be sure to monitor your usage on the OpenAI Dashboard.

-> Contact
Julia Grace Muddada
Graduate Student | AI & Software Developer
juliagrace424@gmail.com
LinkedIn- www.linkedin.com/in/julia-grace-muddada-6708271b3
