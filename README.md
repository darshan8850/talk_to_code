# Talk to Code App

This Flask application serves as an interface for interacting with code repositories and generating questions based on code content.

## Setup Instructions

1. Clone the repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Set up a `.env` file with the required environment variables:

   ```plaintext
   OPENAI_API_KEY=<your_openai_api_key>
   LANCEDB_PATH=<path_to_lancedb>
   
4. Run the Flask application using python app.py.
   
### Endpoints

**/load_repo (POST)**:
- Input: JSON object containing programming_language, uuid, and repo_url.
- Output: Clones the repository specified by the repo_url and processes the uploaded files.
  
**/ask_question (POST)**:
- Input: JSON object containing question and uuid.
- Output: Responds with an answer to the provided question based on the code repository.

**/gen_question (POST)**:
- Input: JSON object containing programming_language and uuid.
- Output: Generates questions based on the code repository and the specified programming language.
