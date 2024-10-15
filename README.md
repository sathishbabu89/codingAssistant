# codingAssistant

Setting Up Your AI Chatbot in Visual Studio Code in WINDOWS machine

 Set Up Your Project Folder
---------------------------
Create a new folder for your project, e.g., C:\Projects\CodingAssistant.
Open VS Code and navigate to the project folder using File -> Open Folder or drag and drop the folder into the VS Code window.

Create a Virtual Environment

Open a terminal in VS Code (Terminal -> New Terminal or Ctrl + \ Ctrl + \).

Run the following commands to create and activate a virtual environment:
python -m venv my_env
.\my_env\Scripts\activate

After activating your virtual environment with .\my_env\Scripts\activate, you can follow these steps to install the dependencies from requirements.txt

pip install -r requirements.txt

docker run -d -p 6333:6333 qdrant/qdrant


After the dependencies are installed, you can run your app.py file using Streamlit with this
streamlit run app.py
