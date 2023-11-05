Create a virtual environment with venv to avoid conflicts in library versions and modules
```
python -m venv .venv
```
Activate the environment
```
.\.venv\Scripts\activate
```
Install all neccessary libraries with a specific version
```
pip install -r requirements.txt
```
To run the server backend flask python, run this line of command
```
flask --debug run
```
Now, the website should be available at the port `127.0.0.1:5000`

To run the streamlit app, move into the `src` folder
```
cd src
```
Now run the app with this command
```
streamlit run main.py
```