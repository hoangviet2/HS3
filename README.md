# Virtual Therapist

[![> Research Paper](https://img.shields.io/badge/Research%20Paper-blue)](https://github.com/Hackathon-LHP-Team/Virtual-Therapist/blob/main/Virtual%20Therapist.pdf)
[![> Notebook & dataset](https://img.shields.io/badge/Notebook%20Dataset-red)](https://colab.research.google.com/drive/1BHydT5sFQQIgXgVWzGiryCj2Kn9h0GgW?usp=sharing)
[![> Notebook & dataset](https://img.shields.io/badge/Notebook%20Dataset-red)](https://github.com/hoangviet2/HS3/blob/main/Model/Sentimental_task.ipynb)
[![> Pitching slide](https://img.shields.io/badge/Pitching%20slide-black)](https://www.canva.com/design/DAFxhdWUyEc/DF4pJIczCQUfDAFDY-LRbg/edit?utm_content=DAFxhdWUyEc&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)


## Acknowledgement
- Mentor: Hồ Ngọc Lâp
- Team members:
- Hoàng Việt
- Trang Gia Khang
- Nguyễn Quốc Anh
- Trần Nguyễn Bảo Khang
- Phan Lê Tường Bách

## Abstract & introduction

Giới trẻ hiện nay đang ngày càng ít chia sẻ cảm xúc cá nhân ra bên ngoài.

## Approach

The application uses natural language processing and artificial intelligence to interact with users in a conversational manner, and offers a toolbox of features to help them cope with stress, anxiety, depression, and other challenges. The application also integrates mental health assessment tools to monitor the users’ progress and provide

feedback. We assume that technology-based applications can be a viable and scalable alternative to face-to-face mental health services for adolescents. Our solution consists of four main components:

- A general emotion classifier that can categorize the user’s story (diary) into positive, negative, or neutral emotions, based on a deep neural network with bidirectional LSTM (BiLSTM) architecture. We evaluated our solution using a self-scraping dataset of online diaries from various websites. We compared different architectures and models for each component and selected the best ones based on their accuracy and performance using logistic regression.

# What's next
- Expand the dataset to make the AI model more generalized and practical
- Impove the time series for progess record analyis 
- Solve the "cold start" problem of the recommender system

## How to run the code
First you need to clone this repository to your local system. Open terminal and then paste this command line
```
git clone https://github.com/hoangviet2/HS3.git
```
Next move into the cloned directory
```
cd HS3
```
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