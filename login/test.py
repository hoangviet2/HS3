import pyrebase

firebaseConfig = {
        'apiKey': "AIzaSyCdjchW9VDCwJa38sSmM5GqNu8GuLMmhNE",
        'authDomain': "fir-login-78707.firebaseapp.com",
        'databaseURL': "https://fir-login-78707-default-rtdb.firebaseio.com",
        'projectId': "fir-login-78707",
        'storageBucket': "fir-login-78707.appspot.com",
        'messagingSenderId': "589888925876",
        'appId': "1:589888925876:web:24d05ad6224c5934d567e1"
}

firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()

def signin():
    pass

def signup():
    print("Sign up...")
    email = input("Enter your email address: ")
    password = input("Enter your password: ")
    try:
        user = auth.create_user_with_email_and_password(email, password)
    except:
        print("Email already in use!")
    return 

ans = input("Are you a new user?[y/n]")
if ans == 'n':
    signin()
elif ans == 'y':
    signup()