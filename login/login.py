import firebase_admin
from firebase_admin import credentials
from firebase_admin import auth
from firebase_admin import firestore

cred = credentials.Certificate("B:\HS3\login\Wfirebase_sdk.json")
firebase_admin.initialize_app(cred)

#Firestore
db = firestore.client()
feeling = input("How do you feel today: ")
doc = input("How is it? ")
doc_ref = db.collection('prob').document('emote')
doc_ref.set({
    feeling: doc
})

'''
#Auth
email = input('Enter your email address: ')
password = input("Enter your password: ")

user = auth.create_user(email = email, password = password)

print("User created: {0}".format(user.uid))
'''
