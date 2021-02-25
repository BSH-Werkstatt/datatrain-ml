from pymongo import MongoClient
import json
import os
import urllib
import pprint

username = "db_username"
password = "db_password"
db_name = "db_name"
host = "localhost"
port = "27017"

client = MongoClient('mongodb://%s:%s@%s:%s/%s' % (username, password, host, port, db_name))


db = client["datatrain"]

images = db["images"]
campaigns = db["campaigns"]
users = db["users"]

current_path = os.path.dirname(os.path.abspath(__file__))
with open(current_path + '/samples/images.json') as f:
    file_data = json.load(f)
    images.insert_many(file_data)  

with open(current_path + '/samples/campaigns.json') as f:
    file_data = json.load(f)
    campaigns.insert_many(file_data) 

with open(current_path + '/samples/users.json') as f:
    file_data = json.load(f)
    users.insert_many(file_data) 

print("Inserted sample data to the database.")

image_info = db.get_collection('images')
print("Here is the demo data for in the images collection: ")
pprint.pprint(image_info.find_one())

client.close()