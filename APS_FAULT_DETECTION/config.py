import pymongo
import os

class EnvironmentVariable:
    def __init__(self):
        self.MONGODB_URL:str = os.getenv("MONGODB_URL")

environment_variable = EnvironmentVariable()
mongo_client = pymongo.MongoClient(environment_variable.MONGODB_URL)