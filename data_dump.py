from APS_FAULT_DETECTION.config import mongo_client
import pandas as pd
import json

FILE_PATH = 'aps_data.csv'
mongo_client = mongo_client
Database_name = 'aps'
collection_name = 'sensor'

if __name__=="__main__":
    df = pd.read_csv(FILE_PATH)
    print(f"Rows and columns: {df.shape}")

    #Convert dataframe to json so that we can dump these record in mongo db
    df.reset_index(drop=True,inplace=True)

    json_record = list(json.loads(df.T.to_json()).values())
    print(json_record[0])
    #insert converted json record to mongo db
    mongo_client[Database_name][collection_name].insert_many(json_record)

