import pandas as pd

# Based off of https://stackoverflow.com/questions/16249736/how-to-import-data-from-mongodb-to-pandas
def read_mongo(db, collection, query={}, host='localhost', port=27017, username=None, password=None, no_id=True):
    try:
        from pymongo import MongoClient
    except ImportError as error:
        print('pymongo does not appear to be installed')
        return None

    if username and password:
        mongo_uri = 'mongodb://%s:%s@%s:%s/%s' % (username, password, host, port, db)
        conn = MongoClient(mongo_uri)
    else:
        conn = MongoClient(host, port)
    db = conn[db]

    # Make a query to the specific DB and Collection
    cursor = db[collection].find(query)

    # Expand the cursor and construct the DataFrame
    df = pd.DataFrame(list(cursor))

    # Delete the _id
    if no_id:
        del df['_id']

    return df