# app.py

import uuid
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
import os
import json
from azure.core.exceptions import ResourceExistsError


load_dotenv()

id1 = uuid.uuid4()
print(f'UUID 1: {id1}')
class AzureProfiles():
    def __init__(self):
        CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        CONTAINER_NAME = 'user-profiles'        
        self.blob_service_client = BlobServiceClient.from_connection_string(
            CONNECTION_STRING
        )        
        self.container_client = self.blob_service_client.get_container_client(
            CONTAINER_NAME
        )
        
    def getUserId(self,userName, blob_client=None):
        if not blob_client:
            blob_client = self.container_client.get_blob_client('userList.json')    
        downloader = blob_client.download_blob()    
        content = downloader.readall()
        userIds = json.loads(content.decode("utf-8"))
        if userName not in userIds.keys():
            print(f"""'user name {userName} not found. \n stored usernames are: {userIds} """)
            return None, userIds
        else:
            self.user_id = userIds[userName]
            return userIds[userName], userIds
        
        
    
    def initUserProfile(self, user_name):
        bucketName = str(uuid.uuid4())
        blob_client = self.container_client.get_blob_client(
            bucketName +"/.keep"
        )
        
        blob_client.upload_blob(
            b"",
            overwrite=True
        )
        _, userIds = self.getUserId(user_name)
        userIds[user_name] = bucketName

        # with open('userList.json', "w", encoding="utf-8") as f:
        #     f.write(str(userIds))
        
        self.upload(userIds,'userList.json')
        print("New user id: " + bucketName + " added")
        
    
    # UPLOAD ROUTE
    # @app.route("/upload", methods=["POST"])

    def upload(self, userIds, outputFile): 
    
        blob_client = self.container_client.get_blob_client(outputFile)
    
        blob_client.upload_blob(
            json.dumps(userIds),
            overwrite=True
        )
    
        print(f"""uploaded the file: {outputFile}""")

    def addLogs(self, entry, outputFile):
    
        blob_client = self.container_client.get_blob_client(outputFile)
    
        if not blob_client.exists():
            print('creating log file')
            blob_client.create_append_blob()
            
        
        blob_client.append_block(entry.encode('utf--8'))
    
        print(f"added info: {entry}")
        
    # LIST FILES
    
    # @app.route("/files")
    def files(self, directory):
    
        blobs = self.container_client.list_blobs(name_starts_with=directory)
    
        results = []
    
        for blob in blobs:
            results.append(blob.name)            
    
        return results
    
    
    
    # DOWNLOAD FILE
    # @app.route("/download/<filename>")
    def download(self, filename):
    
        blob_client = self.container_client.get_blob_client(filename)
    
        downloader = blob_client.download_blob()
    
        content = downloader.readall()
    
        return content


