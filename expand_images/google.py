import os
os.system('export http_proxy=http://localhost:8123')
os.system('export https_proxy=http://localhost:8123')

from google_images_download import google_images_download   #importing the library

response = google_images_download.googleimagesdownload()   #class instantiation

arguments = {
    "keywords":"浆水鱼鱼",
    "limit":20,
    "print_urls":True, 
    "language":"Chinese (Simplified)", 
    "output_directory":"./data/expand"
    }   #creating list of arguments
paths = response.download(arguments)   #passing the arguments to the function
print(paths)   #printing absolute paths of the downloaded images