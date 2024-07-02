import requests
from PIL import Image
import io

# Path to the image file
image_path = 'datasets/processed_data/images/train/vid_4_600.jpg'

# URL of the API endpoint
url = 'http://127.0.0.1:5000/predict'

# Open the image file
with open(image_path, 'rb') as image_file:
    response = requests.post(url, files={'file': image_file})

# Check the response
if response.status_code == 200:
    img = Image.open(io.BytesIO(response.content))
    img.save('test.jpg')
else:
    print("Error:", response.status_code, response.text)
