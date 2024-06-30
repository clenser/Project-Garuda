import requests
url = "https://api.callchimp.ai/v1/calls"  
data = {
    "lead": "837496"
}

headers = {
    "content-type": "application/json",
    "x-api-key":'uWCv5aSv.ptSy0eQc4ZXLdfBVOURMDN8UBxwe66Eo'
}

response = requests.post(url, json=data,headers=headers)
# Check the response status code
if response.status_code == 200:
    print("Request successful!")
    print(response.json())  # Parse and print the JSON response
else:
    print("Request failed with status code:", response.status_code)
