import requests

url = "http://127.0.0.1:8000/query"

def test_request():
    question = "Can I get a green card if I'm married to another green card holder?"
    response = requests.get(url, params={"text": question})

    if response.status_code == 200:
        # Print the returned data
        print("Response:", response.json())
    else:
        print("Failed to retrieve data:", response.status_code)
