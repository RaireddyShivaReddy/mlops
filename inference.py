import pickle
from flask import Flask, request

app = Flask(__name__)
@app.route("/")

def console():
    filename = "data.pkl"
    loaded_model = pickle.load(open(filename, 'rb'))
    output = loaded_model
    return (output)

if __name__ == '__main__':
    print("**Starting Server...")
app.run(host='0.0.0.0',port=80)
