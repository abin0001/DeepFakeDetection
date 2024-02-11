from Elvy import Elvy
from flask import Flask,render_template,request, redirect, url_for, session,jsonify

app = Flask(__name__)
app.secret_key = 'ELVY_1.0' 

@app.route('/')
def chat():

    return render_template('bot.html')
    
@app.route("/get", methods=["GET", "POST"])
def chatbot():
    if request.method == "POST":
        prompt = request.form["msg"]
        elvy=Elvy(prompt)
        response =elvy.answer()
        return response
    
if __name__ == '__main__':

	app.run(debug=True)