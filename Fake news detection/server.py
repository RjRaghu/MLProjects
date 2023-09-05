from flask import Flask, render_template, request
from model_testing import manual_testing


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        news_text = request.form['news_input']
        # print(news_text)
        prediction = manual_testing(news_text)
        result = 'The news is Fake' if prediction == 0 else 'The news is Real'
        return render_template('index.html', result=result)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)

