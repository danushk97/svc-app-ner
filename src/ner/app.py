from flask import Flask, request
from nemo.collections.nlp.models import TokenClassificationModel


app = Flask(__name__)
model = TokenClassificationModel.from_pretrained("ner_en_bert")
# model.cfg['dataset']['num_workers'] = 0


@app.post('/ner')
def ner():
    body = request.json
    result = model.add_predictions(body['inputs'])

    return {
        'entities': result
    }


if __name__ == '__main__':
    app.run(port=6000)
