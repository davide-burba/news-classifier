import json
import pandas as pd

model_file = '/opt/ml/model.p'
model = pd.read_pickle(model_file)


def lambda_handler(event, context):
    body = json.loads(event["body"])
    text =  body["text"]
    prediction = model.inference(text)

    return {
        'statusCode': 200,
        'body': json.dumps(prediction)
    }
