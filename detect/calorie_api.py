from flask import Flask, render_template , request , jsonify
from py_edamam import Edamam

app = Flask(__name__)

# CLASSES = [
#     'apple pie', 'carrot cake', 'breakfast burrito', 'bread pudding', 'baby back ribs',
#     'cheesecake', 'chicken curry', 'chocolate cake', 'chocolate mousse', 'hot dog'
# ]

def get_calories(class_name, volume=100):
    query = str(volume) + " gm " + str(class_name)
    api = Edamam(
        nutrition_appid='92a2c29f',
        nutrition_appkey='caffdf3cb786a101c0a8a74138b15a08',
    )

    calories = api.search_nutrient(query)
    return calories['calories']


@app.route('/' , methods=['POST'])
def calories():
    res = {}
    if request.is_json:
        content = request.get_json()
        #import pdb;pdb.set_trace()
        # print(len(content))
        for i in content:
            
            #print(content)
            #print(content['class_name'])
            #print(content['volume'])
            cal = get_calories(content['class_name'], content['volume'])
            response_class=content['class_name']
            #print(res)
            res[response_class]=cal
    response={
        "status":200,
        "data":res,
        "message":"Calorie count"
    }
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5002', debug=True)


'''
@app.route('/' , methods=['GET'])
def calories():
    res = {}
    if request.is_json:
        content = request.get_json()
        # print(len(content))
        for i in content:
            print(i, content[i])
            cal = get_calories(i, content[i])
            res[i] = cal
    {
    "status": "success",
    "data": # some data here
}
    return res'''