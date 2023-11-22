from flask import Flask, render_template, request
from PIL import Image
import numpy as np
from tensorflow.keras.applications.vgg19 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import VGG19
import os
from io import BytesIO

app = Flask(__name__)
model = VGG19(weights='imagenet')
#model = load_model('vgg19DBPmodel.h5')

# Create the 'static' directory if it doesn't exist
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/feedback')
def feedback():
    return render_template('feedback.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', prediction="No file selected!")

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', prediction="No file selected!")

    img = Image.open(file)
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=1)[0][0]

    breed_prediction = decoded_predictions[1]
    
    # Replace this with your actual class mapping
    #class_mapping = {
        #0: 'boston_bull', 1: 'dingo', 2: 'pekinese', 3: 'bluetick', 4: 'golden_retriever', 5: 'bedlington_terrier', 6: 'borzoi', 7: 'basenji', 8: 'scottish_deerhound', 9: 'shetland_sheepdog', 
        #10: 'walker_hound', 11: 'maltese_dog', 12: 'norfolk_terrier', 13: 'african_hunting_dog', 14: 'wire-haired_fox_terrier', 15: 'redbone', 16: 'lakeland_terrier', 17: 'boxer', 18: 'doberman', 19: 'otterhound', 
        #20: 'standard_schnauzer', 21: 'irish_water_spaniel', 22: 'black-and-tan_coonhound', 23: 'cairn', 24: 'affenpinscher', 25: 'labrador_retriever', 26: 'ibizan_hound', 27: 'english_setter', 28: 'weimaraner', 29: 'giant_schnauzer', 
        #30: 'groenendael', 31: 'dhole', 32: 'toy_poodle', 33: 'border_terrier', 34: 'tibetan_terrier', 35: 'norwegian_elkhound', 36: 'shih-tzu', 37: 'irish_terrier', 38: 'kuvasz', 39: 'german_shepherd', 
        #40: 'greater_swiss_mountain_dog', 41: 'basset', 42: 'australian_terrier', 43: 'schipperke', 44: 'rhodesian_ridgeback', 45: 'irish_setter', 46: 'appenzeller', 47: 'bloodhound', 48: 'samoyed', 49: 'miniature_schnauzer',
        #50: 'brittany_spaniel', 51: 'kelpie', 52: 'papillon', 53: 'border_collie', 54: 'entlebucher', 55: 'collie', 56: 'malamute', 57: 'welsh_springer_spaniel', 58: 'chihuahua', 59: 'saluki',
        #60: 'pug', 61: 'malinois', 62: 'komondor', 63: 'airedale', 64: 'leonberg', 65: 'mexican_hairless', 66: 'bull_mastiff', 67: 'bernese_mountain_dog', 68: 'american_staffordshire_terrier', 69: 'lhasa',
        #70: 'cardigan', 71: 'italian_greyhound', 72: 'clumber', 73: 'scotch_terrier', 74: 'afghan_hound', 75: 'old_english_sheepdog', 76: 'saint_bernard', 77: 'miniature_pinscher', 78: 'eskimo_dog', 79: 'irish_wolfhound',
        #80: 'brabancon_griffon', 81: 'toy_terrier', 82: 'chow', 83: 'flat-coated_retriever', 84: 'norwich_terrier', 85: 'soft-coated_wheaten_terrier', 86: 'staffordshire_bullterrier', 87: 'english_foxhound', 88: 'gordon_setter', 89: 'siberian_husky',
        #90: 'newfoundland', 91: 'briard', 92: 'chesapeake_bay_retriever', 93: 'dandie_dinmont', 94: 'great_pyrenees', 95: 'beagle', 96: 'vizsla', 97: 'west_highland_white_terrier', 98: 'kerry_blue_terrier', 99: 'whippet',
        #100: 'sealyham_terrier', 101: 'standard_poodle', 102: 'keeshond', 103: 'japanese_spaniel', 104: 'miniature_poodle', 105: 'pomeranian', 106: 'curly-coated_retriever', 107: 'yorkshire_terrier', 108: 'pembroke', 109: 'great_dane',
        #110: 'blenheim_spaniel', 111: 'silky_terrier', 112: 'sussex_spaniel', 113: 'german_short-haired_pointer', 114: 'french_bulldog', 115: 'bouvier_des_flandres',  116: 'tibetan_mastiff', 117: 'english_springer', 118: 'cocker_spaniel', 119: 'rottweiler'
    #}

    #breed_prediction = class_mapping.get(predicted_class_index, "Unknown")
    
    temp_img_path = os.path.join('static', 'temp_img.jpg')
    img.save(temp_img_path)
    return render_template('index.html', prediction=breed_prediction, image_path=temp_img_path)

if __name__ == '__main__':
    app.run(debug=True)