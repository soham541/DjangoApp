import numpy as np
from tensorflow import keras
from django.shortcuts import render
from keras.preprocessing import image

from keras.applications.resnet import ResNet50
from keras.applications.resnet import preprocess_input, decode_predictions

from .forms import ImageUploadForm


# Create your views here.
# Stores image to server
def handle_uploaded_file(f):
    with open('img.jpg', 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)


# def home(request):
#     return render(request, 'home.html')


def imageUpload(request):
    return render(request, 'uploadImage.html')


def imageprocess(request):
    form = ImageUploadForm(request.POST, request.FILES)
    if form.is_valid():
        handle_uploaded_file(request.FILES['image'])
        model = ResNet50(weights='imagenet')
        path = 'img.jpg'

        img = keras.utils.load_img(path, target_size=(224, 224))
        x = keras.utils.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = model.predict(x)
        print('Predicted:', decode_predictions(preds, top=3)[0])

        html = decode_predictions(preds, top=3)[0]
        res = []
        for e in html:
            res.append((e[1], np.round(e[2] * 100, 2)))  # Species with percentage

        return render(request, 'result.html', {'res': res})
    return render(request, 'uploadImage.html')
