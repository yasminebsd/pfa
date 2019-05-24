import os

from PIL import Image, ImageOps

from test import binarize

import numpy as np
from keras.engine.saving import load_model
from scipy.io import loadmat
from yaspin import yaspin

import settings
from display_utils import groundTruthDifferenceVisualise, display_confusion_matrix, progress
from preprocess import prepareData, PATCH_SIZE
from preprocess_utils import normalize, create_patch
from train import train

def getParams(choice):
    filename = 'diff.jpg'
    image_key = 'landsat'
    gt_filename = 'cleanchangemap.jpg'
    gt_key = 'changemap'
    settings.dataSetName = 'landsat'
    settings.classes = ['change','no change']

    settings.modelFile = 'models/' + settings.dataSetName + '.h5'

    return './data/' + filename, image_key, './data/' + gt_filename, gt_key


def get_choice():
    print("+----------------- MENU -----------------+")
    print("+ choisir le dataset :                   +")
    print("+ 1 - Indian Pines                       +")
    print("+ 2 - Salinas                            +")
    print("+ 3 - Pavia Centre                       +")
    print("+ 4 - Pavia University                   +")
    print("+----------------------------------------+")

    while True:
        try:
            choice = int(input("Choix : "))
        except ValueError:
            print("Erreur : SVP, il faut choisir un entier")
            continue
        if choice in range(1, 5):
            break
        print("Erreur : SVP, il faut choisir un entier entre 1 et 4")

    return choice


def get_nb_epochs():
    while True:
        try:
            epochs = int(input("Nombre des epochs (nombre >= 20) : "))
        except ValueError:
            print("Erreur : SVP, il faut choisir un entier > 20")
            continue
        if epochs >= 0:
            break
        print("Erreur : SVP, il faut choisir un entier > 20")
    return epochs


def ask_save_model():
    response = input("Sauvgarder le model [O/N] ? : ")
    if response.upper() == "O":
        return True
    return False


def ask_train():
    response = input("Model deja sauvegardé , relancer le train [O/N] ? : ")
    if response.upper() == "O":
        return True
    return False


def wanna_train():
    exists = os.path.isfile(settings.modelFile)
    if exists:
        return ask_train()
    else:
        return True


def save_model(model):
    wana_save_model = ask_save_model()
    if wana_save_model:
        model.save(settings.modelFile)
        print("Model sauvegardé avec succès : " + settings.modelFile)


def predict(model, img):
    img = np.array(normalize(img))

    image_width = img.shape[0]
    image_height = img.shape[1]
    img = img.reshape((image_width, image_height,1))
    img = np.transpose(img, (2, 0, 1))
    new_image = []

    for i in range(img.shape[0]):
        p = np.pad(img[i, :, :], int((PATCH_SIZE - 1) / 2), 'constant', constant_values=0)
        new_image.append(p)

    new_image = np.array(new_image)

    predicted_model = np.zeros((image_width, image_height), dtype=np.int)

    for i in range(image_width):
        for j in range(image_height):
            patch = create_patch(new_image, i, j, PATCH_SIZE)

            if patch.shape[1] == PATCH_SIZE and patch.shape[2] == PATCH_SIZE:
                inputPred = np.array([patch])
                inputPred = np.transpose(inputPred, (0, 2, 3, 1))
                prediction = model.predict_classes(inputPred)
                predicted_model[i, j] = np.round(prediction[0]) + 1
        progress(i + 1, image_width)

    return predicted_model


# def main(filename, image_key, gt_filename, gt_key):
def main():
    model = None
    # image_dict = loadmat(filename)
    # img = np.array(image_dict[image_key])
    #
    # gt_image_dict = loadmat(gt_filename)
    # img_labels = np.array(gt_image_dict[gt_key])
    #
    filename = './data/diff.jpg'
    img = np.array(Image.open(filename))

    gt_fileName = './data/cleanchangemap.jpg'
    im = Image.open((gt_fileName))
    changedmap = ImageOps.expand(im, border=2)
    image_labels = np.array(changedmap)

    img_labels = binarize(image_labels)

    # CRISP
    if wanna_train():
        with yaspin(text="Preprocessing", color="yellow") as spinner:
            # preprocessing
            x_train, y_train, x_test, y_test, nb_classes = prepareData(img, img_labels)
            spinner.ok("✅ ")

        # train
        epochs = get_nb_epochs()
        model = train(x_train, y_train, x_test, y_test, nb_classes, epochs)
        save_model(model)
    else:
        model = load_model(settings.modelFile)

    predicted_labels = predict(model, img)
    groundTruthDifferenceVisualise(img_labels, predicted_labels)

    for i in range(img_labels.shape[0]):
        for j in range(img_labels.shape[1]):
            if img_labels[i, j] == 0:
                predicted_labels[i, j] = 0

    groundTruthDifferenceVisualise(img_labels, predicted_labels)
    display_confusion_matrix(img_labels, predicted_labels)


if __name__ == '__main__':
    # os.environ["KERAS_BACKEND"] = "plaidml"

    # choice = get_choice()
    # filename, image_key, gt_filename, gt_key = getParams(choice)
    # main(filename, image_key, gt_filename, gt_key)
    settings.dataSetName = 'landsat'
    settings.classes = ['unchanged', 'changed']

    settings.modelFile = 'models/' + settings.dataSetName + '.h5'
    main()
