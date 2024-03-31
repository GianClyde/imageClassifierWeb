from django.shortcuts import render, redirect, HttpResponse
from .forms import UploadForm
import cv2
import imghdr
import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.metrics import Precision, Recall, CategoricalAccuracy
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from django.http import JsonResponse
import io
import base64
from io import BytesIO
import time
from django.conf import settings
from django.core.files.storage import default_storage
import matplotlib
matplotlib.use('agg') 

def home(request):

    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            instance = form.save()
            image_path = instance.image
            image_path_url = instance.image.url
            # Encode the image_path properly
            request.session['image_path_url'] = image_path_url
            return redirect('classify', image_path=image_path)
    else:
        form = UploadForm()
        
    context = {'form': form}
    return render(request, 'main/home.html', context)

def classify(request, image_path):
    image = image_path
    context = {'image':image}
    return render(request, 'main/classify.html', context)


def bridge(request, image_path):
    

     # Construct the absolute file path
    image_full_path = os.path.join(settings.MEDIA_ROOT, image_path)

    
    # Data preprocessing
    data_dir = 'data'
    image_size = (128, 128)
    batch_size = 32

    train_data = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=image_size,
        batch_size=batch_size,
    )

    val_data = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=image_size,
        batch_size=batch_size,
    )

    # Use a pre-trained base model
    base_model = tf.keras.applications.MobileNetV2(input_shape=(128, 128, 3),
                                                include_top=False,
                                                weights='imagenet')

    # Freeze the pre-trained layers
    base_model.trainable = False

    # Add custom classification head
    inputs = tf.keras.Input(shape=(128, 128, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # Learning rate scheduling
    def lr_scheduler(epoch, lr):
        if epoch % 10 == 0 and epoch != 0:
            return lr * 0.9
        else:
            return lr

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

    # Training
    history = model.fit(train_data,
                        validation_data=val_data,
                        epochs=20,
                        callbacks=[lr_callback])

    # Evaluate the model
    loss, accuracy = model.evaluate(val_data)
    print("Validation Accuracy:", accuracy)

    # Predict on new images
    class_names = train_data.class_names

    # Function to load and preprocess a single image
    def preprocess_image(image_path):
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=image_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        return img_array

    # Sample prediction
    sample_image_path = image_full_path
    preprocessed_img = preprocess_image(sample_image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class = class_names[np.argmax(predictions)]
    print("Predicted Class:", predicted_class)
    
    # Evaluate the model
    loss, accuracy = model.evaluate(val_data)
    print("Validation Accuracy:", accuracy)

    # Function to calculate metrics
    def calculate_metrics(model, data):
        true_labels = []
        predicted_labels = []
        for images, labels in data:
            predictions = model.predict(images)
            predicted_labels.extend(np.argmax(predictions, axis=1))
            true_labels.extend(labels.numpy())
            
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, average='weighted')
        recall = recall_score(true_labels, predicted_labels, average='weighted')
        f1 = f1_score(true_labels, predicted_labels, average='weighted')
        
        return accuracy, precision, recall, f1

    # Calculate metrics
    val_accuracy, val_precision, val_recall, val_f1 = calculate_metrics(model, val_data)

    print("Validation Accuracy:", val_accuracy)
    print("Validation Precision:", val_precision)
    print("Validation Recall:", val_recall)
    print("Validation F1 Score:", val_f1)

    # Extracting accuracy values
    train_accuracy = history.history['accuracy']
    val_accuracy_two = history.history['val_accuracy']

 # Plotting accuracy
    def plot_accuracy(train_accuracy, val_accuracy):
        plt.plot(train_accuracy, label='Training Accuracy')
        plt.plot(val_accuracy, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        
        # Save the plot as a PNG image in memory
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_image_png = buffer.getvalue()
        buffer.close()
        
        # Encode the PNG image data to base64
        plot_image_base64 = base64.b64encode(plot_image_png).decode('utf-8')
        
        return plot_image_base64

    train_accuracy = history.history['accuracy']
    val_accuracy_two = history.history['val_accuracy']

    # Call the plot function asynchronously
    plot_image_png = plot_accuracy(train_accuracy, val_accuracy_two)

 

   

      
    # Store percentage likelihood for each class in variables
    bird_percentage = predictions[0][0] * 100
    cat_percentage = predictions[0][1] * 100
    dog_percentage = predictions[0][2] * 100

    
    if bird_percentage < float(60) and  cat_percentage < float(60) and  dog_percentage < float(60):
        final_predicted_class = "Others"
    else:
        final_predicted_class = predicted_class
    # Print the percentage likelihood for each class
    print(f"Likelihood of Cat: {cat_percentage:.2f}%")
    print(f"Likelihood of Dog: {dog_percentage:.2f}%")
    print(f"Likelihood of Bird: {bird_percentage:.2f}%")
    
    val_accuracy_rnd = round(float(val_accuracy),2)
    val_precision_rnd = round(float(val_precision),2)
    val_recall_rnd = round(float(val_recall),2)
    val_f1_rnd = round(float(val_f1),2)
    
    request.session['accuracy_graph_img'] = plot_image_png
    request.session['val_accuracy'] = float(val_accuracy_rnd) * 100
    request.session['val_precision'] = float(val_precision_rnd) * 100
    request.session['val_recall'] = float(val_recall_rnd) * 100
    request.session['val_f1'] = float(val_f1_rnd) * 100
    request.session['predicted_class'] = final_predicted_class

    #return HttpResponse(f'cat:{cat_percentage} - Dog:{dog_percentage} - Bird:{bird_percentage} - Predicted Class:{final_predicted_class}')

    
    return redirect('results')
    
    
def results(request):
    
    val_accuracy = request.session.get('val_accuracy')
    val_precision = request.session.get('val_precision')
    val_recall = request.session.get('val_recall')
    val_f1 = request.session.get('val_f1')
    predicted_class = request.session.get('predicted_class')
    image_path_url = request.session.get('image_path_url')
    accuracy_graph_img = request.session.get('accuracy_graph_img')
    
    print(accuracy_graph_img)

    context ={'val_accuracy':val_accuracy,
              'val_precision':val_precision,
              'val_recall':val_recall,
              'val_f1':val_f1,
              'predicted_class':predicted_class,
              'image_path_url':image_path_url,
              'accuracy_graph_img':accuracy_graph_img
              }
    return render(request, 'main/result.html', context)