import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from sklearn.metrics.pairwise import cosine_similarity
import os
import pandas as pd

# تحميل وحفظ الميزات
import pickle
features_file = 'features.pkl'

def save_features(features, filename):
    with open(filename, 'wb') as f:
        pickle.dump(features, f)

def load_features(filename):
    with open(filename, 'rb') as f:
        features = pickle.load(f)
    return features


# تحميل النموذج المدرب ResNet50
base_model = ResNet50(weights='imagenet', include_top=False)
model = Model(inputs=base_model.input, outputs=GlobalAveragePooling2D()(base_model.output))


# تحضير الصورة وتجهيزها
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


# استخراج الميزات من الصورة
def extract_features(img_path):
    img_array = preprocess_image(img_path)
    features = model.predict(img_array)
    return features


# حساب التشابه بين الميزات
def compute_similarity(feature1, feature2):
    similarity = cosine_similarity(feature1.reshape(1, -1), feature2.reshape(1, -1))
    return similarity[0][0]


# الحصول على الميزات للصور
def get_images_features():
    if os.path.exists(features_file):
        return load_features(features_file)
    else:
        # تحميل ملف CSV الذي يحتوي على مسارات الصور
        dataset_path = "./product_paths.csv"  # استبدل بمسار ملف CSV الخاص بك
        data = pd.read_csv(dataset_path)

        image_features = {}
        for index, row in data.iterrows():
        # for imageName in os.listdir('./dataset'):
        #     imagePath='./dataset/'+imageName
        #     features = extract_features(imagePath)  
            img_path = row['path']
            features = extract_features('../server/'+img_path+'.jpg')  #استخراج الصفات من الصور
            image_features[img_path] = features

        # pkl حفظ الميزات في ملف
        save_features(image_features, features_file)

        return image_features


# الحصول على التشابهات
def get_similarities(input_image_features):
    similarities = {}
    # اكير عدد هو 15
    for img_name, features in get_images_features().items():
        if len(similarities) >= 15:
            break
        similarity = compute_similarity(input_image_features, features)
        #يجب ان تكون نسبة التشابه 0.65
        
        if similarity > 0.65:
            similarities[img_name] = similarity
   
    sorted_images = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return sorted_images    


# الدالة الرئيسية
def go(image_url):
    input_image_features = extract_features(image_url)
    res = get_similarities(input_image_features)
    similarities = [[img[0], img[1]] for img in res]
    print(similarities)
    return similarities


# تنفيذ الدالة الرئيسية
go('C:/Users/sami/Desktop/projects/ai-commerce/server/data/65fe04e4ed78d3923631a00b/37de7453-8870-42c3-a9b2-ec28a07ad972.jpg')