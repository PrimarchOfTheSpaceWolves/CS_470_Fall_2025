import cv2
import numpy as np
import pandas
import sklearn
from torchvision import datasets
import time

from sklearn.neighbors import (KNeighborsClassifier, NearestCentroid)
from sklearn.metrics import (accuracy_score, f1_score, 
                             roc_auc_score, confusion_matrix, 
                             classification_report)
import torch
import torch.nn.functional as F

def compute_metrics(ground, pred, prob, class_cnt):
    accuracy = accuracy_score(ground, pred)
    f1 = f1_score(ground, pred, average="macro")
    
    one_hot_ground = F.one_hot(torch.from_numpy(ground), num_classes=class_cnt).numpy()
    auc = roc_auc_score(one_hot_ground, prob, average="macro", multi_class="ovr")
    
    return {
        "Accuracy": accuracy,
        "F1-score": f1,
        "AUC": auc
    }

def convert_images(data):
    data = data.numpy()
    data = data.astype("float32")
    data /= 255.0
    data -= 0.5
    data *= 2.0
    return data

def display_image(label, image):
    image = cv2.resize(image, dsize=None, fx=5.0, fy=5.0)
    cv2.imshow(label, image)
    cv2.waitKey(-1)
    cv2.destroyAllWindows()

def main():
    
    train_ds = datasets.MNIST(root="./data", train=True, download=True)
    test_ds = datasets.MNIST(root="./data", train=False, download=True)
    
    x_train = convert_images(train_ds.data)
    x_test = convert_images(test_ds.data)
    
    x_train = np.reshape(x_train, (x_train.shape[0], -1))
    x_test = np.reshape(x_test, (x_test.shape[0], -1))
    
    y_train = train_ds.targets.numpy()
    y_test = test_ds.targets.numpy()
    
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    
    #print(x_train[0].shape)
    #display_image("IMAGE", x_train[0])
    
    start_time = time.time()
    #classifier = KNeighborsClassifier(n_neighbors=3, 
    #                                  weights="uniform", 
    #                                  algorithm="auto",
    #                                  n_jobs=None)
    classifier = NearestCentroid()
    classifier.fit(x_train, y_train)
    end_time = time.time()
    print("Training done:", (end_time-start_time), "seconds")
    
    start_time = time.time()
    pred_train = classifier.predict(x_train)
    pred_test = classifier.predict(x_test)
    end_time = time.time()
    print("Prediction done:", (end_time-start_time), "seconds")
    
    prob_train = classifier.predict_proba(x_train)
    prob_test = classifier.predict_proba(x_test)
    
    print(pred_train.shape)
    print(prob_train.shape)
    
    class_cnt = 10
    train_metrics = compute_metrics(y_train, pred_train, prob_train, class_cnt)
    test_metrics = compute_metrics(y_test, pred_test, prob_test, class_cnt)
    
    print("TRAINING:")
    print(train_metrics)
    print("TESTING:")
    print(test_metrics)
    
    print("CONFUSION MATRIX (train):")
    print(confusion_matrix(y_train, pred_train))
    
    print("CLASSIFICATION REPORT (train):")
    print(classification_report(y_train, pred_train))
    
    
    
    

if __name__ == "__main__":
    main()
    