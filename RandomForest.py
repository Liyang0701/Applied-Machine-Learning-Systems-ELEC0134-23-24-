#Import required libraries
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def RFRun():
    # Load the dataset
    data = np.load('./Datasets/pathmnist.npz')

    x_train = data['train_images']
    y_train = data['train_labels'].ravel()
    x_val = data['val_images']
    y_val = data['val_labels'].ravel()
    x_test = data['test_images']
    y_test = data['test_labels'].ravel()

    # Preprocess Data，normalizing and flattening the images
    scaler = StandardScaler()
    x_train_flattened = x_train.reshape(x_train.shape[0], -1)  # Flattening the images
    x_train_normalized = scaler.fit_transform(x_train_flattened)  # Normalizing the pixel values
    x_val_flattened = x_val.reshape(x_val.shape[0], -1)
    x_val_normalized = scaler.transform(x_val_flattened)
    x_test_flattened = x_test.reshape(x_test.shape[0], -1)
    x_test_normalized = scaler.transform(x_test_flattened)

    # Dimensionality Reduction with PCA
    pca = PCA(n_components=0.95)  # Adjust the n_components as needed
    x_train_pca = pca.fit_transform(x_train_normalized)
    x_val_pca = pca.transform(x_val_normalized)
    x_test_pca = pca.transform(x_test_normalized)

    # Training the RandomForest Model with initial hyperparameters
    RF_model = RandomForestClassifier(n_estimators=100, random_state=40, n_jobs=-1)
    RF_model.fit(x_train_pca, y_train)

    # Evaluating the initial model on the validation set
    y_val_pred = RF_model.predict(x_val_pca)
    initial_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Initial Validation Accuracy: {initial_accuracy}")

    # Hyperparameter Tuning using Randomized Search Cross-Validation
    combinations = {
        'n_estimators': [30, 50, 80, 100],         # the number of trees in the forest
        'max_depth': [10, 20, 30],                 # the maximum depth of each tree in the forest.
        'min_samples_split': [50, 100, 300]        # the minimum number of samples required to split an internal node in a tree
    }
    random_search = RandomizedSearchCV(RF_model, combinations, n_iter=10, cv=5, n_jobs=-1, verbose=2, random_state=40)
    random_search.fit(x_train_pca, y_train)

    # Best parameters 
    best_params = random_search.best_params_
    best_RF_model = random_search.best_estimator_
    print(f"Best Parameters: {best_params}")

    # Final Model Evaluation on the Test Set
    y_test_pred = best_RF_model.predict(x_test_pca)
    final_accuracy = accuracy_score(y_test, y_test_pred)
    classification_rep = classification_report(y_test, y_test_pred)
    conf_matrix= confusion_matrix(y_test, y_test_pred)
    print(f"Test Accuracy: {final_accuracy}")
    print(f"Classification Report on Test Set:\n{classification_rep}")
    print(f"Confusion Matrix on Test Set:\n{conf_matrix}")

def Runall():
    RFRun()

if __name__ == "__main__":
    Runall()
