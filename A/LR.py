import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# create a funtion to evaluate the model
def evaluate(C, tol, x_train, y_train, x_val, y_val):
    model = LogisticRegression(solver='liblinear', C=C, max_iter=1000, tol=tol, class_weight='balanced')
    model.fit(x_train, y_train)
    y_pred_val = model.predict(x_val)
    accuracy = accuracy_score(y_val, y_pred_val)
    classification_rep = classification_report(y_val, y_pred_val)
    return accuracy, classification_rep
    
def LR_RUN():  
    # Load the dataset
    data = np.load('./Datasets/pneumoniamnist.npz')
    # Extracting the data
    train_images = data['train_images'] 
    val_images = data['val_images']
    test_images = data['test_images']
    y_train = train_labels = data['train_labels'].ravel()
    y_val = val_labels = data['val_labels'].ravel()
    y_test = test_labels = data['test_labels'].ravel()

    # Reshape and normalize the images
    x_train = train_images.reshape(train_images.shape[0], -1) / 255.0
    x_val = val_images.reshape(val_images.shape[0], -1) / 255.0
    x_test = test_images.reshape(test_images.shape[0], -1) / 255.0

    # Initialize the Logistic Regression model
    logreg = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, tol=0.0001, class_weight='balanced')

    # Manually testing different hyperparameters
    combinations = [
        (0.1, 0.0001),
        (1, 0.0001),
        (10, 0.0001),
        (0.1, 0.001),
        (1, 0.001),
        (10, 0.001),
        (0.1, 0.01),
        (1, 0.01),
        (10, 0.01),
        (0.1, 0.1),
        (1, 0.1),
        (10, 0.1),
    ]

    # Initialize variables to store the best parameters and the highest accuracy
    best_accuracy = 0
    best_params = {'C': None, 'tol': None}

    # Evaluate each combination on the validation set
    for C, tol in combinations:
        accuracy, report = evaluate(C, tol, x_train, y_train, x_val, y_val)
        print(f"Parameters: C={C}, tol={tol}")
        print(f"Validation Accuracy: {accuracy}")
        print(f"Validation Classification Report:\n{report}\n")
        
        # Update the best parameters if current accuracy is higher
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = {'C': C, 'tol': tol}
            
    print(f"best_params = {best_params}")    

    # Produce the final model, fit the model and make predictions
    final_model = LogisticRegression(C=best_params['C'], tol=best_params['tol'],solver='liblinear', max_iter=10000, class_weight='balanced')
    final_model.fit(x_train, y_train)
    y_pred = final_model.predict(x_test)

    # Evaluation metrics
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    print("Test Confusion Matrix:\n", conf_matrix)
    print("Test Classification Report:\n", class_report)
    print("Test Accuracy Score:", accuracy)

def Runall():
    LR_RUN()

if __name__ == "__main__":
    Runall()