from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate_model(model_path, test_generator):
    model = load_model(model_path)
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"Test Accuracy: {test_acc}")

    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes

    print(classification_report(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    data_dir = 'data/'
    _, test_generator = create_data_generator(data_dir)  # Assuming this function has a test subset as well
    evaluate_model('models/saved_model', test_generator)
