#IMPORTING THE LIBRARIES
#IM USING TENSORFLOW TO LOAD THE MNIST DATASET BECUASE THE OPENML.ORG DATASET WONT WORK ON MY SYSTEM, SORRY FOR THE INCONVENIENCE CAUSED:( .
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
mnist = tf.keras.datasets.mnist
(hand_written_digits_train_full, actual_digits_train_full), (hand_written_digits_test, actual_digits_test) = mnist.load_data()
# HERE WE PREPROCESS THE DATAL
hand_written_digits_train_full = hand_written_digits_train_full.reshape(-1, 28*28) / 255.0  # Flatten and normalize
hand_written_digits_test = hand_written_digits_test.reshape(-1, 28*28) / 255.0
# REDUCING THE DATA SET SIZE FOR OPTIMAL PERFORMANCE
hand_written_digits_train, hand_written_digits_val, actual_digits_train, actual_digits_val = train_test_split(
    hand_written_digits_train_full, actual_digits_train_full, test_size=0.2, random_state=42
)
k = 5  # 5 IS TAKEN AS THE NUMBER OF NEIGHBOURS TO INCREASE THE CREDIBILITY OF KNN CLASSIFICATION
knn = KNeighborsClassifier(n_neighbors=k)
# TRAINING THE KNN MODEL :
knn.fit(hand_written_digits_train, actual_digits_train)
actual_digits_val_pred = knn.predict(hand_written_digits_val)
val_accuracy = accuracy_score(actual_digits_val, actual_digits_val_pred)
print("Validation Accuracy:",val_accuracy)
# REPRORT OF THE TRAINED DATA;
actual_digits_test_pred = knn.predict(hand_written_digits_test)
test_accuracy = accuracy_score(actual_digits_test, actual_digits_test_pred)
print("Test Accuracy:",test_accuracy)
print("\nClassification Report:")
print(classification_report(actual_digits_test, actual_digits_test_pred))



