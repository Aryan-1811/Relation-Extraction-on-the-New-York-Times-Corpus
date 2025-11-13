# %%
import pickle
import torch
import numpy as np
import torch.nn as nn

# Define the SVM Model
class SVM(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SVM, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)  # Linear classifier

    def forward(self, x):
        return self.fc(x)  # No activation (SVM uses raw scores)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SVM(input_dim=5000, num_classes=24).to(device)  # Adjust `num_classes` based on your dataset
model.load_state_dict(torch.load("svm_text_classifier.pth", map_location=device))
model.eval()

# %%
# Load the TF-IDF vectorizer
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# %%
# Load the LabelEncoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# %%
def predict_relation(text):
    """Predicts the relationship for a given sentence."""
    with torch.no_grad():
        # Convert text to TF-IDF vector
        encoded_text = vectorizer.transform([text]).toarray()
        tensor_text = torch.tensor(encoded_text, dtype=torch.float32).to(device)

        # Get model output
        output = model(tensor_text)
        _, pred = torch.max(output, dim=1)  # Get highest scoring class

        # Convert to label
        predicted_label = label_encoder.inverse_transform([pred.cpu().item()])[0]
        return predicted_label

# %%
# Accept user input for inference
if __name__ == "__main__":
    print("Enter a sentence to predict the relationship (type 'exit' to quit):")
    while True:
        sentence = input(">> ")
        if sentence.lower() == "exit":
            break
        prediction = predict_relation(sentence)
        print(f"Predicted Relationship: {prediction}")


