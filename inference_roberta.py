# %%
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Define model class (same as in training)
class RelationshipExtractionModel(torch.nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

# %%
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('roberta-base')

# %%
# Load label encoder (same mapping used during training)
label_encoder = {
    'location/location/contains': 0,
    'people/person/nationality': 1,
    'location/country/capital': 2,
    'people/person/place_lived': 3,
    'location/country/administrative_divisions': 4,
    'location/administrative_division/country': 5,
    'business/person/company': 6,
    'location/neighborhood/neighborhood_of': 7,
    'people/person/place_of_birth': 8,
    'people/deceased_person/place_of_death': 9,
    'business/company/founders': 10,
    'people/person/children': 11,
    'business/company/place_founded': 12,
    'business/company_shareholder/major_shareholder_of': 13,
    'business/company/major_shareholders': 14,
    'sports/sports_team/location': 15,
    'sports/sports_team_location/teams': 16,
    'people/person/religion': 17,
    'people/ethnicity/geographic_distribution': 18,
    'business/company/advisors': 19,
    'people/person/ethnicity': 20,
    'people/ethnicity/people': 21,
    'people/person/profession': 22,
    'business/company/industry': 23
}
reverse_label_encoder = {v: k for k, v in label_encoder.items()}  # Reverse mapping

# %%
# Load the trained model
num_labels = len(label_encoder)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
model = RelationshipExtractionModel(num_labels=24)
model.load_state_dict(torch.load("relationship_extraction_model.pth", map_location=device))
model.to(device)
model.eval()  # Set model to evaluation mode

# %%
# Define inference function
def predict_relationship(sentence):
    encoded_sentence = tokenizer.encode_plus(
        sentence,
        add_special_tokens=True,
        max_length=256,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    ).to(device)

    input_ids = encoded_sentence['input_ids']
    attention_mask = encoded_sentence['attention_mask']

    with torch.no_grad():  # Disable gradients for faster inference
        outputs = model(input_ids, attention_mask)
        _, preds = torch.max(outputs, dim=1)

    predicted_label = reverse_label_encoder[preds.item()]
    return predicted_label

# %%
# Accept user input for inference
if __name__ == "__main__":
    print("Enter a sentence to predict the relationship (type 'exit' to quit):")
    while True:
        sentence = input(">> ")
        if sentence.lower() == "exit":
            break
        prediction = predict_relationship(sentence)
        print(f"Predicted Relationship: {prediction}")


