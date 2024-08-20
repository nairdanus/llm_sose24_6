# Step 1: Install Necessary Libraries
# !pip install lime transformers

# Step 2: Load GPT-2 Model
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Step 3: Define a Function to Get Model Predictions
import torch

def predict_probabilities(texts):
    # Prepare input tensors
    inputs = [tokenizer.encode(text, return_tensors="pt") for text in texts]
    
    # Get the last token's logit (the model's output before softmax)
    logits = [model(input_tensor).logits[:, -1, :].squeeze(0) for input_tensor in inputs]
    
    # Convert logits to probabilities
    probs = [torch.nn.functional.softmax(logit, dim=-1) for logit in logits]
    
    return probs

# Step 4: Implement LIME Explainer
from lime.lime_text import LimeTextExplainer
import numpy as np

# Initialize LIME explainer
explainer = LimeTextExplainer(class_names=tokenizer.get_vocab())

# Define a prediction function for LIME to use
def lime_predict(texts):
    # Get probabilities from the model
    probs = predict_probabilities(texts)
    
    # LIME expects a 2D numpy array
    # We'll convert our list of probabilities to this format
    prob_np = np.array([prob.detach().numpy() for prob in probs])
    
    return prob_np

# Explain a sample text
text_to_explain = "The quick brown fox jumps over the lazy dog"
explanation = explainer.explain_instance(text_to_explain, lime_predict, num_features=10)

# Step 5: Visualize the Explanation
# Print the explanation
print(explanation.as_list())

# Or visualize it
explanation.show_in_notebook()
