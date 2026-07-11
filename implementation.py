import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

def fairness_metric(predictions, labels):
    """
    A simple fairness metric: checks if the predictions are equally distributed across classes.
    """
    class_counts = torch.bincount(torch.argmax(predictions, dim=1), minlength=predictions.size(1))
    fairness_score = torch.std(class_counts.float())  # Standard deviation of class distribution
    return fairness_score

def robustness_metric(predictions, perturbed_predictions):
    """
    A simple robustness metric: measures the difference between original and perturbed predictions.
    """
    robustness_score = torch.mean(torch.abs(predictions - perturbed_predictions))
    return robustness_score

def train_model(model, criterion, optimizer, data, labels, epochs=100):
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

if __name__ == '__main__':
    # Dummy data
    torch.manual_seed(42)
    np.random.seed(42)
    input_dim = 10
    hidden_dim = 20
    output_dim = 3
    num_samples = 100

    # Generate random data
    data = torch.randn(num_samples, input_dim)
    labels = torch.randint(0, output_dim, (num_samples,))
    labels_one_hot = nn.functional.one_hot(labels, num_classes=output_dim).float()

    # Initialize model, loss, and optimizer
    model = SimpleMLP(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train the model
    train_model(model, criterion, optimizer, data, labels, epochs=50)

    # Evaluate fairness and robustness
    with torch.no_grad():
        predictions = model(data)

        # Simulate perturbed data
        noise = torch.randn_like(data) * 0.1
        perturbed_data = data + noise
        perturbed_predictions = model(perturbed_data)

        fairness_score = fairness_metric(predictions, labels)
        robustness_score = robustness_metric(predictions, perturbed_predictions)

        print(f"Fairness Metric (lower is better): {fairness_score.item()}")
        print(f"Robustness Metric (lower is better): {robustness_score.item()}")