from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torchvision.datasets import ImageFolder
import numpy as np
import io
import threading
import queue

app = Flask(__name__)
CORS(app)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def training_step(model, batch):
    images, labels = batch
    images, labels = images.to(device), labels.to(device)
    out = model(images)
    loss = F.cross_entropy(out, labels)
    return loss

def validation_step(model, batch):
    images, labels = batch
    images, labels = images.to(device), labels.to(device)
    out = model(images)
    loss = F.cross_entropy(out, labels)
    acc = accuracy(out, labels)
    return {'Loss': loss.detach(), 'Acc': acc}

def validation_epoch_end(model, outputs):
    batch_losses = [x['Loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()
    batch_accs = [x['Acc'] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()
    return {'Loss': epoch_loss.item(), 'Acc': epoch_acc.item()}

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [validation_step(model, batch) for batch in val_loader]
    return validation_epoch_end(model, outputs)

class Noise(nn.Module):
    def __init__(self, *dim):
        super().__init__()
        self.noise = torch.nn.Parameter(torch.randn(*dim), requires_grad=True)

    def forward(self):
        return self.noise

class UnlearningTask:
    def __init__(self, model, classes_to_forget):
        self.model = model
        self.classes_to_forget = classes_to_forget
        self.progress = 0
        self.result = None
        self.error = None

task_queue = queue.Queue()
tasks = {}

def worker():
    while True:
        task_id, task = task_queue.get()
        try:
            unlearn_model(task)
        except Exception as e:
            task.error = str(e)
        finally:
            task_queue.task_done()

threading.Thread(target=worker, daemon=True).start()

def unlearn_model(task):
    model = task.model
    classes_to_forget = task.classes_to_forget

    # Load datasets
    data_dir = './data/cifar10'
    transform = tt.Compose([
        tt.ToTensor(),
        tt.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_ds = ImageFolder(data_dir+'/train', transform)
    valid_ds = ImageFolder(data_dir+'/test', transform)

    # Prepare datasets for retained and forgotten classes
    retain_samples = [(img, torch.tensor(label, dtype=torch.long)) for img, label in train_ds if label not in classes_to_forget]
    forget_valid = [(img, torch.tensor(label, dtype=torch.long)) for img, label in valid_ds if label in classes_to_forget]
    retain_valid = [(img, torch.tensor(label, dtype=torch.long)) for img, label in valid_ds if label not in classes_to_forget]

    batch_size = 256
    forget_valid_dl = DataLoader(forget_valid, batch_size, num_workers=3, pin_memory=True)
    retain_valid_dl = DataLoader(retain_valid, batch_size*2, num_workers=3, pin_memory=True)

    task.progress = 10
    
    # Unlearning process
    # Impair step
    noises = {}
    for cls in classes_to_forget:
        noises[cls] = Noise(batch_size, 3, 32, 32).to(device)
        opt = torch.optim.Adam(noises[cls].parameters(), lr=0.1)
        
        for _ in range(3):  # Reduced num_epochs
            for _ in range(5):  # Reduced num_steps
                inputs = noises[cls]()
                labels = torch.full((batch_size,), cls, dtype=torch.long).to(device)
                outputs = model(inputs)
                loss = -F.cross_entropy(outputs, labels) + 0.1 * torch.mean(torch.sum(inputs.pow(2), dim=(1, 2, 3)))
                opt.zero_grad()
                loss.backward()
                opt.step()

    task.progress = 30

    noisy_data = []
    for cls in classes_to_forget:
        for _ in range(10):  # Reduced num_batches
            batch = noises[cls]().cpu().detach()
            for i in range(batch.size(0)):
                noisy_data.append((batch[i], torch.tensor(cls, dtype=torch.long)))

    noisy_data += retain_samples
    noisy_loader = DataLoader(noisy_data, batch_size=256, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    for _ in range(1):  # num_epochs for impair step
        model.train()
        for inputs, labels in noisy_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

    task.progress = 60

    # Repair step
    heal_loader = DataLoader(retain_samples, batch_size=256, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for _ in range(1):  # num_epochs for repair step
        model.train()
        for inputs, labels in heal_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

    task.progress = 80

    # Evaluate performance
    forget_performance = evaluate(model, forget_valid_dl)
    retain_performance = evaluate(model, retain_valid_dl)

    # Save the unlearned model
    unlearned_model_buffer = io.BytesIO()
    torch.save(model.state_dict(), unlearned_model_buffer)
    unlearned_model_buffer.seek(0)

    task.progress = 100
    task.result = {
        'message': 'Unlearning successful',
        'forget_accuracy': forget_performance['Acc'] * 100,
        'forget_loss': forget_performance['Loss'],
        'retain_accuracy': retain_performance['Acc'] * 100,
        'retain_loss': retain_performance['Loss'],
        'unlearned_model': unlearned_model_buffer.getvalue().decode('latin1')
    }

@app.route('/unlearn', methods=['POST'])
def unlearn():
    if 'model' not in request.files:
        return jsonify({'error': 'No model file uploaded'}), 400
    
    model_file = request.files['model']
    classes_to_forget = request.form.get('classes')

    if not model_file or not classes_to_forget:
        return jsonify({'error': 'Missing model file or classes'}), 400

    # Load the model
    model_buffer = io.BytesIO(model_file.read())
    model = resnet18(num_classes=10).to(device)
    model.load_state_dict(torch.load(model_buffer, map_location=device))

    # Parse classes to forget
    classes_to_forget = [int(cls) for cls in classes_to_forget.split(',')]

    # Create a new task
    task = UnlearningTask(model, classes_to_forget)
    task_id = len(tasks)
    tasks[task_id] = task

    # Add task to queue
    task_queue.put((task_id, task))

    return jsonify({'task_id': task_id}), 202

@app.route('/progress/<int:task_id>', methods=['GET'])
def get_progress(task_id):
    if task_id not in tasks:
        return jsonify({'error': 'Task not found'}), 404

    task = tasks[task_id]
    if task.error:
        return jsonify({'error': task.error}), 500
    elif task.result:
        return jsonify(task.result), 200
    else:
        return jsonify({'progress': task.progress}), 200

if __name__ == '__main__':
    app.run(debug=True, threaded=True)