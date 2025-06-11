import os
import torch
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from PIL import Image
import csv
from tqdm import tqdm

CLASS2NUM = {'Class A': 18, 'Class B': 19, 'Class C': 20}
CRITERIA_GROUPS = {
    18: [0],
    19: [3, 4, 6, 8],
    20: [1, 5, 7, 9, 10]
}

CRITERIA_NAMES = [
    "Exposed rebar",
    "No significant damage",
    "Huge Spalling",
    "X and V-shaped cracks",
    "Continuous Diagonal cracks",
    "Discontinuous Diagonal cracks",
    "Continuous vertical cracks",
    "Discontinuous vertical cracks",
    "Continuous horizontal cracks",
    "Discontinuous horizontal cracks",
    "Small cracks"
]
CRITERIA_THRESHOLDS = {i: 0.5 for i in range(len(CRITERIA_NAMES))}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tfm = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])


damage_class_names = ['Class A', 'Class B', 'Class C']
# load damage model (single-label)
damage_model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
damage_model.classifier[2] = torch.nn.Linear(damage_model.classifier[2].in_features, 3)
damage_state = torch.load(
    r"D:\NTU CE\1132_Deep Learning\Advanced-damage-classification\best_damage_model.pth",
    weights_only=True
)
damage_model.load_state_dict(damage_state)
damage_model.eval().to(device)

# load crack model (multi-label)
crack_model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
crack_model.classifier[2] = torch.nn.Linear(crack_model.classifier[2].in_features, 11)
crack_state = torch.load(
    r"D:\NTU CE\1132_Deep Learning\Advanced-damage-classification\best_crack_model.pth",
    weights_only=True
)
crack_model.load_state_dict(crack_state)
crack_model.eval().to(device)

test_dir = r"D:\NTU CE\1132_Deep Learning\HW4\test_data\beam"
test_imgs = sorted([f for f in os.listdir(test_dir) if f.endswith('.jpg')],
                   key=lambda x: int(os.path.splitext(x)[0]))

results = []

for img_name in tqdm(test_imgs, desc="Processing images"):
    img_path = os.path.join(test_dir, img_name)
    img = Image.open(img_path).convert('RGB')
    img_tensor = tfm(img).unsqueeze(0).to(device)

    # Damage
    with torch.no_grad():
        out = damage_model(img_tensor)
        pred_class_idx = out.argmax(1).item()
        class_name = damage_class_names[pred_class_idx]
        damage_class = CLASS2NUM[class_name]
    # Crack
    with torch.no_grad():
        crack_logits = crack_model(img_tensor)
        crack_probs = torch.sigmoid(crack_logits).cpu().numpy()[0]

    allowed = CRITERIA_GROUPS[damage_class]
    crack_pred = [i for i in allowed if crack_probs[i] >= CRITERIA_THRESHOLDS[i]]
    if not crack_pred:
        best_i = max(allowed, key=lambda i: crack_probs[i])
        crack_pred = [best_i]
    crack_pred = sorted(crack_pred)

    value = ",".join([str(damage_class)] + [str(i) for i in crack_pred])
    row = [os.path.splitext(img_name)[0], value]
    results.append(row)

with open(r"D:\NTU CE\1132_Deep Learning\Advanced-damage-classification\submission.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["ID", "class"])
    writer.writerows(results)
print("Submission saved to submission.csv")
