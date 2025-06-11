# Advanced-damage-classification---beam
Final Competition (Homework4) --beam.      Team name: Mikasa sio-kim-tsuá.

# 🦾 Beam Damage Classification and Crack Criteria Prediction

This project implements a complete end-to-end pipeline for detecting beam surface damage severity (3 classes) and predicting associated crack characteristics (11 multi-label criteria). It uses PyTorch and ConvNeXt-Tiny architecture, and supports full inference on unlabeled test sets.

> **Note**: Please follow the steps below to reproduce results consistent with our local and Kaggle submissions.

---

## 📦 Setup

1. Clone the repository:

```bash
git clone https://github.com/Mikasasiokimtsua/Advanced-damage-classification---beam.git
cd Advanced-damage-classification---beam
```

> **Note**: This will be your\_local\_project\_directory

2. Create Conda environment:

```bash
conda env create -f environment.yml    # if fail, try: conda env create -f environment_short.yml or environment_detail.yml instead.
conda activate damage_beam_env
```

3. (Alternative) Install via pip:

```bash
pip install torch torchvision pandas tqdm pillow
```

---

## 🔗 Google Drive (required files)

To run this project, please download the following from our shared Google Drive:

🔗 **[Download Link (Google Drive)](https://drive.google.com/drive/u/0/folders/1PpHnzgqc5nhJz24ZNMgGbF4-NNCmvnb9)**

Contents:

* `datasets/` folder (contains all the training/validation/testing data)
* `best_damage_model.pth`
* `best_crack_model.pth`

After download, place them into your\_local\_project\_directory --> Advanced-damage-classification---beam like this:

## 📁 Directory Structure

```bash
your_local_project_directory -->
Advanced-damage-classification---beam/
├── datasets/
│   ├── damage_classification_forTrain/beam_damage/      # 3-class training images (ImageFolder)
│   ├── crack/train/                                     # Multi-label crack image folders
│   ├── crack/valid/
│   └── test_data/beam/                                  # Unlabeled test images
├── best_damage_model.pth                                # Saved damage classification model
├── best_crack_model.pth                                 # Saved crack multi-label model
├── p01_train_damage_model.py
├── p02_train_crack_model.py
├── train_labels.csv                                     # Auto-generated from crack/train
├── valid_labels.csv                                     # Auto-generated from crack/valid
├── p03_inference.py
├── submission.csv                                       # Final submission (ID, class, criteria)
├── environment.yml
├── environment_short.yml
└── README.md
```

> **Note**: datasets/crack/train/ and datasets/crack/valid/ have been labelled using Roboflow.

---

## 🧠 Model Descriptions

### 1. Damage Level Classification (3 Classes)

* Uses `ImageFolder`-based dataset
* Model: `convnext_tiny` with modified classifier output to 3
* Loss: CrossEntropyLoss
* Output:

  * Class A → 18
  * Class B → 19
  * Class C → 20

### 2. Crack Feature Multi-Label Detection (11 Classes)

* CSV-based custom dataset with one-hot encoded labels
* Model: `convnext_tiny`, output layer → 11
* Loss: BCEWithLogitsLoss
* Output: Criteria such as:

  ```
  0: Exposed rebar
  1: No significant damage
  2: Huge Spalling
  3: X and V-shaped cracks
  ...
  10: Small cracks
  ```

---

## 🚀 Training Workflow

### Damage Classification:

```bash
python p01_train_damage_model.py
# Trains the 3-class ConvNeXt model
# Outputs best_damage_model.pth, train_labels.csv, valid_labels.csv
```

### Crack Feature Detection + CSV Generator:

```bash
python p02_train_crack_model.py
# Trains the 11-label ConvNeXt model using CSV dataset
# Runs `make_multilabel_csv()` to generate csv from folder names
# Outputs best_crack_model.pth
```

---

## 🔍 Inference Pipeline

Generate submission CSV for test images:

```bash
python p03_inference.py
```

Output:

```
submission.csv
```

Detailed steps:

1. Load `best_damage_model.pth` and `best_crack_model.pth`
2. For each image in `datasets/test_data/beam`:

   * Predict damage class → map to class ID: 18, 19, or 20
   * Restrict crack criteria based on class:

     * Class A → \[0]
     * Class B → \[3, 4, 6, 8]
     * Class C → \[1, 5, 7, 9, 10]
   * Run sigmoid on crack model outputs
   * Use threshold 0.5 or fallback to max-probability allowed criterion
3. Save predictions to `submission.csv`

## 📑 Submission Format

```
ID	class
1	20,10
2	18,2
...
```

---

## 📬 Contact

If you have questions or encounter bugs, feel free to open an issue or fork and contribute.

