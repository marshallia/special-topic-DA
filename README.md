"# special-topic-DA" 
<h2>Step 1: Install Conda</h2>
If you don’t already have Conda, download and install Miniconda or Anaconda:
Miniconda: https://docs.conda.io/en/latest/miniconda.html
Anaconda: https://www.anaconda.com/products/distribution

<h2>Step 2: Create the Conda Environment</h2>
Open a terminal (or Anaconda Prompt) and navigate to the folder containing your requirements.yaml. Then run:

```conda env create -f requirements.yaml```

This will create a new environment named sim2real_component_classification (as specified in requirements.yaml) with all dependencies installed.

<h2>Step 3: Activate the Environment</h2>
```conda activate sim2real_component_classification```
You should see the environment name in your terminal prompt.

<h2>Step 4: Verify Installation</h2>

Check if PyTorch and other packages are installed correctly:

```python -c "import torch; print(torch.__version__)"```
```python -c "import torchvision; print(torchvision.__version__)"```
```python -c "import cv2; print(cv2.__version__)"```
```python -c "import sklearn; print(sklearn.__version__)"```

<h2>Step 5: Create Project Folder Structure</h2>

Inside your project directory, create the following folders for datasets and saved models:

```mkdir -p data/sim/train data/real/unlabeled data/real/val data/real/test```
```mkdir -p models```


Explanation of folders:

data/sim/train → Simulated training images (labeled)

data/real/unlabeled → Real images without labels for domain adaptation

data/real/val → Labeled real images for validation

data/real/test → Labeled real images for final testing

models → To save trained model checkpoints (baseline, DANN, CDAN, etc.)

<h2>Step 6: Run Notebooks and Scripts</h2>

Start JupyterLab or Notebook in the environment:

jupyter lab
# or
jupyter notebook


Open your notebooks (data_exploration.ipynb, baseline_results.ipynb, etc.) and run them.

Run training scripts from terminal if needed:

```python train_baseline.py```
python train_randomized.py```
```python adapt_dann.py```
```python adapt_cdan.py```


Trained models will be automatically saved in the models/ folder.
