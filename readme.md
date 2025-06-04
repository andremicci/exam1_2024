
## 🚀 How to Run the Project

You can run this project either **locally on the TeoGPU cluster (recommended)** or in **Google Colab**.

---

### 🖥️ Option 1 — Run on teogpu via VSCode (Recommended)

> ✅ Best for fast training with direct access to local files

1. Clone the project or access it from your user folder.
2. Open the project folder in **VSCode**.
3. Set the **Jupyter kernel** to the pre-configured Conda environment: amilici_exam


### 🌐 Option 2 — Run on Google Colab


1. Save or clone the project inside your **Google Drive**.
2. Go to Task*.ipynb notebook
3. Remove # from the following lines at the top of the notebooks to mount your Drive and navigate to the project folder:

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Add project folder to Python path
import sys
sys.path.append('/content/drive/MyDrive/ml_exam/quench_project')  #YOUR PATH TO THE PROJECT IN GOOGLE DRIVE

# Change directory to the project
%cd /content/drive/MyDrive/ml_exam/quench_project

# Check current path (optional)
!pwd

# Specify the input file path if needed
file = ''  # YOUR PATH TO THE DATASET IN GOOGLE DRIVE

