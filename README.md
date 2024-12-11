# Super-Resolution Reconstruction of Low-Field MRI Images

## Overview
This project implements a Super-Resolution Generative Adversarial Network (SRGAN) to reconstruct high-resolution images from low-resolution inputs, with a specific focus on improving the quality of low-field MRI images.

---

## Code Organization
The project is structured as follows:

### **1. `Baseline_Model.ipynb`**
**Description:** Jupyter notebook implementing the baseline model to compare against SRGAN.

### **2. `SRGAN_Implementation.ipynb`**
**Description:** Jupyter notebook implementing the SRGAN model, combining generator and discriminator training with custom loss functions.
**Requires:** The files `custom_loss.py`, `model_architecture.py`, `model_metrics.py` and `prepare_data.py` need to be imported for suscessful execution. 

### **3. `custom_loss.py`**
**Description:** Defines the custom loss functions used in training the SRGAN model, including perceptual loss, adversarial loss, and content loss.

### **4. `model_architecture.py`**
**Description:** Implements the architecture of the generator and discriminator networks.
- **Generator:** A deep residual network optimized for upscaling low-resolution images to high resolution.
- **Discriminator:** A convolutional neural network distinguishing between real and generated images.

### **5. `model_metrics.py`**
**Description:** Contains code for computing evaluation metrics such as PSNR (Peak Signal-to-Noise Ratio), SSIM (Structural Similarity Index Measure), and reconstruction error.

### **6. `prepare_data.py`**
**Description:** Prepares and preprocesses the MRI dataset for training and evaluation. Includes augmentation and downscaling for low-resolution simulation.

### **7. `reconstruction_error.py`**
**Description:** Implements the reconstruction error metric to quantify the quality of super-resolution outputs compared to ground truth.

### **8. `requirements.txt`**
**Description:** Lists the dependencies required for running the code, ensuring a consistent environment.

---

## Contribution
- **Developed by Me:**
  - `custom_loss.py`: Designed and implemented custom loss functions tailored to super-resolution tasks.
  - `model_metrics.py`: Added evaluation metrics for quantitative analysis.
  - `reconstruction_error.py`: Implemented reconstruction error calculations for detailed performance evaluation.

- **Adapted Code:**
  - `model_architecture.py`: Modified generator and discriminator networks for medical imaging tasks.
  - `prepare_data.py`: Customized to preprocess low-field MRI data for this specific project.

---

## Running Commands

### **1. Baseline Model**
Run the baseline model:
```bash
jupyter notebook Baseline_Model.ipynb
```

### **2. SRGAN Model Implementation**
Train and evaluate the SRGAN model:
```bash
jupyter notebook SRGAN_Implementation.ipynb
```

---

## Notes
1. Ensure all dependencies listed in `requirements.txt` are installed.
2. Use the `--help` flag with any script to view detailed command-line options.
3. The project structure and code organization aim to maintain modularity for easy extensibility and reproducibility.
