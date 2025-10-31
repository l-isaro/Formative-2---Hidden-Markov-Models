# Formative 2 – Human Activity Recognition using Hidden Markov Models

**Course:** Machine Learning Techniques – African Leadership University (2025)  
**Authors:** Leslie Isaro & David Ubushakebwimana
---

## Project Overview
This project applies **Hidden Markov Models (HMMs)** to classify smartphone motion-sensor data into human activities.  
Each group member recorded accelerometer and gyroscope signals for **Jumping, Standing, Still, and Walking** using the **Sensor Logger** app.  

The dataset was preprocessed, feature-engineered in both time and frequency domains, and modeled with an HMM to infer hidden activity states from noisy sensor measurements.

The goal was to design a full machine learning pipeline that:
1. Collects and cleans real-world motion data,  
2. Extracts time and frequency domain features,  
3. Trains an HMM to model each activity’s temporal dynamics,  
4. Evaluates model accuracy on unseen data.


---

## Key Notebook Sections

| Section | Purpose |
|----------|----------|
| **Data Loading & Visualization** | Loads accelerometer & gyroscope CSVs, merges, resamples, plots raw data. |
| **Feature Extraction** | Computes time-domain (mean, std, RMS, SMA) and frequency-domain (FFT, spectral entropy) features. |
| **Normalization & Clipping** | Applies Z-score normalization, removes NaNs/Infs, and clips outliers. |
| **HMM Training & Decoding** | Fits a `GaussianHMM` for each activity, decodes sequences using Viterbi. |
| **Evaluation (LOO)** | Tests on unseen clips, calculates accuracy, sensitivity, specificity, and confusion matrices. |
| **Reflection** | Analyzes misclassifications and suggests improvements. |

---

## How to Run

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Launch Jupyter Notebook**
   Start the notebook environment and open the main analysis file:
   ```bash
   jupyter notebook HAR_HMM_all_activities.ipynb
   ```
   
4. **Prepare your data**

- Place all `*_acc_*.csv` and `*_gyro_*.csv` files in the `data/` folder.
- Ensure proper naming: `activity_sensor_clip.csv`, e.g. `walking_acc_1.csv`.

---

4. **Run notebook cells sequentially to:**

- Merge and resample signals  
- Extract and normalize features  
- Train and evaluate HMMs  
- Generate figures and metrics



