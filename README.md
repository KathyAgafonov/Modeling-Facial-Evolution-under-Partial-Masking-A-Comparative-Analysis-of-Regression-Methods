# Modeling Facial Evolution under Partial Masking

This repository explores how different regression models predict a craniofacial measurement (`head_width`) when part of the data is artificially masked, simulating incomplete observations. Three models—HistGradientBoosting (HGB), Random Forest (RF), and a Neural Network (NN)—are compared using mean squared error (MSE) and the coefficient of determination (\(R^2\)) on extracted facial features.

## 1. Overview

- **Data Source**  
  Frames extracted from John Gurche’s [*Shaping Humanity*](https://www.youtube.com/watch?v=ru8ifph_q9o) video.
  ![image](https://github.com/user-attachments/assets/91a8e1a1-b9b0-467b-b740-a77c6295eb08)


  Each frame undergoes:
  1. **Face Detection** (OpenCV Haar cascade)
  2. **Landmark Extraction** (68-point dlib predictor)
  3. **Feature Computation** (e.g., `head_width`, `angle_forehead`)

- **Partial Masking**  
  A fraction of `head_width` entries is replaced with `NaN`, excluded from training to test each model’s robustness to missing target values.

- **Comparative Regression**  
  1. **Neural Network (NN)**: Two dense layers with ReLU activations; MSE loss  
  2. **Random Forest (RF)**: Ensemble of decision trees (`n_estimators=100`)  
  3. **HistGradientBoosting (HGB)**: Tree-boosting with histogram-based splits  
  Performance is assessed via **MSE** and **\(R^2\)**.

## 2. Installation and Requirements

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/KathyAgafonov/Modeling-Facial-Evolution-under-Partial-Masking-A-Comparative-Analysis-of-Regression-Methods.git
   cd Modeling-Facial-Evolution-under-Partial-Masking-A-Comparative-Analysis-of-Regression-Methods


2. **Environment Setup**  
   - Python 3.7+ recommended  
   - Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```
   - Typical libraries include:
     - `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `dlib`, `opencv-python`

3. **Model Weights** (Optional)  
   - If you have pretrained NN weights or additional files, place them in an accessible folder and document here.

## 3. Usage

1. **Extract Frames and Landmarks**  
   - Run scripts to capture frames from the video and detect 68 landmarks:
     ```bash
     python extract_frames.py
     python detect_landmarks.py
     python compute_features.py
     ```
2. **Partial Masking**  
   - Apply masking to `head_width`:
     ```bash
     python partial_masking.py --mask_ratio 0.2
     ```
   - This sets 20% of `head_width` to `NaN`.
3. **Train Models**  
   - Fit each regression model on unmasked data:
     ```bash
     python train_models.py
     ```
4. **Evaluation**  
   - Evaluate the predictions:
     ```bash
     python evaluate.py
     ```
   - Compare **MSE** and **\(R^2\)**, and optionally produce scatter/bar plots.

## 4. Results

- **Neural Network (NN)**  
  Lowest MSE, indicating tighter clustering of predictions around actual values.
- **HistGradientBoosting (HGB)**  
  Highest \(R^2\), suggesting better capture of overall variance.
- **Random Forest (RF)**  
  Sits between the two, with MSE close to NN and moderately lower \(R^2\).

These differences highlight how partial masking challenges each model differently.

## 5. Future Directions

- **Hyperparameter Tuning**  
  Use grid or random search to refine model parameters (e.g., deeper NN, more tree iterations).
- **Additional Features**  
  Explore more sophisticated facial measures or expand to temporal analyses across frames.
- **Alternative Masking Schemes**  
  Instead of random fractions, investigate structured or cluster-based missingness to replicate real-world fossil gaps.

## 6. License and Citation

- Licensed under [MIT License](LICENSE) (update if you have a different license).
- If you use this code, please cite:
  > Kathy Eva Agafonov. *Modeling Facial Evolution under Partial Masking: A Comparative Analysis of Regression Methods.*

## 7. Contact

Author: **Kathy Eva Agafonov**  
Email: [kathyeva@post.bgu.ac.il](mailto:kathyeva@post.bgu.ac.il)  
GitHub: [Modeling-Facial-Evolution](https://github.com/KathyAgafonov/Modeling-Facial-Evolution-under-Partial-Masking-A-Comparative-Analysis-of-Regression-Methods)

Contributions or issues? Feel free to open a pull request or create an issue.
```
