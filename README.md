# 🎬 Netflix Movie Recommendation System

A **content-based movie recommendation system** built using **Python**, **Pandas**, and **scikit-learn**, designed to suggest movies/TV shows based on similarity in metadata such as language, availability, release date, and viewing statistics.  

The system supports **retraining on new datasets** and provides a **Flask-based web interface** for real-time recommendations.

---

## 📌 Features

- **Content-Based Filtering**: Recommendations are based on similarity in categorical and numerical attributes (e.g., genre, language, popularity).
- **Data Cleaning Pipeline**: Automatic handling of missing values, duplicate titles, and inconsistent formats.
- **Custom Feature Engineering**: Generates `Content_ID` for efficient mapping and similarity computation.
- **Web Interface**: Flask app for user-friendly recommendation queries.
- **Retraining Support**: Easily retrain on updated or custom datasets.
- **Optimized for Deployment**: Saves model, preprocessor, and metadata for quick loading.

---

## ⚙️ How It Works

1. **Data Loading & Cleaning**  
   - Reads CSV dataset  
   - Cleans numeric fields (`Hours Viewed`)  
   - Drops duplicate and missing titles  
   - Extracts `Release_Year` from `Release Date`  
   - Assigns unique `Content_ID` to each item  

2. **Feature Engineering**  
   - **Categorical Columns**: One-hot encoded  
   - **Numerical Columns**: Standard scaled  
   - Saves metadata and preprocessor for future use  

3. **Recommendation**  
   - Computes cosine similarity between feature vectors  
   - Returns top-N most similar items to a given title  

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/Netflix_Movie_Recommendation_System.git
cd Netflix_Movie_Recommendation_System
```
### 2️⃣ Install Dependencies
```
pip install  -r requirements.txt

```
### 3️⃣ Prepare the Dataset
Place your dataset (CSV) inside the data/ folder.<br>
Required Columns:<br>
Title Available Globally? Release Date  Hours Viewed  Language Indicator  Content Type

### 4️⃣ Retrain the Model (Optional)  
If you want to train on your dataset:<br>
```
python train.py --input (datapath) --outdir ./models --neighbors 50(optional)

```
###5️⃣ Run the Web App
```
python app.py

```
### Browser View:
```
http://127.0.0.1:5000/

```


