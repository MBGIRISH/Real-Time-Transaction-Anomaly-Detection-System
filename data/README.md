# Dataset Directory

This directory contains the ULB Credit Card Fraud Detection Dataset.

## Download Instructions

Due to GitHub's file size limitations (100MB), the dataset files are not included in this repository.

### To download the dataset:

1. Visit: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Download `creditcard.csv` (approximately 144 MB)
3. Place it in this `data/` directory

### Expected File Structure:

```
data/
├── README.md           (this file)
└── creditcard.csv      (download from Kaggle)
```

### Dataset Information:

- **Name**: ULB Credit Card Fraud Detection Dataset
- **Size**: ~144 MB
- **Records**: 284,807 transactions
- **Features**: 31 (Time, Amount, V1-V28 PCA components, Class)
- **Fraud Rate**: 0.172% (492 fraud cases)

Once the file is downloaded, you can run the `Fraud_Detection_System.ipynb` notebook.

