from src.__00__paths import processed_data_dir, curated_data_dir, model_dir
from src.__01__data_preprocessing import *
from src.__02__model_creating import *


def main():
    print(f"Creating Needed Directories...")

    print(f"Downloading Dataset...")
    download_dataset()

    print(f"Preprocessing Dataset...")
    raw_df = load_dataset(raw_data_dir / "loan_approval_dataset.csv")
    processed_df = preprocess_dataset(raw_df)
    print(f"✔️ Dataset Preprocessed.")

    print(f"Splitting Dataset...")
    train_df, test_df = split_dataset(processed_df, test_size=0.2)
    print(f"✔️ Dataset Splitting Completed.")

    print(f"Saving Dataset...")
    save_dataset(processed_df, processed_data_dir / "processed_data.csv")
    save_dataset(train_df, curated_data_dir / "train.csv")
    save_dataset(test_df, curated_data_dir / "test.csv")
    print(f"✔️ Dataset Saved.")

    print(f"Creating Logistic Regression & SVM Models...")
    lr_model = get_lr_model()
    fit_model(lr_model, train_df.drop(columns=['loan_status']), train_df['loan_status'])

    svm_model = get_svm_model()
    fit_model(svm_model, train_df.drop(columns=['loan_status']), train_df['loan_status'])
    print(f"✔️ Models Created Successfully.")

    print(f"Saving Models...")
    save_model(svm_model, model_dir / "SVM_model.joblib")
    save_model(lr_model, model_dir / "LogisticRegression_model.joblib")
    print(f"✔️ Models Saved at {'/'.join(model_dir.parts[-2:])}.")

    print("📊 Testing Models Accuracy: ")
    print_model_accuracy(svm_model, test_df.drop(columns=['loan_status']), test_df['loan_status'], "SVM")
    print_model_accuracy(lr_model, test_df.drop(columns=['loan_status']), test_df['loan_status'], "Logistic Regression")


if __name__ == "__main__":
    main()
