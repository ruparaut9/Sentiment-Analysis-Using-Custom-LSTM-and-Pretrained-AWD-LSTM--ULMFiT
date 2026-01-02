from datasets import load_dataset, concatenate_datasets

def save_imdb_to_data_folder(data_dir="data", seed=42):
    print(" Starting IMDb dataset download...")
    dataset = load_dataset("imdb")
    print(" Download complete. Preparing splits...")

    # Merge train + test (50k samples total)
    print(" Merging train and test splits...")
    full_dataset = concatenate_datasets([dataset["train"], dataset["test"]])
    full_dataset = full_dataset.shuffle(seed=seed)
    print(" Dataset shuffled.")

    # First split: 70% train, 30% remaining
    print("Splitting into train (70%) and remaining (30%)...")
    train_valid_test = full_dataset.train_test_split(test_size=0.3, seed=seed)
    train_data = train_valid_test["train"]
    remaining = train_valid_test["test"]

    # Second split: 20% test, 10% valid from remaining
    print(" Splitting remaining into validation (10%) and test (20%)...")
    valid_test = remaining.train_test_split(test_size=2/3, seed=seed)
    valid_data = valid_test["train"]
    test_data = valid_test["test"]

    # Save splits into CSV files inside data/
    print(" Saving splits to CSV files...")
    train_data.to_csv(f"{data_dir}/imdb_train.csv")
    valid_data.to_csv(f"{data_dir}/imdb_valid.csv")
    test_data.to_csv(f"{data_dir}/imdb_test.csv")

    print(" Done! Files saved in:", data_dir)
    print("Train size:", len(train_data))
    print("Validation size:", len(valid_data))
    print("Test size:", len(test_data))

if __name__ == "__main__":
    save_imdb_to_data_folder()

