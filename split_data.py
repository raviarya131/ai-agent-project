import json
import random
from sklearn.model_selection import train_test_split


def load_data(filename='dataset.json'):
    """Loads the main JSON dataset file."""
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} total examples from {filename}.")
    return data


def save_jsonl(data_list, filename):
    """Saves a list of dictionaries to a JSONL file."""
    with open(filename, 'w', encoding='utf-8') as f:
        for entry in data_list:
            json.dump(entry, f)
            f.write('\n')
    print(f"Saved {len(data_list)} examples to {filename}.")


def main():
    # Load the data
    all_data = load_data()

    # Extract texts and labels for stratified splitting
    texts = [d['text'] for d in all_data]
    labels = [d['label'] for d in all_data]

    # Split the data (80% train, 20% test)
    # stratify=labels ensures that the train and test sets have the
    # same proportion of labels as the original dataset. This is critical.
    train_data, test_data = train_test_split(
        all_data,
        test_size=0.20,
        random_state=42,
        stratify=labels
    )

    # Save the files in .jsonl format
    save_jsonl(train_data, 'train.jsonl')
    save_jsonl(test_data, 'test.jsonl')

    print("\nData splitting complete!")
    print(f"  Train set size: {len(train_data)}")
    print(f"  Test set size:  {len(test_data)}")


if __name__ == "__main__":
    # Before running, you must install scikit-learn:
    # pip install scikit-learn
    main()