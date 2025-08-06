import pandas as pd

from constants import CLEANED_TEST_DATA_PATH, ENCODED_TEST_DATA_PATH
from models import Qwen3EmbeddingModel2
from utils import evaluate_qwen3_embedding, evaluate_dummy_model, load_qwen3_embedding

def evaluate_models(dataset_path: str, column_name: str, n_samples: int):
    df = pd.read_csv(dataset_path)
    print(f"Number of samples: {n_samples}")
    print(f"The used column to test the models: {column_name}")
    embedding_score = evaluate_qwen3_embedding(df, column_name, Qwen3EmbeddingModel2, n_samples)
    print(f"The F1 score for the embedding model: {embedding_score}")

    dummy_score = evaluate_dummy_model(df, column_name, n_samples)
    print(f"The F1 score for the dummy model: {dummy_score}")

def encode_dataset(dataset_path: str, save_path: str):
    df = pd.read_csv(dataset_path)
    model = load_qwen3_embedding(Qwen3EmbeddingModel2)

    product_names = df["Item_Name"].tolist()
    classes = df["class"].tolist()
    unique_classes = list(set(classes))

    product_name_embeddings = model.get_embeddings(product_names, "query")
    class_embeddings = model.get_embeddings(unique_classes)

    keys = []
    values = []
    for i in range(len(unique_classes)):
        key = unique_classes[i]
        value = class_embeddings[i, :].float().cpu().numpy()
        keys.append(key)
        values.append(value)

    zipped_classes = zip(keys, values)
    encoded_classes_dict = dict(zipped_classes)

    encoded_classes = []
    for cls in classes:
        cls_embd = encoded_classes_dict[cls]
        encoded_classes.append(cls_embd)

    df["Item_Name_Emb"] = product_name_embeddings.float().cpu().numpy().tolist()
    df["class_emb"] = encoded_classes

    df.to_csv(save_path, index=False, encoding="utf-8-sig")

def main():
    # encode_dataset(CLEANED_TEST_DATA_PATH, ENCODED_TEST_DATA_PATH)
    # Item_Name, cleaned_item_name
    N_SAMPLES = None
    evaluate_models(CLEANED_TEST_DATA_PATH, "cleaned_item_name", N_SAMPLES)

if __name__ == "__main__":
    main()