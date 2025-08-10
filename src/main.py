import pandas as pd
import matplotlib.pyplot as plt

import time

from utils import load_embedding_model
from evaluation import (
    evaluate_dummy_model, 
    evaluate_embedding_topk_model, 
    evaluate_tfidf, 
    evaluate_tfidf_model, 
    evaluate_embedding_model,
    evaluate_random_forest_embedding_model,
    evaluate_qwen_llm
)
from constants import (
    CLEANED_TEST_DATA_PATH,
    ENCODED_TEST_DATA_PATH,
    CLEANED_TRAIN_DATA_PATH,
)

def evaluate_models(
    dataset_path: str, 
    column_name: str,
    config_path: str,  
    n_samples: int
):
    df = pd.read_csv(dataset_path)
    print(f"Number of samples: {n_samples}")
    print(f"The used column to test the models: {column_name}")
    embedding_score = evaluate_embedding_model(df, column_name, config_path, n_samples)
    print(f"The F1 score for the embedding model: {embedding_score}")

    dummy_score = evaluate_dummy_model(df, column_name, n_samples)
    print(f"The F1 score for the dummy model: {dummy_score}")

def encode_dataset(
    dataset_path: str, 
    config_path: str,
    save_path: str,
):
    df = pd.read_csv(dataset_path)
    model = load_embedding_model(config_path)

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
    N_SAMPLES = None
    # Item_Name, cleaned_item_name
    # evaluate_models(CLEANED_TEST_DATA_PATH, "cleaned_item_name", E5_LARGE_CONFIG_PATH, N_SAMPLES)
    # score = evaluate_tfidf_model(CLEANED_TRAIN_DATA_PATH, CLEANED_TEST_DATA_PATH)
    score, _ = evaluate_random_forest_embedding_model(CLEANED_TRAIN_DATA_PATH, CLEANED_TEST_DATA_PATH)
    # score = evaluate_tfidf(CLEANED_TRAIN_DATA_PATH, CLEANED_TEST_DATA_PATH)
    print(f"F1 score for tfidf: {score}")
    
    try:
        df_test = pd.read_csv(CLEANED_TEST_DATA_PATH)

        qwen_score = evaluate_qwen_llm(
            df_test,
            config_path="config/qwen_config.json",
            column_name="cleaned_text",
            log_csv_path="qwen_predictions_log.csv",
            print_every=50,
        )
        print(f"F1 score for Qwen3-8B-bnb-4bit: {qwen_score:.4f}")
    except Exception as e:
        print(f"Qwen evaluation skipped due to error: {e}")

if __name__ == "__main__":
    main()
   