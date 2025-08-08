import pandas as pd
import time

from utils import load_embedding_model
from evaluation import (
    evaluate_dummy_model,
    evaluate_embedding_model,
    evaluate_tfidf_model,
    evaluate_ensemble_model,
)
from constants import (
    CLEANED_TEST_DATA_PATH,
    ENCODED_TEST_DATA_PATH,
    PARAPHRASER_EMBEDDING_CONFIG_PATH,
    E5_LARGE_CONFIG_PATH,
    E5_LARGE_INSTRUCT_CONFIG_PATH
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
    N_SAMPLES = None
    # Column can be "Item_Name" if you haven’t created a cleaned column yet.
    run_comparative_evaluation(
        dataset_path=CLEANED_TEST_DATA_PATH,
        column_name="cleaned_item_name",
        model_config_path=E5_LARGE_INSTRUCT_CONFIG_PATH,
        n_samples=N_SAMPLES,
        alpha=0.2,  # weight given to Model embeddings vs TF-IDF
    )


def run_comparative_evaluation(
    dataset_path: str,
    column_name: str,
    model_config_path: str,
    n_samples: int = None,
    alpha: float = 0.6,
):
    df = pd.read_csv(dataset_path)
    print(f"Number of samples: {n_samples if n_samples is not None else len(df)}")
    print(f"Using column for evaluation: {column_name}")

    # 1) TF-IDF alone
    start = time.perf_counter()
    tfidf_f1 = evaluate_tfidf_model(df, column_name, n_samples)
    tfidf_dt = time.perf_counter() - start
    print(f"[TF-IDF] Weighted F1: {tfidf_f1:.4f} (took {tfidf_dt:.3f}s)")

    # 2) Model embeddings alone
    start = time.perf_counter()
    para_f1 = evaluate_embedding_model(
        df=df,
        column_name=column_name,
        config_path=model_config_path,
        n_samples=n_samples,
    )
    para_dt = time.perf_counter() - start
    print(f"[Paraphraser embeddings] Weighted F1: {para_f1:.4f} (took {para_dt:.3f}s)")

    # 3) Ensemble (TF-IDF + Model embeddings)
    start = time.perf_counter()
    ensemble_f1 = evaluate_ensemble_model(
        df=df,
        column_name=column_name,
        paraphraser_config_path=model_config_path,
        alpha=alpha,
        n_samples=n_samples,
    )
    ensemble_dt = time.perf_counter() - start
    print(f"[Ensemble alpha={alpha}] Weighted F1: {ensemble_f1:.4f} (took {ensemble_dt:.3f}s)")

if __name__ == "__main__":
    main()