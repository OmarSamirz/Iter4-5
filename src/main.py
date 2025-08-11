import pandas as pd
import matplotlib.pyplot as plt
import os 
import time

from utils import load_embedding_model

from evaluation import (
    evaluate_dummy_model, 
    evaluate_embedding_topk_model, 
    evaluate_tfidf, 
    evaluate_tfidf_model, 
    evaluate_embedding_model,
    evaluate_ensemble_models  
)

from constants import (
    CLEANED_TEST_DATA_PATH,
    ENCODED_TEST_DATA_PATH,
    CLEANED_TRAIN_DATA_PATH,
    E5_LARGE_INSTRUCT_CONFIG_PATH,
    OUTPUT_TEST_DATA_PATH,
    OUTPUT_PATH
)

def evaluate_models(
    dataset_path: str, 
    column_name: str,
    config_path: str,  
    n_samples: int,
    k: int,
):
    df = pd.read_csv(dataset_path)
    print(f"Number of samples: {n_samples}")
    print(f"The used column to test the models: {column_name}")
    embedding_score, topk_preds = evaluate_embedding_topk_model(df, column_name, config_path, n_samples, k)
    print(f"The F1 score for the embedding model: {embedding_score}")
    return topk_preds
    # dummy_score = evaluate_dummy_model(df, column_name, n_samples)
    # print(f"The F1 score for the dummy model: {dummy_score}")

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
    #encode_dataset(CLEANED_TEST_DATA_PATH, E5_LARGE_INSTRUCT_CONFIG_PATH, ENCODED_TEST_DATA_PATH)
    N_SAMPLES = None
    # Item_Name, cleaned_item_name
    #pred = evaluate_models(CLEANED_TEST_DATA_PATH, "cleaned_text", E5_LARGE_INSTRUCT_CONFIG_PATH , N_SAMPLES, 5)
    #score_mk  = evaluate_tfidf_model(CLEANED_TRAIN_DATA_PATH, CLEANED_TEST_DATA_PATH)
    #score_samir, pred = evaluate_tfidf(CLEANED_TRAIN_DATA_PATH, CLEANED_TEST_DATA_PATH)
    bagging_results, boosting_results = evaluate_ensemble_models(CLEANED_TRAIN_DATA_PATH,  CLEANED_TEST_DATA_PATH, E5_LARGE_INSTRUCT_CONFIG_PATH)
   
    # df = pd.read_csv(CLEANED_TEST_DATA_PATH)
    # df_output = df[["cleaned_text", "class"]].copy()
    # df_output["prediction"] = pred
    # df_output["correct"] = (df_output["prediction"] == df_output["class"]).astype(int)
    # df_output.to_csv(OUTPUT_TEST_DATA_PATH, index=False, encoding="utf-8-sig")
    # plot_class_distributions(df_output, OUTPUT_PATH)
    #save_ensemble_results(df, ensemble_results, OUTPUT_PATH)


    #print(f"F1 score random forest + tfidf: {score_mk}")
    #print(f"F1 score tfidf: {score_samir}")
    print(f"F1 score Bagging: {bagging_results}, F1 Score Boosting {boosting_results}")



if __name__ == "__main__":
    start = time.perf_counter()
    main()
    end = time.perf_counter()
    print(f"Runtime: {end - start:.4f}")
   