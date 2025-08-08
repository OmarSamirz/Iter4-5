import nltk
from nltk.corpus import stopwords

from pathlib import Path

nltk.download('stopwords')

ALL_STOPWORDS = set(stopwords.words("arabic") + stopwords.words("english"))

CONFIGS_PATH = Path("config")

QWEN3_EMBEDDING_CONFIG_PATH = CONFIGS_PATH / "qwen3_embedding_config.json"

PARAPHRASER_EMBEDDING_CONFIG_PATH = CONFIGS_PATH / "paraphraser_embedding_config.json"

BGE_M3_CONFIG_PATH = CONFIGS_PATH / "bge_m3_config.json"

E5_LARGE_CONFIG_PATH = CONFIGS_PATH / "e5_large_config.json"

E5_LARGE_INSTRUCT_CONFIG_PATH = CONFIGS_PATH / "e5_large_instruct_config.json"

DUMMY_MODEL_CONFIG_PATH = CONFIGS_PATH / "dummy_model_config.json"

DATA_PATH = Path("data")

TEST_DATA_PATH = DATA_PATH / "test.csv"

CLEANED_TEST_DATA_PATH = DATA_PATH / "cleaned_test.csv"

ENCODED_TEST_DATA_PATH = DATA_PATH / "encoded_test.csv"

TRAIN_VAL_DATA_PATH = DATA_PATH / "train_val.csv"

TRAIN_VAL_SAMPLE_DATA_PATH = DATA_PATH / "train_val_sample.csv"

PROMPTS_PATH = "prompts.json"