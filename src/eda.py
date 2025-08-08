import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from constants import CLEANED_TEST_DATA_PATH, ENCODED_TEST_DATA_PATH


class MultilingualEDA:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)

    @staticmethod
    def _detect_lang(text: str) -> str:
        if pd.isna(text): return 'unknown'
        arabic_ratio = len(re.findall(r'[\u0600-\u06FF]', text)) / max(len(re.findall(r'[a-zA-Z\u0600-\u06FF]', text)), 1)
        return 'arabic' if arabic_ratio > 0.5 else 'english' if arabic_ratio < 0.1 else 'mixed'

    def add_language_column(self):
        if 'language' not in self.df.columns:
            self.df['language'] = self.df['Item_Name'].astype(str).apply(self._detect_lang)

    def analyze_categories(self) -> pd.Series:
        return self.df['class'].value_counts()

    def analyze_language_distribution(self) -> pd.DataFrame:
        self.add_language_column()
        return self.df.groupby('class')['language'].value_counts().unstack(fill_value=0)

    def dominant_language_by_category(self) -> pd.DataFrame:
        self.add_language_column()
        dist = self.df.groupby('class')['language'].value_counts(normalize=True).unstack(fill_value=0)
        dist['dominant_language'] = dist[['arabic', 'english']].idxmax(axis=1)
        return dist.sort_values('dominant_language')

    def plot_category_frequency(self):
        plt.figure(figsize=(12, 6))
        ax = self.df['class'].value_counts().plot(kind='bar', color='skyblue')
        plt.title("Category Frequency")
        plt.xlabel("Category")
        plt.ylabel("Count")
        for i, v in enumerate(self.df['class'].value_counts()):
            ax.text(i, v + 1, str(v), ha='center')
        plt.tight_layout()
        plt.show()

    def plot_language_distribution(self):
        self.add_language_column()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Pie chart
        self.df['language'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax1)
        ax1.set_title("Overall Language Distribution")
        ax1.set_ylabel("")
        
        # Stacked bar chart
        lang_cat = self.df.groupby('class')['language'].value_counts().unstack(fill_value=0)
        lang_cat.plot(kind='bar', stacked=True, ax=ax2, colormap='tab20')
        ax2.set_title("Language Distribution by Category")
        ax2.set_xlabel("Category")
        ax2.set_ylabel("Count")
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()


eda = MultilingualEDA(CLEANED_TEST_DATA_PATH)

# Visualize
eda.plot_category_frequency()
eda.plot_language_distribution()

# Dominant languages
print(eda.dominant_language_by_category())
