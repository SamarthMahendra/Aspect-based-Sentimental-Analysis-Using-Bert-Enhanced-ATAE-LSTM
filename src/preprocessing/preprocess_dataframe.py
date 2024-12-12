import ast
import pandas as pd


# Polarity Encoding Mapping (including 'conflict')
polarity_encoding = {
    'neutral': 0,
    'positive': 1,
    'negative': 2,
    'conflict': 3,  # Add 'conflict' to the mapping
}

# Function to Preprocess the DataFrame
def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    processed_rows = []
    for _, row in df.iterrows():
        raw_text = row['raw_text']
        # Use ast.literal_eval instead of eval for safety
        try:
            aspect_terms = ast.literal_eval(row['aspectTerms'])
        except (ValueError, SyntaxError):
            aspect_terms = []
        for aspect in aspect_terms:
            polarity = aspect.get('polarity', 'none')
            if polarity != 'none':
                processed_rows.append({
                    'raw_text': raw_text,
                    'aspect_term': aspect['term'],
                    'polarity_encoded': polarity_encoding.get(polarity, 0)  # Default to 'neutral' if not found
                })
    processed_df = pd.DataFrame(processed_rows)
    return processed_df