"""
Download, clean, and split the CFPB Consumer Complaint Database subsample.

Source: https://www.consumerfinance.gov/data-research/consumer-complaints/

Contract:
    - CATEGORIES: hardcoded list of 6-8 product categories to keep (decide once, don't change mid-project)
    - clean_narrative(text) -> str: strip redaction placeholders (e.g. "XXXX"), boilerplate, empty rows
    - stratified_split(df, seed=42) -> (train_df, val_df, test_df): saved indices to data/splits.json
    - Output: data/processed/{train,val,test}.parquet, ~30-40k narratives total after subsampling
"""

CATEGORIES: list[str] = [
    # TODO: fill in 6-8 product categories from the raw dataset's `Product` column
    # e.g. "Credit reporting", "Debt collection", "Mortgage", "Credit card",
    #      "Student loan", "Checking or savings account"
]

SEED = 42


def clean_narrative(text: str) -> str:
    """Strip CFPB redaction placeholders and boilerplate. Return cleaned string."""
    raise NotImplementedError


def stratified_split(df, seed: int = SEED):
    """Stratified train/val/test split by category. Save split indices to data/splits.json."""
    raise NotImplementedError


def main():
    """Entry point for `make data`."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
