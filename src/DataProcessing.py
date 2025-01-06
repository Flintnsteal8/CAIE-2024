import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
import configparser


def read_config(config_file):
    """Read configuration from a file."""
    try:
        config = configparser.ConfigParser()
        config.read(config_file)
        return config
    except Exception as e:
        print(f"Error reading configuration file: {e}")
        return None


def load_datasets(config):
    """Load datasets from paths specified in the configuration."""
    try:
        customer_df = pd.read_csv(config.get("PARAMETERS", "CUSTOMER_DATASET"))
        geolocation_df = pd.read_csv(
            config.get(
                "PARAMETERS",
                "GEOLOCATION_DATASET"))
        order_items_df = pd.read_csv(
            config.get(
                "PARAMETERS",
                "ORDER_ITEMS_DATASET"))
        order_payments_df = pd.read_csv(config.get(
            "PARAMETERS", "ORDER_PAYMENTS_DATASET"))
        order_review_df = pd.read_csv(
            config.get(
                "PARAMETERS",
                "ORDER_REVIEW_DATASET"))
        order_df = pd.read_csv(config.get("PARAMETERS", "ORDER_DATASET"))
        product_df = pd.read_csv(config.get("PARAMETERS", "PRODUCT_DATASET"))
        seller_df = pd.read_csv(config.get("PARAMETERS", "SELLER_DATASET"))
        translated_category_df = pd.read_csv(config.get(
            "PARAMETERS", "TRANSLATED_CATEGORY_DATASET"))

        return {
            'customer': customer_df,
            'geolocation': geolocation_df,
            'order_items': order_items_df,
            'order_payments': order_payments_df,
            'order_review': order_review_df,
            'order': order_df,
            'product': product_df,
            'seller': seller_df,
            'translated_category': translated_category_df
        }
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return None


def handle_null_values(datasets):
    """Handle null values in the datasets."""
    try:
        
        print("Data Processing Stage 1: Feature engineering, Null value handling, aggregation and merging of datasets")
        datasets['order_review']['review_comment_title'].fillna(
            "No comment title", inplace=True)
        datasets['order_review']['review_comment_message'].fillna(
            "No comment message", inplace=True)
        datasets['order'].dropna(inplace=True)
        datasets['product'].dropna(inplace=True)
        return datasets
    except Exception as e:
        print(f"Error handling null values: {e}")
        return datasets


def merge_data(datasets):
    """Merge various datasets into a single DataFrame."""
    try:
        main_items = datasets['order_items'].merge(
            datasets['product'], on="product_id")
        main_items = main_items.groupby("order_id").agg({
            "order_item_id": "max", "price": "sum", "freight_value": "sum",
            "product_category_name": "first",
            "product_photos_qty": "mean", "product_weight_g": "mean",
            "product_length_cm": "mean", "product_height_cm": "mean",
            "product_width_cm": "mean"}).reset_index()

        order_payments = datasets['order_payments'].groupby("order_id").agg({
            "payment_sequential": "max",
            "payment_type": lambda x: ', '.join(x.unique()),
            "payment_installments": "sum",
            "payment_value": "sum"
        }).reset_index()

        order_review = datasets['order_review'].groupby('order_id').agg({
            'review_id': 'count',
            'review_score': 'median',
            'review_creation_date': 'max',
            'review_answer_timestamp': 'max'
        }).reset_index()

        olist = datasets['order'].merge(order_review, on="order_id")
        olist = olist.merge(datasets['customer'], on="customer_id")
        olist = olist.merge(main_items, on="order_id")
        olist = olist.merge(order_payments, on="order_id")
        olist = olist.merge(
            datasets['translated_category'],
            on="product_category_name",
            how="left")

        return olist
    except Exception as e:
        print(f"Error merging datasets: {e}")
        return None


def feature_engineering(olist):
    """Perform feature engineering on the merged DataFrame."""
    try:

        olist['order_purchase_timestamp'] = pd.to_datetime(
            olist['order_purchase_timestamp'])
        olist['order_delivered_customer_date'] = pd.to_datetime(
            olist['order_delivered_customer_date'])
        olist['order_estimated_delivery_date'] = pd.to_datetime(
            olist['order_estimated_delivery_date'])
        olist['order_approved_at'] = pd.to_datetime(olist['order_approved_at'])
        olist['delivery_time'] = (
            olist['order_delivered_customer_date'] -
            olist['order_purchase_timestamp']).dt.days
        olist['diff_delivery_estimated'] = (
            olist['order_estimated_delivery_date'] -
            olist['order_delivered_customer_date']).dt.days
        olist['repeat_buyer'] = olist.duplicated(
            'customer_unique_id', keep=False).astype(int)
        olist['delivery_duration'] = (
            olist['order_delivered_customer_date'] -
            olist['order_approved_at']).dt.days
        olist['review_class'] = olist['review_score'].apply(
            lambda x: 'Positive' if x >= 2.5 else 'Negative')
        olist['on_time'] = olist['order_delivered_customer_date'] <= olist['order_estimated_delivery_date']

        return olist
    except Exception as e:
        print(f"Error in feature engineering: {e}")
        return None


def select_columns(data, cols_to_use):
    """Select specified columns from the dataset."""
    try:
        return data[[col for col in cols_to_use if col in data.columns]]
    except Exception as e:
        print(f"Error selecting columns: {e}")
        return data


def label_encode(data, cat_cols):
    """Apply Label Encoding to categorical columns."""
    try:
        print("Data Processing Stage 2: Label Encoder")
        le = LabelEncoder()
        for col in cat_cols:
            data[col] = le.fit_transform(data[col].astype(str))
        return data
    except Exception as e:
        print(f"Error in label encoding: {e}")
        return data


def remove_outliers(data, float_cols, multiplier=3):
    """Remove outliers from specified float columns using a customizable IQR method."""
    try:
        initial_count = data.shape[0]
        for col in float_cols:
            q1 = data[col].quantile(0.02)
            q3 = data[col].quantile(0.98)
            iqr = q3 - q1
            lower_bound = q1 - multiplier * iqr
            upper_bound = q3 + multiplier * iqr
            data = data[(data[col] >= lower_bound) &
                        (data[col] <= upper_bound)]
        removed_count = initial_count - data.shape[0]
        print(f"Removed {removed_count} outliers.")
        return data
    except Exception as e:
        print(f"Error in removing outliers: {e}")
        return data


def scale_features(data, float_cols):
    """Apply MinMaxScaler to specified float columns after removing outliers."""
    try:
        print("Data Processing Stage 3: Outlier Removal and MinMaxScaler")
        data = remove_outliers(data, float_cols)

        imputer = SimpleImputer(strategy='mean')
        data[float_cols] = imputer.fit_transform(data[float_cols])

        for col in float_cols:
            mm_scaler = MinMaxScaler()
            data[col] = mm_scaler.fit_transform(
                data[col].values.reshape(-1, 1))

        return data
    except Exception as e:
        print(f"Error in scaling features: {e}")
        return data


def retain_features(data, features):
    """Retain only the specified features and the target variable."""
    try:
        return data[features]
    except Exception as e:
        print(f"Error retaining features: {e}")
        return data


def save_data(data, output_file):
    """Save the processed data to a CSV file."""
    try:
        print(f"Data Processing completed. Saving file to {output_file}")
        data.to_csv(output_file, index=False)
    except Exception as e:
        print(f"Error saving data: {e}")


if __name__ == "__main__":
    config_file = "parameter.env"
    config = read_config(config_file)

    if config is None:
        exit(1)  # Exit the script if configuration file loading fails

    datasets = load_datasets(config)

    if datasets is None:
        exit(1)  # Exit the script if dataset loading fails

    datasets = handle_null_values(datasets)
    olist = merge_data(datasets)

    if olist is None:
        exit(1)  # Exit the script if dataset merging fails

    processed_data = feature_engineering(olist)

    if processed_data is None:
        exit(1)  # Exit the script if feature engineering fails

    cols_to_use = [
        'order_id',
        'customer_id',
        'order_status',
        'order_purchase_timestamp',
        'order_approved_at',
        'order_delivered_carrier_date',
        'order_delivered_customer_date',
        'order_estimated_delivery_date',
        'review_id',
        'review_score',
        'review_comment_title',
        'review_comment_message',
        'review_creation_date',
        'review_answer_timestamp',
        'customer_unique_id',
        'customer_zip_code_prefix',
        'customer_city',
        'customer_state',
        'order_item_id',
        'price',
        'freight_value',
        'product_category_name',
        'product_photos_qty',
        'product_weight_g',
        'product_length_cm',
        'product_height_cm',
        'product_width_cm',
        'payment_sequential',
        'payment_type',
        'payment_installments',
        'payment_value',
        'product_category_name_english',
        'repeat_buyer',
        'month',
        'delivery_time',
        'diff_delivery_estimated',
        'review_class',
        'review_year_month',
        'delivery_duration',
        'on_time']

    data = select_columns(processed_data, cols_to_use)

    target = 'repeat_buyer'
    target_data = data[[target]]

    cat_cols = [
        "customer_city", "customer_state", "order_status", "payment_type",
        "product_category_name_english", "review_class"
    ]

    float_cols = [
        "price", "freight_value", "product_photos_qty", "product_weight_g",
        "product_length_cm", "product_height_cm", "product_width_cm",
        "payment_value", "delivery_time", "diff_delivery_estimated",
        "delivery_duration"
    ]
    
    data = label_encode(data, cat_cols)
    data = scale_features(data, float_cols)

    data[target] = target_data

    # Combine new categorical columns and float columns
    features = float_cols + cat_cols + ['on_time']

    data = retain_features(data, features + [target])

    output_file = config.get("PARAMETERS", "PROCESSED_FILE")
    save_data(data, output_file)
