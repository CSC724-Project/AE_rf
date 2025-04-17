import pandas as pd

class OptimalThroughputProcessor:
    def __init__(self, input_csv: str, output_csv: str, quantile: float = 0.65):
        """
        Initializes the processor with input/output paths and the quantile threshold.
        
        :param input_csv: Path to the input CSV file.
        :param output_csv: Path where the processed CSV will be saved.
        :param quantile: Quantile to use for thresholding (default: 0.65).
        """
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.quantile = quantile
        self.df = None

    def load_data(self):
        """Loads the CSV into a DataFrame."""
        self.df = pd.read_csv(self.input_csv)
        print(f"Loaded data from {self.input_csv}, shape = {self.df.shape}")

    def label_access_count(self):
        """Adds 'access_count_label' based on 'access_count' ranges."""
        self.df['access_count_label'] = self.df['access_count'].apply(
            lambda x: 1 if x <= 10 else (2 if x <= 20 else 3)
        )
        print("Added 'access_count_label' column.")

    def build_combination(self):
        """Creates the 'combination' column by concatenating file size and access count label."""
        self.df['combination'] = (
            self.df['file_size_KB'].astype(str)
            + ' | '
            + self.df['access_count_label'].astype(str)
        )
        print("Built 'combination' column.")

    def compute_OT(self):
        """
        Computes the threshold per combination and
        adds the 'OT' column: 1 if throughput >= threshold, else 0.
        """
        # Compute per-group quantile threshold
        self.df['threshold'] = self.df.groupby('combination')['throughput_KBps']\
                                      .transform(lambda x: x.quantile(self.quantile))
        # Fill OT based on comparison
        self.df['OT'] = (self.df['throughput_KBps'] >= self.df['threshold']).astype(int)
        # Clean up
        self.df.drop(columns='threshold', inplace=True)
        print(f"Computed 'OT' column using the {self.quantile * 100}th percentile threshold.")

    def save_data(self):
        """Saves the processed DataFrame to the output CSV."""
        self.df.to_csv(self.output_csv, index=False)
        print(f"Saved processed data to {self.output_csv}")

    def run(self):
        """Executes the full processing pipeline."""
        self.load_data()
        self.label_access_count()
        self.build_combination()
        self.compute_OT()
        self.save_data()


if __name__ == "__main__":
    processor = ThroughputProcessor(
        input_csv='sample.csv',
        output_csv='out_sample.csv',
        quantile=0.65
    )
    processor.run()



# import pandas as pd

# # Load the dataset (update the file path if necessary)
# df = pd.read_csv('train_defaultCore_with_OT.csv')

# # Add a new column 'access_count_label' based on the 'access_count' column:
# # 1 if access_count is 1–10, 2 if 11–20, 3 if >20
# df['access_count_label'] = df['access_count'].apply(
#     lambda x: 1 if x <= 10 else (2 if x <= 20 else 3)
# )

# # Create a new column by concatenating A and D with " | " between
# df['combination'] = df['file_size_KB'].astype(str) + ' | ' + df['access_count_label'].astype(str)


# # Compute the 65th percentile threshold for each group in the 'combination' column.
# # The transform function applies the lambda to each group and returns a series with the same index as df.
# df['threshold'] = df.groupby('combination')['throughput_KBps'].transform(lambda x: x.quantile(0.65))

# # Create or fill the 'OT' column:
# # Set the value to 1 if the 'throughput_KBps' value is greater than or equal to the corresponding group's threshold,
# # otherwise set it to 0.
# df['OT'] = (df['throughput_KBps'] >= df['threshold']).astype(int)

# # (Optional) Drop the temporary 'threshold' column if you no longer need it.
# df.drop(columns='threshold', inplace=True)

# # Display a few rows to verify the changes
# print("Sample output with 'combination', 'throughput_KBps', and 'OT':")
# print(df[['combination', 'throughput_KBps', 'OT']].head(10))

# # Save the updated DataFrame to a new CSV file
# df.to_csv('test_OT.csv', index=False)
# print("Updated dataset exported as 'train_OT.csv'.")