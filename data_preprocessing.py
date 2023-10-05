import os
import string
import re
import concurrent.futures

data_path = 'data/'

def preprocess_text(text):
    # Remove punctuation and special characters
    text = re.sub(r'[{}]+'.format(re.escape(string.punctuation)), '', text)

    # Convert to lowercase and split into words
    tokens = text.lower().split()

    # Join tokens to form preprocessed text
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text

def preprocess_file(file_path):
    # Check if the path is a regular file before processing
    if os.path.isfile(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            return preprocess_text(text)
    else:
        return None  # Skip directories

def preprocess_data_parallel(file_list):
    # Create a thread pool
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Map the preprocess_file function to process each file in parallel
        preprocessed_texts = list(filter(None, executor.map(preprocess_file, file_list)))

    return preprocessed_texts

# Example usage:
if __name__ == "__main__":
    # Filter out directories before processing
    file_list = [os.path.join(data_path, file_name) for file_name in os.listdir(data_path) if not os.path.isdir(file_name)]
    preprocessed_texts = preprocess_data_parallel(file_list)

    # Save preprocessed data to a text file in the tmp_execution directory
    output_directory = 'tmp_execution'
    os.makedirs(output_directory, exist_ok=True)
    output_file_path = os.path.join(output_directory, 'preprocessed_data.txt')

    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for text in preprocessed_texts:
            output_file.write(text + '\n')

    print(f"Preprocessed data saved to {output_file_path}")
