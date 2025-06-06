import os
from modules import faq_input_manager, db_manager


try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "csv", "faq.csv")

    sentences = faq_input_manager.extract_sentences_from_csv(csv_path)
    db_manager.vector_store.add_texts(sentences)
    print("Sentences have been successfully added to the database.")

except Exception as e:
    print("An error occurred:", e)
