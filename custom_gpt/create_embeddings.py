"""

!pip install spacy
!python -m spacy download en_core_web_lg

Based on code from https://github.com/Me163/youtube/tree/main/Transformers


"""


import os
import spacy
from pdfminer.high_level import extract_text
import docx2txt
import pickle
import numpy as np
# import jsonpickle


# Load the spaCy model
nlp = spacy.load('en_core_web_lg')


def create_embeddings(input_path: str, output_path: str = '') -> list:
    """Create embeddings from PDF or Word documents and save them to a file.

    :param input_path: The path to the PDF or Word document file, or the folder containing the files.
    :type input_path: str
    :param output_path: The path to save the embeddings file.
    :type output_path: str

    :Example 1: Create embeddings from files and save to disk.
    >>> embeddings = create_embeddings('path/to/files')

    :Example 2: Create embeddings from files and return embeddings.
    # >>> create_embeddings('path/to/files', 'path/to/embeddings.npy')
    >>> create_embeddings('path/to/files', 'path/to/embeddings.pickle')
    """
    # Create an empty list to hold the embeddings
    embeddings = []

    # If input_path is a file, add it to the files list
    if os.path.isfile(input_path):
        root, ext = os.path.splitext(input_path)
        if ext.lower() in ('.pdf', '.docx'):
            files = [input_path]
    # If input_path is a folder, add all PDF and Word document files in the folder to the files list
    else:
        files = [os.path.join(input_path, f) for f in os.listdir(input_path)
                 if os.path.isfile(os.path.join(input_path, f)) and
                 os.path.splitext(f)[-1].lower() in ('.pdf', '.docx')]

    # Loop through the files and create embeddings from the documents
    for f in files:
        # Process the document and get the embeddings
        if f.lower().endswith('.pdf'):
            doc_text = extract_text(f)
            pages = doc_text.split('\f')
            doc_objs = list(nlp.pipe(pages))
        elif f.lower().endswith('.docx'):
            doc_text = docx2txt.process(f)
            docs = doc_text.split('\n')
            doc_objs = list(nlp.pipe(docs))
        else:
            raise ValueError(f"Invalid extension. '{f}' does not have extension '.pdf' or '.docx'  ")

    embeddings.extend([doc.vector for doc in doc_objs])

    if output_path:
    # Save the embeddings to a file
        if isinstance(output_path, str) and output_path.lower().endswith('.npy'):
            np.save(output_path, np.array(embeddings))
        elif isinstance(output_path, str) and output_path.lower().endswith('.pickle'):
            with open(output_path, 'wb') as f:
                pickle.dump(embeddings, f)
    else:
        return embeddings
