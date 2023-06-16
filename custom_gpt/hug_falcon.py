"""
FalconChatbot
-------------

A class that encapsulates the functionality of the Falcon-7B-Instruct or Falcon-40B-Instruct model for use in a
console-based chatbot.

The FalconChatbot class includes methods for downloading the model and tokenizer, loading the model from disk
(or downloading it if it's not already on disk), setting up a text-generation pipeline, and generating a response from
a given prompt.

https://huggingface.co/tiiuae/falcon-7b-instruct

pip install transformers
pip install huggingface_hub
pip install einops


"""

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, pipeline
from huggingface_hub import hf_hub_download
# import transformers
import torch
import os
import sqlite3
import datetime
import getpass
from pathlib import Path
import shutil


class FalconChatbot:
    def __init__(self, model_size: str ="7b",
                 model_dir: str | Path | None = None,
                 lazy_loading: bool =False,
                 redownload_model: bool = False,
                 trust_remote_code: bool = True):
        """
        Initialize the FalconChatbot.

        Parameters:
        model_size (str): Size of the Falcon model to use. Either "7b" or "40b".
        model_dir (str): Directory to store the model. If not provided, the model is stored in the same directory as this module.
        """
        model_size = model_size.lower()
        if model_size not in ["7b", "40b"]:
            raise ValueError("model_size must be either '7b' or '40b'")

        self.model_name = "tiiuae/falcon-" + model_size + "-instruct"
        self.model_dir = Path(model_dir) if model_dir else Path(__file__).resolve().parent / self.model_name.replace("/", "_")
            # os.path.join(os.path.dirname(os.path.realpath(__file__)), self.model_name.replace("/", "_"))
        self.model = None  # The Falcon transformer model
        self.tokenizer = None  # The tokenizer for the Falcon transformer model
        self.pipeline = None  # A Hugging Face pipeline for text generation using the Falcon transformer model
        self.context = ""
        self.username = getpass.getuser()
        self.lazy_loading = lazy_loading
        self._trust_remote_code = trust_remote_code
        self.redownload_model = redownload_model
        self.setup_db()

        if not lazy_loading:
            self.load()

    def setup_db(self):
        """
        Set up a SQLite database to capture all prompts and responses.
        """
        self.conn = sqlite3.connect('falcon_chatbot.db')
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY,
                model_name TEXT,
                username TEXT,
                prompt TEXT,
                response TEXT,
                timestamp TEXT
            )
        ''')
        self.conn.commit()

    def log_chat(self, prompt, response):
        """
        Log a chat in the SQLite database.

        Parameters:
        prompt (str): The prompt for the model to respond to.
        response (str): The model's response.
        """
        timestamp = datetime.datetime.now().isoformat()
        self.cursor.execute('''
            INSERT INTO chat_history (model_name, username, prompt, response, timestamp) VALUES (?, ?, ?, ?, ?)
        ''', (self.model_name, self.username,  prompt, response, timestamp))
        self.conn.commit()

    def set_environment_variables_for_offline_training(self, transformers_offline: int = 1,
                                                       datasets_offline: int = 1):
        # For help: https://huggingface.co/docs/transformers/installation#offline-mode

        # Transformers is able to run in a firewalled or offline environment by only using local files.
        # Set the environment variable TRANSFORMERS_OFFLINE=1 to enable this behavior.
        os.environ['TRANSFORMERS_OFFLINE'] = str(transformers_offline)
        # Add 🤗 Datasets to your offline training workflow by setting the environment variable HF_DATASETS_OFFLINE=1.
        os.environ['HF_DATASETS_OFFLINE'] = str(datasets_offline)

    def load(self):
        if self.redownload_model and self.model_dir.exists():
            self.model = None
            shutil.rmtree(self.model_dir, ignore_errors=True)

        if not self.model_dir.exists():
            self.model_dir.mkdir(parents=True, exist_ok=True)

        if not self.redownload_model and self.model_dir.exists():
            self.set_environment_variables_for_offline_training(transformers_offline=1, datasets_offline=1)

        def load_config():
            if not self.redownload_model and self.config is not None:
                return
            self.config = None

            config_path = Path(self.model_dir) / "config.json"

            if not config_path.exists():
                hf_hub_download(repo_id=self.model_name, filename="config.json", local_dir=self.model_dir)

            try:
                self.config = AutoConfig.from_pretrained(config_path, trust_remote_code=self._trust_remote_code)
            except Exception as e:
                print(f"An error occurred while loading the config from config_path: {e}")
                try:
                    self.config = AutoConfig.from_pretrained(self.model_dir, trust_remote_code=self._trust_remote_code)
                except Exception as e:
                    print(f"An error occurred while loading the config from self.model_dir: {e}")
                    self.config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=self._trust_remote_code)

        def load_model():
            if not self.redownload_model and self.model is not None:
                return
            self.model = None

            try:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                                  trust_remote_code=self._trust_remote_code)
            except Exception as e:
                print(f"An error occurred while loading the model: {e}")

            if self.model is None:
                self.model = AutoModelForCausalLM.from_pretrained(self.mod,
                                                                  trust_remote_code=self._trust_remote_code)

        def load_tokenizer():
            if not self.redownload_model and self.tokenizer is not None:
                return

            self.tokenizer = None
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir,
                                                               trust_remote_code=self._trust_remote_code)
            except Exception as e:
                print(f"An error occurred while loading the tokenizer: {e}")

            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name,
                                                               trust_remote_code=self._trust_remote_code)
                self.tokenizer.save_pretrained(self.model_dir)

        def setup_pipeline():
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                torch_dtype=torch.bfloat16,
                trust_remote_code=self._trust_remote_code,
                device_map="auto",
            )
            self.redownload_model = False

        load_config()
        load_model()
        load_tokenizer()
        setup_pipeline()
        self.redownload_model = False

    def generate_response(self, prompt):
        """
        Generate a response from the Falcon transformer model given a prompt.

        Parameters:
        prompt (str): The prompt for the model to respond to.

        Returns:
        str: The model's response.
        """
        if self.lazy_loading:
            if self.config is None:
                self.load_config()
            if self.model is None:
                self.load_model()
            if self.tokenizer is None:
                self.load_tokenizer()
            if self.pipeline is None:
                self.setup_pipeline()

        sequences = self.pipeline(
            prompt,
            max_length=200,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        response = sequences[0]['generated_text']
        self.log_chat(prompt, response)
        return response


if __name__ == '__main__':
    # comment
    chatbot = FalconChatbot(model_size="7b", trust_remote_code=True, redownload_model=False)  # Change "7b" to "40b" to use the Falcon-40B-Instruct model
    chatbot.load()
    chatbot.set_environment_variables_for_offline_training(transformers_offline=1, datasets_offline=1)

    chatbot.context = "Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Girafatron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe."

    while True:
        user_input = input("You: ")
        chatbot.context += f"\nYou: {user_input}\nGirafatron:"
        response = chatbot.generate_response(chatbot.context)
        print(f"Girafatron: {response}")
        chatbot.context += f" {response}"
