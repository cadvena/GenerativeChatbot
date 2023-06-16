"""

Be sure to run these pip installation:
    pip install transformers
    pip install torch 2.0.1+cpu  # Needed to support ChatBot.let_model()
        Alaternatively, to use a GPU:
        pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
    Optionally, if you wish to manually download models and transformers:
        python -m pip install huggingface_hub


Sample code for Falson 13B
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/falcon-13B")
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/falcon-13B")

"""
import os
import subprocess
import sys
from pathlib import Path
import warnings
from typing import Union

import transformers  # "pip install transformers"
import torch
from huggingface_hub import hf_hub_download


default_model_name: str =  'tiiuae/falcon-7b-instruct'  # 'leutherAI/falcon-13B'
# default_model_name: str = 'google/flan-t5-xl'
# default_model_name: str = 'google/flan-t5-base'


class ChatBot:
    def __init__(self, update_transformers_lib: bool = False,
                 set_environ_variables: bool = True,
                 model_name: str = default_model_name,
                 max_new_tokens: int = 4096,
                 force_cpu: bool = False,
                 transformers_root_folder: str | Path = Path(r'E:/AI/transformers'),
                 hug_hub_download: bool = False,
                 overwrite_model: bool = False,
                 ):
        """
        Create a chatbot object.  Init sets attributes that determine the model and
        transformer to be used and where the model and transformer are stored.  It
        also runs let_model and let_tokenizer to load the model and tokenizer. you
        may then use process_prompt to generate a response from a prompt.

        :param update_transformers_lib: if True, update the transformers library
        :type update_transformers_lib: bool
        :param set_environ_variables: if True, set the environment variables
        :type set_environ_variables: bool
        :param model_name: The name of the model to use exactly as on HuggingFace.
        :type model_name: str
        :param max_new_tokens: The maximum number of tokens to generate
        :type max_new_tokens: int
        :param force_cpu: if True, use CPU, else use GPU if available.
        :type force_cpu: bool
        :param transformers_root_folder: the path to the folder where the
                                        transformers library is installed.
        :type transformers_root_folder: pathlib.Path
        :param hug_hub_download: If True, download the transformers library from
                                 HUGGINGFACE using hf_hub_download instead
                                 of using the transformers library.
        :type hug_hub_download: bool
        """
        self.import_transformers_lib()
        if update_transformers_lib:
            self.update_transformers_lib()
        if set_environ_variables:
            self.set_environment_variables_for_offline_training()
        self._transformers_root_folder: Path = Path(transformers_root_folder)
        self._model_name: str = model_name
        self.max_new_tokens = max_new_tokens
        self._hug_hub_download: bool = hug_hub_download
        self._tokenizer = None
        self.let_tokenizer(overwrite_tokenizer=True)
        self._model = None
        self._force_cpu = force_cpu
        self._device = None
        self.let_model(overwrite_model=True)
        self._overwrite_model = overwrite_model

    @property
    def transformer_path(self) -> Path:
        fp =  Path(self._transformers_root_folder) / self.model_name
        if not fp.exists():
            fp.mkdir(parents=True, exist_ok=True)
        return fp

    @property
    def config_path(self) -> Path:
        fp = self._transformers_root_folder / 'config' / self.model_name / "config.json"
        if not fp.parent.exists():
            fp.parent.mkdir(parents=True, exist_ok=True)
        return fp

    def import_transformers_lib(self, lib_name: str = 'transformers'):
        """
        Pip install the transformers library from HUGGINGFACE.  This library allows you to install transformers.
        For help, see https://huggingface.co/docs/transformers/installation#install-with-pip

        :param lib_name: The name of the library to install. Options include:
            transformers, tokenizers, and datasets.
            transformers: 'transformers', 'transformers[torch]', 'transformers[tf-cpu]', 'transformers[flax]'
        :return: True if the update was successful, False otherwise.
        """
        try:
            import transformers
        except ModuleNotFoundError:
            subprocess.check_call(f"pip install {lib_name}")
            import transformers
        return True

    def update_transformers_lib(self, transformers_subfolder: str = 'transformers'):
        """
        Update transformers via git pull
        For help, see https://huggingface.co/docs/transformers/installation#install-with-pip

        :param transformers_subfolder: The name of the library to install. Options include:
            transformers, tokenizers, and datasets.
            transformers: 'transformers', 'transformers[torch]', 'transformers[tf-cpu]', 'transformers[flax]'
        """
        # change directories to the path to the folder for this python file using pathlib.Path
        __folder__ = Path(os.path.dirname(os.path.abspath(__file__))).absolute()
        os.chdir(__folder__)
        # change to transformers_folder subdirectory of here.
        transformers_subfolder = os.path.join(__folder__, transformers_subfolder)
        os.chdir(transformers_subfolder)
        # check if the folder exists
        assert os.path.exists(transformers_subfolder), \
            f'git pull failed.  Folder {transformers_subfolder} does not exist.'
        # perform a git pull
        subprocess.check_call(['git', 'pull'])

    def set_environment_variables_for_offline_training(self, transformers_offline: int = 1,
                                                       datasets_offline: int = 1):
        # For help: https://huggingface.co/docs/transformers/installation#offline-mode

        # Transformers is able to run in a firewalled or offline environment by only using local files.
        # Set the environment variable TRANSFORMERS_OFFLINE=1 to enable this behavior.
        os.environ['TRANSFORMERS_OFFLINE'] = str(transformers_offline)
        # Add 🤗 Datasets to your offline training workflow by setting the environment variable HF_DATASETS_OFFLINE=1.
        os.environ['HF_DATASETS_OFFLINE'] = str(datasets_offline)

        # python examples/pytorch/translation/run_translation.py --model_name_or_path t5-small --dataset_name wmt16 --dataset_config ro-en ...
        # Run trasnslation.py

    @property
    def device(self):
        """If an NVIDIA GPU is available, set device to cuda, otherwise set devise to cpu.  Return self._device"""
        if self._device:
            return self._device
        elif self._force_cpu:
            self._device = 'cpu'
        else:
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return self._device

    def hug_hub_download(self):
        """
        Download transformers rom Hugging Face via hf_hub_download
        instead of transformers.
        """
        from huggingface_hub import hf_hub_download
        # self._transformers_root_folder
        hf_hub_download(repo_id=self._model_name,
                        filename=self.config_path,
                        cache_dir=self.transformer_path)

    def let_tokenizer(self, overwrite_tokenizer: bool = None):
        """
        Retrieves an AutoTokenizer from the transformers library for the specified model.

        :param overwrite_tokenizer: Optional, if tokenizer was already set, overwrite it?
        :type overwrite_tokenizer: bool
        :return: The AutoTokenizer instance.
        :rtype: transformers.AutoTokenizer
        """
        if overwrite_tokenizer is None:
            overwrite_tokenizer = self._overwrite_model
        # From: https://www.youtube.com/watch?v=tL1zltXuHO8
        if not hasattr(self, 'tokenizer'):
            self._tokenizer = None
        elif self._tokenizer and not overwrite_tokenizer:
            # self._tokenizer already exists.  Do not overwrite.
            return self._tokenizer

        if self.transformer_path.exists():
            try:
                self._tokenizer = transformers.AutoTokenizer.from_pretrained(self.transformer_path)
                self._tokenizer.save_pretrained(self.transformer_path)
                return self._tokenizer
            except OSError:
                # OSError: E:\AI\transformers\google\flan-t5-base does not appear to have a file named config.json.
                # Checkout 'https://huggingface.co/E:\AI\transformers\google\flan-t5-base/None' for available files.
                pass
        elif self._hug_hub_download:
            # Download from Hugging Face
            self.hug_hub_download()
        else:
            self.import_transformers_lib()
            # Download your files ahead of time with PreTrainedModel.from_pretrained():
            # ExmapleS: "EleutherAI/falcon-13B", Examples:'google/flan-t5-xl', 'google/flan-t5-base'
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(self._model_name)

            # Save your files to a specified directory with PreTrainedModel.save_pretrained():
            self._tokenizer.save_pretrained(self.transformer_path)

        return self._tokenizer

    def tokenize_prompt(self, prompt: str = 'What color is the sky?') -> list:
        """
        Takes a prompt (str) and tokenizes it.
        :param prompt: A prompt/question to send to the model (i.e., the chatbot).
        :type prompt: str
        :return: The tokenized prompt (list of strings).
        :rtype: list
        """
        # From: https://www.youtube.com/watch?v=tL1zltXuHO8
        self.let_tokenizer(overwrite_tokenizer=False)
        # tokens = self._tokenizer.tokenize(prompt).to(self.device)
        # token_ids = self._tokenizer.convert_tokens_to_ids(tokens)
        # return token_ids
        tensor = self._tokenizer(prompt, return_tensors='pt', add_special_tokens=True).to(self.device)
        return tensor

    def let_model(self, overwrite_model: bool = None):
        """
        Takes a model (str) and tokenizes it.
        :param model: The name or path of the pre-trained model to use. Defaults to 'google/flan-t5-base'.
        :type model: str
        :return: The AutoModel instance.
        :rtype: transformers.AutoModel
        """
        if overwrite_model is None:
            overwrite_model = self._overwrite_model

        if not hasattr(self, '_model') or overwrite_model:
            loaded = False
            if self.transformer_path.exists():
                try:
                    self._model = transformers.AutoModel.from_pretrained(self.transformer_path)
                    loaded = True
                except OSError:
                    # OSError: E:\AI\transformers\google\flan-t5-base does not appear to have a file named config.json.
                    # Checkout 'https://huggingface.co/E:\AI\transformers\google\flan-t5-base/None' for available files.
                    loaded = False

            if not loaded and self._hug_hub_download:
                # Download from Hugging Face
                self.hug_hub_download()
            elif not loaded:
                # Download your files ahead of time with PreTrainedModel.from_pretrained():
                try:
                    self._model = transformers.AutoModelForCausalLM.from_pretrained(self._model_name,
                                                                                    trust_remote_code=True)
                except:
                    self._model = transformers.AutoModelForSeq2SeqLM.from_pretrained(self._model_name, trust_remote_code=True)

                # Save your files to a specified directory with PreTrainedModel.save_pretrained():
                self._model.save_pretrained(self.transformer_path)

                # Now when you’re offline, reload your files with PreTrainedModel.from_pretrained() from the
                # specified directory:
                self._model = transformers.AutoModel.from_pretrained(self.transformer_path)

            self._gen_config = transformers.GenerationConfig(max_new_tokens=self.max_new_tokens)

            self._model.to(self.device)

            # Once  file is downloaded and locally cached, specify it’s local path to load and use it:
            self._config = transformers.AutoConfig.from_pretrained(self.transformer_path)

        return self._model

    @property
    def model(self):
        return self._model
    
    @property
    def model_name(self):
        return self._model_name

    def embed(self, tokens):
        # Add the tokenized prompt to the embeddings.
        if not hasattr(self, 'input_embeddings') or not self.input_embeddings:
            self._input_embeddings = self._model.get_input_embeddings()
        if tokens:
            if isinstance(tokens, str):
                tokens = self.tokenize_prompt(prompt=tokens)
            self.our_embeddings = self._input_embeddings(tokens['input_ids'][0])
        else:
            self.our_embeddings = self._input_embeddings

    def process_prompt(self, prompt: str):
        # In case tokenizer is not yet created:
        self.let_tokenizer(overwrite_tokenizer=False)
        # Tokenize the prompt.
        tokens = self.tokenize_prompt(prompt=prompt)

        # In case model is not yet created:
        self.let_model(overwrite_model=False)

        # Create output tokens tensor
        output_tokens = self._model.generate(**tokens, generation_config=self._gen_config)

        # Convert the output tokens tensor to a list of str.
        outputs = self._tokenizer.batch_decode(output_tokens, skip_special_tokens=True)

        return '\n'.join(outputs)


def main() -> bool:
    """

    :param install: install the transformers library
    :type install: bool
    :param update: update the transformers library
    :type update: bool
    :return: A dict of the results
    :rtype: dict
    """
    chatbot = ChatBot(model_name=default_model_name, overwrite_model=True)
    # prompt = 'What color is the sky?'
    # # tokenize_prompt returns a tensor of token IDs.
    # tokens = chatbot.tokenize_prompt(prompt=prompt)
    # print(prompt, tokens)
    # chatbot.let_model()
    # for line in sys.stdin:

    # This does not yet work.  The embed method is probably the wrong approach.  Need to investigate further.
    chatbot.embed('PJM is a company in Pennsylvania that manages the electric power grid for 13 states and D.C.')

    prompt = input('Enter prompt: ')
    while prompt.lower() not in ['e', 'exit', 'q', 'quit']:
        response = chatbot.process_prompt(prompt)
        print(response)
        prompt = input('Enter prompt: ')


if __name__ == '__main__':
    result = main()

