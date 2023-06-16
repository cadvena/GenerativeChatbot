"""


Be sure to run these pip installation:
    pip install transformers
    pip install torch 2.0.1+cpu  # Needed to support ChatBot.let_model()
        Alaternatively, to use a GPU:
        pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html


"""
import os
import subprocess
import sys
from pathlib import Path
import warnings


import transformers  # "pip install transformers"
import torch 


default_model_name: str = 'google/flan-t5-xl'
# default_model_name: str = 'google/flan-t5-base'


class ChatBot:
    def __init__(self, update_transformers_lib: bool = False,
                 set_environ_variables: bool = False,
                 model_name: str = default_model_name,
                 max_new_tokens: int = 4096,
                 force_cpu: bool = False,
                 ):
        self.import_transformers_lib()
        if update_transformers_lib:
            self.update_transformers_lib()
        if set_environ_variables:
            self.set_environment_variables_for_offline_training()
        self._model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._tokenizer = None
        self.let_tokenizer(overwrite_tokenizer=True)
        self._model = None
        self._force_cpu = force_cpu
        self._device = None
        self.let_model(model=self._model, overwrite_model=True)

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

        # Transformers is able to run in a firewalled or offline environment by only using local files. Set the environment variable TRANSFORMERS_OFFLINE=1 to enable this behavior.
        os.environ['TRANSFORMERS_OFFLINE'] = str(transformers_offline)
        # Add 🤗 Datasets to your offline training workflow by setting the environment variable HF_DATASETS_OFFLINE=1.
        os.environ['HF_DATASETS_OFFLINE'] = str(datasets_offline)

        # python examples/pytorch/translation/run_translation.py --model_name_or_path t5-small --dataset_name wmt16 --dataset_config ro-en ...
        # Run trasnslation.py

    @property
    def device(self):
        if self._device:
            return self._device
        elif self._force_cpu:
            self._device = 'cpu'
        else:
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return self._device

    def let_tokenizer(self, overwrite_tokenizer: bool = False):
        """
        Retrieves an AutoTokenizer from the transformers library for the specified model.

        :param overwrite_tokenizer: Optional, if tokenizer was already set, overwrite it?
        :type overwrite_tokenizer: bool
        :return: The AutoTokenizer instance.
        :rtype: transformers.AutoTokenizer
        """
        # From: https://www.youtube.com/watch?v=tL1zltXuHO8
        if not hasattr(self, 'tokenizer'):
            self._tokenizer = None
        elif self._tokenizer and not overwrite_tokenizer:
            # self._tokenizer already exists.  Do not overwrite.
            return

        self.import_transformers_lib()

        self._tokenizer = transformers.AutoTokenizer.from_pretrained(self._model_name)

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

    def let_model(self, model=None, overwrite_model: bool = False):
        """
        Takes a model (str) and tokenizes it.
        :param model: The name or path of the pre-trained model to use. Defaults to 'google/flan-t5-base'.
        :type model: str
        :return: The AutoModel instance.
        :rtype: transformers.AutoModel"""
        if not hasattr(self, '_model') or overwrite_model:
            self._model = transformers.AutoModelForSeq2SeqLM.from_pretrained(self._model_name)
            self._gen_config = transformers.GenerationConfig(max_new_tokens=self.max_new_tokens)
            self._model.to(self.device)
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
        self.let_model(model=self._model, overwrite_model=False)

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
    chatbot = ChatBot(model_name='google/flan-t5-base')
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

