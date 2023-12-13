from cog import BasePredictor, Input, ConcatenateIterator
from llama_cpp import Llama

PROMPT_TEMPLATE = "### System Prompt\n{system_prompt}/n### User Message\n{prompt}/n### Assistant\n"
SYSTEM_PROMPT = "You are an intelligent programming assistant."



class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = Llama(model_path="./phind-codellama-34b-v2.q5_K_M.gguf", n_gpu_layers=-1, n_ctx=4096, n_threads=1, main_gpu=0)

    def predict(
        self,
        prompt: str = Input(description="Instruction for model"),
        system_prompt: str = Input(description="System prompt for the model, helps guides model behaviour.", default=SYSTEM_PROMPT),
        prompt_template: str = Input(description="Template to pass to model. Override if you are providing multi-turn instructions.", default=PROMPT_TEMPLATE),
        max_new_tokens: int = Input(description='Maximum new tokens to generate.', default=-1),
        do_sample: bool = Input(description="if set to True , this parameter enables decoding strategies such as multinomial sampling, beam-search multinomial sampling, Top-K sampling and Top-p sampling. All these strategies select the next token from the probability distribution over the entire vocabulary with various strategy-specific adjustments.", default=True),
        top_p: float = Input(description="This parameter controls how many of the highest-probability words are selected to be included in the generated text", default=0.75),
        top_k: int = Input(description="This is the number of probable next words, to create a pool of words to choose from", default=40),
        temperature: float = Input(description="This parameter used to control the 'warmth' or responsiveness of an AI model based on the LLaMA architecture. It adjusts how likely the model is to generate new, unexpected information versus sticking closely to what it has been trained on. A higher value for this parameter can lead to more creative and diverse responses, while a lower value results in safer, more conservative answers that are closer to those found in its training data. This parameter is particularly useful when fine-tuning models for specific tasks where you want to balance between generating novel insights and maintaining accuracy and coherence.", default=0.01),
    ) -> ConcatenateIterator[str]:
        """Run a single prediction on the model"""

        full_prompt = prompt_template.replace("{prompt}", prompt).replace("{system_prompt}", system_prompt)

        for output in self.model(full_prompt, stream=True, do_sample=do_sample, top_p=top_p, top_k=top_k, max_tokens=max_new_tokens, temperature=temperature):
            yield output['choices'][0]['text']
