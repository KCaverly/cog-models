from cog import BasePredictor, Input, ConcatenateIterator
from llama_cpp import Llama


class Predictor(BasePredictor):
    def setup(self) -> None:
        self.model = Llama(
            model_path="./nexusraven-v2-13b.Q5_K_M.gguf",
            n_gpu_layers=-1,
            n_ctx=8192,
            n_threads=1,
            main_gpu=0,
        )

    def predict(
        self,
        prompt: str = Input(description="Instruction for model"),
        max_new_tokens: int = Input(
            description="Maximum new tokens to generate.", default=-1
        ),
        temperature: float = Input(
            description="This parameter used to control the 'warmth' or responsiveness of an AI model based on the LLaMA architecture. It adjusts how likely the model is to generate new, unexpected information versus sticking closely to what it has been trained on. A higher value for this parameter can lead to more creative and diverse responses, while a lower value results in safer, more conservative answers that are closer to those found in its training data. This parameter is particularly useful when fine-tuning models for specific tasks where you want to balance between generating novel insights and maintaining accuracy and coherence.",
            default=0.001,
        ),
    ) -> ConcatenateIterator[str]:
        """Run a single prediction on the model"""

        for output in self.model(
            prompt, stream=True, temperature=temperature, max_tokens=max_new_tokens
        ):
            yield output["choices"][0]["text"]
