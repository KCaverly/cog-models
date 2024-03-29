from cog import BasePredictor, Input, ConcatenateIterator
from llama_cpp import Llama

PROMPT_TEMPLATE = "SYSTEM: {system_prompt}\nUSER: {prompt}\nASSISTANT: "
SYSTEM_PROMPT = "You are 'Nous-Capybara', an AI assistant and your purpose and drive is to assist the user with any request they have."


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = Llama(
            model_path="./nous-capybara-34b.Q4_0.gguf",
            n_gpu_layers=-1,
            n_ctx=32000,
            n_threads=1,
            main_gpu=0,
        )

    def predict(
        self,
        prompt: str = Input(description="Instruction for model"),
        system_prompt: str = Input(
            description="System prompt for the model, helps guides model behaviour.",
            default=SYSTEM_PROMPT,
        ),
        prompt_template: str = Input(
            description="Template to pass to model. Override if you are providing multi-turn instructions.",
            default=PROMPT_TEMPLATE,
        ),
        max_new_tokens: int = Input(
            description="Maximum new tokens to generate.", default=512
        ),
        repeat_penalty: float = Input(
            description="This parameter plays a role in controlling the behavior of an AI language model during conversation or text generation. Its purpose is to discourage the model from repeating itself too often by increasing the likelihood of following up with different content after each response. By adjusting this parameter, users can influence the model's tendency to either stay within familiar topics (lower penalty) or explore new ones (higher penalty). For instance, setting a high repeat penalty might result in more varied and dynamic conversations, whereas a low penalty could be suitable for scenarios where consistency and predictability are preferred.",
            default=1.1,
        ),
        temperature: float = Input(
            description="This parameter used to control the 'warmth' or responsiveness of an AI model based on the LLaMA architecture. It adjusts how likely the model is to generate new, unexpected information versus sticking closely to what it has been trained on. A higher value for this parameter can lead to more creative and diverse responses, while a lower value results in safer, more conservative answers that are closer to those found in its training data. This parameter is particularly useful when fine-tuning models for specific tasks where you want to balance between generating novel insights and maintaining accuracy and coherence.",
            default=0.5,
        ),
    ) -> ConcatenateIterator[str]:
        """Run a single prediction on the model"""

        full_prompt = prompt_template.replace("{prompt}", prompt).replace(
            "{system_prompt}", system_prompt
        )

        for output in self.model(
            full_prompt,
            stream=True,
            repeat_penalty=repeat_penalty,
            max_tokens=max_new_tokens,
            temperature=temperature,
        ):
            yield output["choices"][0]["text"]
