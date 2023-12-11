# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import json
from cog import BasePredictor, Input, ConcatenateIterator
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from typing import List

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.tokenizer = AutoTokenizer.from_pretrained("tokenizer")
        self.model = AutoModelForCausalLM.from_pretrained("weights")
        self.model.to("cuda")
        self.streamer = TextIteratorStreamer(self.tokenizer)

    def predict(
        self,
        messages: str = Input(description='Chat messages, passed as a json string'),
        max_new_tokens: int = Input(description='Maximum new tokens to generate.', default=512),
        do_sample: bool = Input(description="Whether or not to use sampling; use greedy decoding otherwise.", default=False),
        top_k: int = Input(description="The number of highest probability vocabulary tokens to keep for top-k filtering.", default=None),
        top_p: float = Input(description="If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.", 
                             default=None),
        num_return_sequences: int = Input(description="The number of independently computed returned sequences for each element in the batch.", default=1),
    ) -> ConcatenateIterator[str]:
        """Run a single prediction on the model"""

        input_strings = json.loads(messages)

        inputs = self.tokenizer.apply_chat_template(input_strings, return_tensors='pt').to(self.model.device)
        _ = self.model.generate(inputs, 
                                streamer=self.streamer, 
                                max_new_tokens=max_new_tokens, 
                                do_sample=do_sample, 
                                top_k=top_k, 
                                top_p=top_p, 
                                num_return_sequences=num_return_sequences)

        i = 0
        for word in self.streamer:
            i += 1
            if i > len(inputs) and word != "\u003c|EOT|\u003e":
                yield word
