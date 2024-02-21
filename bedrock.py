from langchain_community.chat_models import BedrockChat
from deepeval.models.base import DeepEvalBaseLLM
from main import bedrock


class AWSBedrock(DeepEvalBaseLLM):
    def __init__(
            self,
            model
    ):
        self.model = model

    def load_model(self):
        return self.model

    def _call(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt)

    def get_model_name(self):
        return "anthropic.claude-v2:1"


def bedrock_model():

    # Replace these with real values
    custom_model = bedrock()
    return AWSBedrock(model=custom_model)

if __name__ == "__main__":
    aws_bedrock = bedrock_model()
    print(aws_bedrock("Write me a joke"))
