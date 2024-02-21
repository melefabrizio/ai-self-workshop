import deepeval
import dotenv
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval, HallucinationMetric, AnswerRelevancyMetric, FaithfulnessMetric
from deepeval import evaluate, assert_test

from main import run_request
from bedrock import bedrock_model
import pytest

test_cases = [
    {"input": "Quanta farina serve per fare panini per 23 persone?", "output": "6,9 kg di farina"},
    {"input": "Quanto costa fare il risotto ai funghi?", "output": "Risposta non possibile"},
    {"input": "Quanta acqua e quanto lievito mi servono per fare il pane con 2 kg di farina?",
     "output": "1,2 l di acqua e 10 g di lievito"},
]


correctness_metric = GEval(
    name="Precisione",
    evaluation_steps=[
        "se expected_output non contiene dati numerici ma un concetto, actual_output deve esprimere lo stesso concetto anche con parole o forme diverse",
        "se expected_output contiene dati numerici, actual_output deve esprimere gli stessi dati",
        "se expected_output contiene dati numerici e actual_ouput li esprime arrotondati, l'arrotondamento deve essere relativo a unità di misura discrete che non possono essere frazionate (es. persone, auto, ecc.) o deve essere un arrotondamento per comodità e di piccola entità (es. 5,54 kg -> 5,6 k oppure 34,5 mele -> 35 mele)",
        "se actual_output usa arrotondamenti che hanno effetto sul risultato finale, e se gli arrotondamenti che anno effetto sul risultato finale sono consentiti, anche il risultato finale può essere arrotondato allo stesso modo"
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
)

math_metric = GEval(
    name="Correttezza matematica",
    criteria="Correttezza matematica - se actual_output contiene calcoli matematici, questi devono essere corretti",
    evaluation_steps=[
        'se actual_output contiene calcoli matematici, questi devono essere corretti',
        'se actual_output NON contiene calcoli matematici, allora non è necessario valutare la correttezza matematica',
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
)

hallucination_metric = FaithfulnessMetric(threshold=0.8)
relevancy_metric = AnswerRelevancyMetric(threshold=0.8)

@pytest.mark.parametrize("chat_input", test_cases)
def test_formulozze(chat_input):
    dotenv.load_dotenv()
    result, formulas = run_request(chat_input['input'])
    retrieval_context = list(map(lambda x: "".join(x.page_content), formulas))
    test_case = LLMTestCase(
        input=chat_input['input'],
        expected_output=chat_input['output'],
        actual_output=result,
        retrieval_context=retrieval_context
    )
    assert_test(test_case, [
        correctness_metric,
        hallucination_metric,
        relevancy_metric,
        math_metric]
                )


@deepeval.set_hyperparameters(model="anthropic.claude-v2:1")
def set_hyperparameters():
    return {
        "chunk_size": 150,
        "chunk_overlap": 0,
        "model_temperature": 0,
        "parser_prompt_template": open("resources/parser.txt").read(),
        "expression_prompt_template": open("resources/expression_generator.txt").read(),
        "embedding": "cohere",
    }
