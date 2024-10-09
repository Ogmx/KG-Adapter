from lm_eval.base import MultipleChoiceTask
from lm_eval.base import rf, Task
import random
from .utils import get_context, get_options


class HaluEval(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "openbookqa"
    DATASET_NAME = "main"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def validation_docs(self):
        self.data_num = len(self.dataset["validation"])
        return map(self._process_doc, self.dataset["validation"])

    def set_sg(self, sg):
        self.kg = sg

    def get_sg(self, idx):
        if hasattr(self, 'kg'):
            sg = self.kg[idx]
        else:
            sg = None
        return sg

    def _process_doc(self, doc):
        label = doc['right_answer']
        choices = [doc['right_answer'], doc['hallucinated_answer']]
        random.shuffle(choices)
        gold = choices.index(label)

        out_doc = {
            "query": doc["question"],
            "choices": choices,
            "gold": gold,
        }
        return out_doc

    def doc_to_text(self, doc):
        question = doc['query']
        opts = get_options(doc["choices"])
        return question, opts

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]

    def fewshot_context(
        self, doc, num_fewshot, provide_description=None, rnd=None, description=None,
            add_special_token=False, prompt="default", instruction="my"
    ):
        question, opts = self.doc_to_text(doc)
        ctx = get_context(question=question, opts=opts, task="mc", prompt=prompt, instruction=instruction, add_special_token=add_special_token)

        return ctx

    def construct_requests(self, doc, ctx):
        lls = [
            rf.loglikelihood(ctx, " {}".format(choice))[0] for choice in doc["choices"]
        ]

        return lls