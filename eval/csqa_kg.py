from lm_eval.base import MultipleChoiceTask
from lm_eval.base import rf, Task
from .utils import get_context, get_options
import random


class CSQA(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "openbookqa"
    DATASET_NAME = "main"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["train"]))
        return self._training_docs

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        out_doc = {
            "id": doc["id"],
            "query": doc["question"],
            "choices": doc["choices"]["text"],
            "gold": ["A", "B", "C", "D", "E", "F", "G", "H", "I"].index(doc["answerKey"].strip()),
        }
        return out_doc

    def doc_to_text(self, doc, replace=False):
        question = doc['query']
        opts = get_options(doc["choices"], replace=replace)
        return question, opts

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]

    def fewshot_context(
            self, doc, num_fewshot, provide_description=None, rnd=None, description=None,
            add_special_token=False, replace=False,
            prompt="default", user_instruction="my", system_instruction=None,
    ):
        question, opts = self.doc_to_text(doc, replace=replace)
        ctx = get_context(question=question, opts=opts, task="mc", prompt=prompt,
                          user_instruction=user_instruction,
                          system_instruction=system_instruction,
                          add_special_token=add_special_token)

        return ctx

    def construct_requests(self, doc, ctx):
        lls = [
            rf.loglikelihood(ctx, " {}".format(choice))[0] for choice in doc["choices"]
        ]

        return lls
