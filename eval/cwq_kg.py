from lm_eval.base import MultipleChoiceTask
from lm_eval.base import rf, Task
from lm_eval.metrics import mean
from .utils import get_context, get_options


class CWQ(Task):
    VERSION = 0
    DATASET_PATH = "web_questions"
    DATASET_NAME = None

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def test_docs(self):
        doc = []
        for i in range(len(self.dataset['test'])):
            answers = eval(self.dataset['test']['label'][i])
            if len(answers) == 0:
                answers = [""]
            doc.append({"question": self.dataset['test']['question'][i],
                        "answers": answers})
        return doc

    def doc_to_text(self, doc):
        return doc['question']

    def fewshot_context(
            self, doc, num_fewshot, provide_description=None, rnd=None, description=None,
            add_special_token=False, prompt="default", replace=None,
            user_instruction="my",
            system_instruction=None,
    ):
        question = self.doc_to_text(doc)
        ctx = get_context(question=question, task="qa", prompt=prompt,
                          user_instruction=user_instruction,
                          system_instruction=system_instruction,
                          add_special_token=add_special_token)

        return ctx

    def doc_to_decontamination_query(self, doc):
        return doc["question"]

    def doc_to_target(self, doc):
        # this picks one answer to be the "correct" one, despite sometimes
        # multiple correct answers being possible.
        # TODO: make sure we're actually handling multi-answer correctly
        return " " + doc["answers"][0]

    def _remove_prefixes(self, aliases):
        # Optimization: Remove any alias that has a strict prefix elsewhere in the list
        # we can do this because if the prefix is acceptable by isgreedy, we can stop looking
        aliases.sort()
        ret = [aliases[0]]
        for alias in aliases[1:]:
            if not alias.startswith(ret[-1]):
                ret.append(alias)

        return ret

    def construct_requests(self, doc, ctx):
        ret = []
        for alias in self._remove_prefixes(doc["answers"]):
            _, is_prediction = rf.loglikelihood(ctx, " " + alias)
            ret.append(is_prediction)
        return ret

    def process_results(self, doc, results):
        return {"acc": float(any(results))}

    def aggregation(self):
        return {
            "acc": mean,
        }

    def higher_is_better(self):
        return {"acc": True}
