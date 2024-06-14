import re
import abc
import datasets
import numpy as np
from benchmark.lmeval import utils
from benchmark.lmeval.metrics import mean, weighted_perplexity, bits_per_byte
from abc import abstractmethod


class Task(abc.ABC):
    DATASET_PATH: str = None
    DATASET_NAME: str = None

    def __init__(self, data_dir=None, cache_dir=None, download_mode=None):
        self.download(data_dir, cache_dir, download_mode)
        self._training_docs = None
        self._fewshot_docs = None

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        # if show: Couldn't reach 'dataset_name' on the Hub (ConnectionError)  ==> run: python cache.py
        self.dataset = datasets.load_dataset(
            path=self.DATASET_PATH,
            name=self.DATASET_NAME,
            data_dir=data_dir,
            cache_dir=cache_dir,
            download_mode=download_mode,
        )

    def should_decontaminate(self):
        return False

    @abstractmethod
    def has_training_docs(self):
        pass

    @abstractmethod
    def has_validation_docs(self):
        pass

    @abstractmethod
    def has_test_docs(self):
        pass

    def training_docs(self):
        return []

    def validation_docs(self):
        return []

    def test_docs(self):
        return []

    def _process_doc(self, doc):
        return doc

    def fewshot_examples(self, k, rnd):
        if self._training_docs is None:
            self._training_docs = list(self.training_docs())

        return rnd.sample(self._training_docs, k)

    def doc_to_decontamination_query(self, doc):
        print("Override doc_to_decontamination_query with document specific decontamination query.")
        assert False

    @abstractmethod
    def doc_to_text(self, doc):
        pass

    @abstractmethod
    def doc_to_target(self, doc):
        pass

    @abstractmethod
    def construct_requests(self, doc, ctx):
        pass

    @abstractmethod
    def process_results(self, doc, results):
        pass

    @abstractmethod
    def aggregation(self):
        pass

    @abstractmethod
    def higher_is_better(self):
        pass

    def fewshot_description(self):
        import warnings

        warnings.warn(
            "`fewshot_description` will be removed in futures versions. Pass "
            "any custom descriptions to the `evaluate` function instead.",
            DeprecationWarning,
        )
        return ""

    @utils.positional_deprecated
    def fewshot_context(self, doc, num_fewshot, provide_description=None, rnd=None, description=None):
        assert rnd is not None, "A `random.Random` generator argument must be provided to `rnd`"
        assert not provide_description, (
            "The `provide_description` arg will be removed in future versions. To prepend "
            "a custom description to the context, supply the corresponding string via the "
            "`description` arg."
        )
        if provide_description is not None:
            # nudge people to not specify it at all
            print(
                "WARNING: provide_description is deprecated and will be removed in a future version in favor of description_dict"
            )

        description = description + "\n\n" if description else ""

        if num_fewshot == 0:
            labeled_examples = ""
        else:
            # for sets with no training docs, draw from other set *but ensure no overlap with current doc*
            if self.has_training_docs():
                fewshotex = self.fewshot_examples(k=num_fewshot, rnd=rnd)
            else:
                if self._fewshot_docs is None:
                    self._fewshot_docs = list(self.validation_docs() if self.has_validation_docs() else self.test_docs())

                fewshotex = rnd.sample(self._fewshot_docs, num_fewshot + 1)
                fewshotex = [x for x in fewshotex if x != doc][:num_fewshot]

            labeled_examples = "\n\n".join([self.doc_to_text(doc) + self.doc_to_target(doc) for doc in fewshotex]) + "\n\n"

        example = self.doc_to_text(doc)
        return description + labeled_examples + example



REQUEST_RETURN_LENGTHS = {
    "loglikelihood": 2,
    "greedy_until": None,
    "loglikelihood_rolling": None,
}


class Request:
    def __init__(self, request_type, args, index=None):
        if request_type not in REQUEST_RETURN_LENGTHS.keys():
            raise NotImplementedError("The request type {} is not implemented!".format(request_type))

        self.request_type = request_type
        self.args = args
        self.index = index

    def __iter__(self):
        if REQUEST_RETURN_LENGTHS[self.request_type] is None:
            raise IndexError("This request type does not return multiple arguments!")
        for i in range(REQUEST_RETURN_LENGTHS[self.request_type]):
            yield Request(self.request_type, self.args, i)

    def __getitem__(self, i):
        if REQUEST_RETURN_LENGTHS[self.request_type] is None:
            raise IndexError("This request type does not return multiple arguments!")
        return Request(self.request_type, self.args, i)

    def __eq__(self, other):
        return self.request_type == other.request_type and self.args == other.args and self.index == other.index

    def __repr__(self):
        return f"Req_{self.request_type}{self.args}[{self.index}]\n"


class MultipleChoiceTask(Task):
    def doc_to_target(self, doc):
        return " " + doc["choices"][doc["gold"]]

    def construct_requests(self, doc, ctx):
        lls = [rf.loglikelihood(ctx, " {}".format(choice))[0] for choice in doc["choices"]]

        return lls

    def process_results(self, doc, results):
        gold = doc["gold"]

        acc = 1.0 if np.argmax(results) == gold else 0.0
        completion_len = np.array([float(len(i)) for i in doc["choices"]])
        acc_norm = 1.0 if np.argmax(results / completion_len) == gold else 0.0

        return {
            "acc": acc,
            "acc_norm": acc_norm,
        }

    def higher_is_better(self):
        return {
            "acc": True,
            "acc_norm": True,
        }

    def aggregation(self):
        return {
            "acc": mean,
            "acc_norm": mean,
        }


class PerplexityTask(Task, abc.ABC):
    def should_decontaminate(self):
        """Whether this task supports decontamination against model training set."""
        return True

    def has_training_docs(self):
        return False

    def fewshot_examples(self, k, rnd):
        assert k == 0
        return []

    def fewshot_context(self, doc, num_fewshot, provide_description=None, rnd=None, description=None):
        # assert num_fewshot == 0, "The number of fewshot examples must be 0 for perplexity tasks."
        assert rnd is not None, "A `random.Random` generator argument must be provided to `rnd`."
        assert not provide_description, (
            "The `provide_description` arg will be removed in future versions. To prepend "
            "a custom description to the context, supply the corresponding string via the "
            "`description` arg."
        )
        if provide_description is not None:
            # nudge people to not specify it at all
            print(
                "WARNING: provide_description is deprecated and will be removed in a future version in favor of description_dict"
            )

        return ""

    def higher_is_better(self):
        return {
            "word_perplexity": False,
            "byte_perplexity": False,
            "bits_per_byte": False,
        }

    def doc_to_decontamination_query(self, doc):
        return doc

    def doc_to_text(self, doc):
        return ""

    def doc_to_target(self, doc):
        return doc

    def construct_requests(self, doc, ctx):
        assert not ctx
        req = rf.loglikelihood_rolling(self.doc_to_target(doc))
        return req

    def process_results(self, doc, results):
        (loglikelihood,) = results
        words = self.count_words(doc)
        bytes_ = self.count_bytes(doc)
        return {
            "word_perplexity": (loglikelihood, words),
            "byte_perplexity": (loglikelihood, bytes_),
            "bits_per_byte": (loglikelihood, bytes_),
        }

    def aggregation(self):
        return {
            "word_perplexity": weighted_perplexity,
            "byte_perplexity": weighted_perplexity,
            "bits_per_byte": bits_per_byte,
        }

    @classmethod
    def count_bytes(cls, doc):
        return len(doc.encode("utf-8"))

    @classmethod
    def count_words(cls, doc):
        """Downstream tasks with custom word boundaries should override this!"""
        return len(re.split(r"\s+", doc))


class RequestFactory:
    def __getattr__(self, attr):
        def fn(*args):
            return Request(attr, args)

        return fn


rf = RequestFactory()
