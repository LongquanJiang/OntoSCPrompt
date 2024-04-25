import sys
sys.path.append("src")

from utils.dataset import serialize_schema, decompose, encode, decode, combine_SC, lower_, VarNameFactory, SPARQL_KEYWORDS
from datasets import load_dataset
from transformers.models.auto import AutoTokenizer
from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast
from tokenizers import AddedToken
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

def check_consistency(dataset, tokenizer, output):
    structures = []
    for ex in dataset:

        normalized_sparql_0 = ex["query"]

        structures.append(ex["structure"])

        structure_0 = encode(ex["structure"])
        content_0 = encode(ex["content"])

        # encoded_sparql_0 = combine_SC(content_0, structure_0)
        #
        # structure_2 = tokenizer(structure_0)["input_ids"]
        # structure_1 = tokenizer.decode(structure_2, skip_special_tokens=True)
        # assert structure_0 == structure_1
        #
        # content_2 = tokenizer(content_0)["input_ids"]
        # content_1 = tokenizer.decode(content_2, skip_special_tokens=True)
        # assert content_0 == content_1
        #
        # encoded_sparql_1 = combine_SC(content_1, structure_1)
        # assert encoded_sparql_0 == encoded_sparql_1
        #
        # normalized_sparql_1 = decode(encoded_sparql_1)
        # assert normalized_sparql_1 == normalized_sparql_0
        #
        # output.write(f"id:{ex['uid']}\n")
        # output.write(f"original:\t{ex['query']}\n")
        # output.write(f"normalized:\t{normalized_sparql_0}\n")
        # output.write(f"encoded:\t{encoded_sparql_0}\n")
        # output.write(f"structure:\t{structure_0}\n")
        # output.write(f"content:\t{content_0}\n")
        # output.write(f"decoded:\t{normalized_sparql_1}\n")
        # output.write("\n\n")
    return structures


if __name__ == '__main__':

    webqsp_train = load_dataset('./src/datasets/webqsp/webqsp.py', split='train')
    webqsp_test = load_dataset('./src/datasets/webqsp/webqsp.py', split='test')

    tokenizer = AutoTokenizer.from_pretrained(
        "google/long-t5-local-base",
        cache_dir="transformers_cache",
        use_fast=True,
        revision="main",
        use_auth_token=None,
    )

    assert isinstance(tokenizer, PreTrainedTokenizerFast), "Only fast tokenizers are currently supported"
    if isinstance(tokenizer, T5TokenizerFast):
        # In T5 `<` is OOV, see https://github.com/google-research/language/blob/master/language/nqg/tasks/spider/restore_oov.py
        tokenizer.add_tokens([AddedToken(" <="), AddedToken(" <"), AddedToken(" ^")])

    output = open("./consistency_check_outputs/webqsp.txt", "w")

    train_structures = check_consistency(webqsp_train, tokenizer, output)
    test_structures = check_consistency(webqsp_test, tokenizer, output)

    output.close()
    print()

