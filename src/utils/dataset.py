from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from transformers.training_args import TrainingArguments
from .args import *
import json
import re

@dataclass
class TrainSplit(object):
    dataset: Dataset
    subgraphs: Dict[str, dict]

@dataclass
class EvalSplit(object):
    dataset: Dataset
    examples: Dataset
    subgraphs: Dict[str, dict]

@dataclass
class DatasetSplits(object):
    train_split: Optional[TrainSplit]
    eval_split: Optional[EvalSplit]
    test_splits: Optional[Dict[str, EvalSplit]]
    subgraphs: Dict[str, dict]


def _get_subgraphs(examples: Dataset) -> Dict[str, dict]:
    subgraphs: Dict[str, dict] = dict()
    for ex in examples:
        if ex["uid"] not in subgraphs:
            subgraphs[ex["uid"]] = ex["subgraph"]
    return subgraphs


def _prepare_train_split(
        dataset: Dataset,
        data_training_args: DataTrainingArguments,
        data_args: DataArguments,
        add_serialized_schema: Callable[[dict, Optional[str]], dict],
        pre_process_function: Callable[[dict, Optional[str]], dict],
) -> TrainSplit:

    if data_args.dataset in ["webqsp", "cwq"]:
        subgraphs = _get_subgraphs(dataset)
    else:
        subgraphs = None

    dataset = dataset.map(
        lambda ex: add_serialized_schema(
            ex=ex,
            mode='train'
        ),
        batched=False,
        num_proc=data_training_args.preprocessing_num_workers,
        load_from_cache_file=True,
    )

    if data_training_args.train_samples_ratio != 1.0:
        dataset = dataset.select(range(int(dataset.num_rows*data_training_args.train_samples_ratio)))

    column_names = dataset.column_names

    dataset = dataset.map(
        lambda batch: pre_process_function(
            batch=batch,
            mode='train',
        ),
        batched=True,
        num_proc=data_training_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=False,
    )
    return TrainSplit(dataset=dataset, subgraphs=subgraphs)


def _prepare_eval_split(
        dataset: Dataset,
        data_training_args: DataTrainingArguments,
        data_args: DataArguments,
        add_serialized_schema: Callable[[dict, Optional[str]], dict],
        pre_process_function: Callable[[dict, Optional[str]], dict],
) -> EvalSplit:

    eval_examples = dataset

    if data_args.dataset in ["webqsp", "cwq"]:
        subgraphs = _get_subgraphs(dataset)
    else:
        subgraphs = None

    eval_dataset = eval_examples.map(
        lambda ex: add_serialized_schema(
            ex=ex,
            mode='eval'
        ),
        batched=False,
        num_proc=data_training_args.preprocessing_num_workers,
        load_from_cache_file=True,
    )
    if data_training_args.max_val_samples is not None:
        eval_dataset = eval_dataset.select(range(data_training_args.max_val_samples))
    column_names = eval_dataset.column_names
    eval_dataset = eval_dataset.map(
        lambda batch: pre_process_function(
            batch=batch,
            mode='eval',
        ),
        batched=True,
        num_proc=data_training_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=False,
    )
    return EvalSplit(dataset=eval_dataset, examples=eval_examples, subgraphs=subgraphs)


def prepare_splits(
        dataset_dict: DatasetDict,
        data_args: DataArguments,
        training_args: TrainingArguments,
        data_training_args: DataTrainingArguments,
        add_serialized_schema: Callable[[dict, Optional[str]], dict],
        pre_process_function: Callable[[dict, Optional[str]], dict],
) -> DatasetSplits:

    train_split, eval_split, test_split = None, None, None
    if training_args.do_train:
        train_split = _prepare_train_split(
            dataset_dict["train"],
            data_training_args=data_training_args,
            data_args=data_args,
            add_serialized_schema=add_serialized_schema,
            pre_process_function=pre_process_function,
        )
    if training_args.do_eval:
        eval_split = _prepare_eval_split(
            dataset_dict["validation"],
            data_args=data_args,
            data_training_args=data_training_args,
            add_serialized_schema=add_serialized_schema,
            pre_process_function=pre_process_function,
        )
    if training_args.do_predict:
        test_split = _prepare_eval_split(
            dataset_dict["test"],
            data_args=data_args,
            data_training_args=data_training_args,
            add_serialized_schema=add_serialized_schema,
            pre_process_function=pre_process_function,
        )

    subgraphs = {
        "train": dict(**(train_split.subgraphs if train_split is not None else {})),
        "eval": dict(**(eval_split.subgraphs if eval_split is not None else {})),
        "test": dict(**(test_split.subgraphs if test_split is not None else {}),)
    }

    return DatasetSplits(
        train_split=train_split,
        eval_split=eval_split,
        test_splits={"test": test_split},
        subgraphs=subgraphs
    )


def serialize_schema(
        db_id: str,
        db_concepts: List[str],
        db_relations: List[str],
        db_concept_labels: List[str],
        db_relation_labels: List[str],
        db_entities: List[str],
        db_entity_labels: List[str],
        schema_serialization_type: str = "peteshaw",
        schema_serialization_with_db_id: bool = True,
        schema_serialization_with_db_content: bool = False,
        schema_serialization_with_db_concept_content: bool = False,
        schema_serialization_with_db_relation_content: bool = False,
        schema_serialization_with_db_entity_content: bool = False
) -> str:
    if schema_serialization_type == "verbose":
        kg_id_str = "ontology: {kg_id}. "
        concepts_str = "concepts: {concepts}. "
        concept_sep = ", "
        concept_str_with_values = "{concept_id} ({concept_label})"
        concept_str_without_values = "{concept_id}"
        relations_str = "relations: {relations}. "
        relation_sep = ", "
        relation_str_with_values = "{relation_id} ({relation_label})"
        relation_str_without_values = "{relation_id}"
        entities_str = "entities: {entities}. "
        entity_sep = ", "
        entity_str_with_values = "{entity_id} ({entity_label})"
        entity_str_without_values = "{entity_id}"
    elif schema_serialization_type == "peteshaw":
        kg_id_str = " {kg_id}"
        concepts_str = " | concepts: {concepts} "
        concept_sep = ", "
        concept_str_with_values = "{concept_id} ({concept_label})"
        concept_str_without_values = "{concept_id}"
        relations_str = " | relations: {relations} "
        relation_sep = ", "
        relation_str_with_values = "{relation_id} ({relation_label})"
        relation_str_without_values = "{relation_id}"
        entities_str = " | entities: {entities}. "
        entity_sep = ", "
        entity_str_with_values = "{entity_id} ({entity_label})"
        entity_str_without_values = "{entity_id}"
    else:
        raise NotImplementedError

    if schema_serialization_with_db_content and schema_serialization_with_db_concept_content:
        concept_str_list = [
            concept_str_with_values.format(
                concept_id=concept_id,
                concept_label=concept_label
            )
            for concept_id, concept_label in zip(db_concepts, db_concept_labels)
        ]
    else:
        concept_str_list = [
            concept_str_without_values.format(
                concept_id=concept_id
            )
            for concept_id in db_concepts
        ]

    if schema_serialization_with_db_content and schema_serialization_with_db_relation_content:
        relation_str_list = [
            relation_str_with_values.format(
                relation_id=relation_id,
                relation_label=relation_label
            )
            for relation_id, relation_label in zip(db_relations, db_relation_labels)
        ]
    else:
        relation_str_list = [
            relation_str_without_values.format(
                relation_id=relation_id
            )
            for relation_id in db_relations
        ]

    if schema_serialization_with_db_content and schema_serialization_with_db_entity_content:
        entity_str_list = [
            entity_str_with_values.format(
                entity_id=entity_id,
                entity_label=entity_label
            )
            for entity_id, entity_label in zip(db_entities, db_entity_labels)
        ]
    else:
        entity_str_list = [
            entity_str_without_values.format(
                entity_id=entity_id
            )
            for entity_id in db_entities
        ]

    if schema_serialization_with_db_id:
        serialized_schema = kg_id_str.format(kg_id=db_id)
    else:
        serialized_schema = "ontology: "

    if len(db_concepts) > 0:
        serialized_schema += concepts_str.format(concepts=concept_sep.join(concept_str_list))
    if len(db_relations) > 0:
        serialized_schema += relations_str.format(relations=relation_sep.join(relation_str_list))
    if len(db_entities) > 0:
        serialized_schema += entities_str.format(entities=entity_sep.join(entity_str_list))

    return serialized_schema


SPARQL_KEYWORDS = {
    'SELECT', 'CONSTRUCT', 'ASK', 'DESCRIBE', 'BIND', 'WHERE', 'LIMIT',
    'VALUES', 'DISTINCT', 'AS', 'FILTER', 'ORDER', 'BY', 'HAVING', 'IN', 'SERVICE', 'OFFSET',
    'NOT', 'EXISTS', 'OPTIONAL', 'UNION', 'FROM', 'GRAPH', 'NAMED', 'DESC',
    'ASC', 'REDUCED', 'STR', 'LANG', 'LANGMATCHES', 'REGEX', 'BOUND', 'DATATYPE',
    'ISBLANK', 'ISLITERAL', 'ISIRI', 'ISURI', 'GROUP_CONCAT', 'GROUP', 'DELETE', 'CLEAR',
    'CREATE', 'COPY', 'DROP', 'INSERT', 'LOAD', 'DATA', 'INTO', 'WITH', 'ALL', 'SILENT',
    'DEFAULT', 'USING', 'MD5', 'SHA1', 'SHA256', 'SHA384', 'SHA512', 'STRSTARTS',
    'STRENDS', 'SAMETERM', 'ISNUMERIC', 'UCASE', 'SUBSTR', 'STRLEN', 'STRBEFORE', 'STRAFTER',
    'REPLACE', 'LEVENSHTEIN_DIST', 'LCASE', 'ENCODE_FOR_URI', 'CONTAINS', 'CONCAT',
    'COALESCE', 'CHOOSE_BY_MAX', 'CHOOSE_BY_MIN', 'YEAR', 'DAY', 'TZ', 'TIMEZONE', 'HOURS',
    'MINUTES', 'MONTH', 'NOW', 'DUR_TO_USECS', 'SECONDS_DBL', 'USECS_TO_DUR', 'IF', 'MINUS',
    'AVG', 'COUNT', 'MAX', 'MIN', 'SAMPLE', 'SUM', 'ABS', 'ADD', 'BASE', 'CEIL', 'COS', 'FLOOR',
    'HAMMING_DIST', 'HAVERSINE_DIST', 'LN', 'LOG2', 'MOD', 'POWER', 'RADIANS', 'RAND',
    'ROUND', 'ROUNDDOWN', 'ROUNDUP', 'TAN', 'VAR', 'VARP'
}

REPLACEMENTS = [
    [' * ', ' asterisk '],
    [' <= ', ' math_leq '],
    [' >= ', ' math_geq '],
    [' != ', ' math_neq '],
    [' = ', ' math_eql '],
    [' < ', ' math_lt '],
    [' > ', ' math_gt '],
    [' ; ', ' separator_semi '],
    #['"', " quote_str "],
    [' , ', ' separator_com '],
    #['^^', ' str_type '],
    ['||', ' or_logical '],
    ['&&', ' and_logical '],
    [' ! ', ' bool_not '],
    #['@', ' lang_at '],
    [' ( ', ' par_open '],
    [' ) ', ' par_close '],
    [' )', ' par_close '],
    ['{', ' brace_open '],
    ['}', ' brace_close '],
    [' . ', ' separator_dot ']
]

REVERSE_REPLACEMENTS = [
    [' asterisk ', ' * '],
    [' math_leq ', ' <= '],
    [' math_geq ', ' >= '],
    [' math_neq ', ' != '],
    [' math_eql ', ' = '],
    [' math_lt ', ' < '],
    [' math_gt ', ' > '],
    [' separator_semi ', ' ; '],
    #[' quote_str ', '"'],
    [' separator_com ', ' , '],
    #[' str_type ', '^^'],
    ['or_logical', ' || '],
    ['and_logical', ' && '],
    [' bool_not ', ' ! '],
    #[' lang_at ', '@'],
    [' par_open ', ' ( '],
    [' par_close ', ' ) '],
    [' par_close', ' ) '],
    [' brace_open ', ' { '],
    [' brace_close ', ' } '],
    [' separator_dot ', ' . ']
]

# Function to check parentheses
def checkParentheses(myStr, open_list, close_list):
    stack = []
    for i in myStr:
        if i in open_list: stack.append(i)
        elif i in close_list:
            pos = close_list.index(i)
            if ((len(stack) > 0) and (open_list[pos] == stack[len(stack) - 1])):
                stack.pop()
            else:
                return True
    return True if len(stack) == 0 else False

def decompose(sparql: str, ent_pattern, rel_pattern, var_pattern):

    open_list = ["par_open"]
    close_list = ["par_close"]

    query_tok_list = sparql.split()

    content = ""
    structure = ""
    is_limit = False
    is_offset = False
    is_filter = False
    is_having = False
    constraint_start_idx = 0
    parentheses = ""

    for idx, query_tok in enumerate(query_tok_list):

        if query_tok == "limit":
            is_limit = True
            structure += query_tok + " "
            continue

        if query_tok == "offset":
            is_offset = True
            structure += query_tok + " "
            continue

        if is_limit:
            content += "[val] " + query_tok + " "
            structure += "[val] "
            is_limit = False
            continue

        if is_offset:
            content += "[val] " + query_tok + " "
            structure += "[val] "
            is_offset = False
            continue

        if query_tok == "filter" and is_filter == False:
            is_filter = True
            constraint_start_idx = idx
            structure += query_tok + " "
        elif query_tok == "having":
            is_having = True
            constraint_start_idx = idx
            structure += query_tok + " "
        else:
            if is_filter or is_having:
                if query_tok == "par_open" or query_tok == "par_close":
                    parentheses += query_tok + " "

                if checkParentheses(parentheses.split(), open_list, close_list) and len(parentheses) != 0:
                    is_filter = False
                    is_having = False
                    parentheses = ""
                    content += "[con] " + " ".join(
                        query_tok_list[constraint_start_idx+1:idx+1]) + " "
                    structure += "[con] "
                else:
                    continue

            else:

                if var_pattern.findall(query_tok):
                    # if query_tok == "var_0":
                    #     content += "[var] " + "*" + " "
                    # else:
                    #     content += "[var] " + query_tok + " "
                    content += "[var] " + query_tok + " "
                    structure += "[var] "
                elif ent_pattern.findall(query_tok):
                    content += "[ent] " + query_tok + " "
                    structure += "[ent] "
                elif rel_pattern.findall(query_tok):
                    content += "[rel] " + query_tok + " "
                    structure += "[rel] "
                else:
                    structure += query_tok + " "

    structure = structure.strip()
    content = content.strip()

    ## future process the structure, because there might be some literals as values
    structure_tok_list = structure.split()
    structure_value_indices = []
    structure_start_idx = 0
    placeholder_counter = 0
    for idx, token in enumerate(structure_tok_list):
        if token in ["[var]", "[ent]", "[rel]", "[con]", "[val]"]:
            placeholder_counter += 1
        else:
            if token in ["single_quote_begin", "quote_begin"]:
                structure_start_idx = idx
            elif token in ["single_quote_end", "quote_end"]:
                structure_value_indices.append([placeholder_counter, structure_start_idx, idx+1])

    content_tok_list = content.split()
    content_value_indices = []
    for value_idx in structure_value_indices:
        content_start_idx = 0
        content_placeholder_counter = 0
        for idx, content_token in enumerate(content_tok_list):
            if content_token in ["[var]", "[ent]", "[rel]", "[con]", "[val]"]:
                content_placeholder_counter += 1
            if content_placeholder_counter == value_idx[0]:
                content_start_idx = idx
                break

        content_end_idx = content_start_idx + 1
        while content_end_idx < len(content_tok_list) and content_tok_list[content_end_idx] not in ["[var]", "[ent]", "[rel]", "[con]", "[val]"]:
            content_end_idx += 1

        content_value_indices.append([content_start_idx, content_end_idx])

    new_structure_tok_list = []
    new_content_tok_list = []
    s_previous_end_idx = 0
    c_previous_end_idx = 0

    for s_value_idx, c_value_idx in zip(structure_value_indices, content_value_indices):
        new_structure_tok_list += structure_tok_list[s_previous_end_idx:s_value_idx[1]]
        new_structure_tok_list += ["[val]"]
        s_previous_end_idx = s_value_idx[2]

        new_content_tok_list += content_tok_list[c_previous_end_idx:c_value_idx[1]]
        new_content_tok_list += ["[val]"] + structure_tok_list[s_value_idx[1]:s_value_idx[2]]
        c_previous_end_idx = c_value_idx[1]

    new_structure_tok_list += structure_tok_list[s_previous_end_idx:]
    new_content_tok_list += content_tok_list[c_previous_end_idx:]

    structure = " ".join(new_structure_tok_list)
    content = " ".join(new_content_tok_list)

    return structure, content


def combine_SC(content: str, structure: str) -> str:

    var_num = structure.count('[var]')
    val_num = structure.count('[val]')
    ent_num = structure.count('[ent]')
    rel_num = structure.count('[rel]')
    con_num = structure.count('[con]')

    if (content.count('[var]') != var_num) or (content.count('[val]') != val_num) or (content.count('[ent]') != ent_num) or (content.count('[rel]') != rel_num) or (content.count('[con]') != con_num):
        return structure

    content_dict = {"[var]": [], "[ent]": [], "[val]": [], "[rel]": [], "[con]": []}
    tok = None
    temp_str = ''
    i = 0
    while i < len(content):
        if content[i] == '[' and i + 4 < len(content) and content[i + 4] == ']' and (
                content[i:i + 5] in ['[var]', '[ent]', '[val]', '[rel]', '[con]']):
            if tok != None:
                content_dict[tok].append(temp_str.strip())
            tok = content[i:i + 5]
            temp_str = ''
            i += 6
            continue
        temp_str += content[i]
        i += 1
    if tok != None:
        content_dict[tok].append(temp_str.strip())

    pred_sql = structure

    # replace [var]
    end_index = 0
    for i in range(var_num):
        begin_index = pred_sql[end_index:].index('[var]') + end_index
        pred_sql = pred_sql[:begin_index] + content_dict['[var]'][i] + pred_sql[begin_index + 5:]
        end_index = begin_index + len(content_dict['[var]'][i]) + 1

    # replace [ent]
    end_index = 0
    for i in range(ent_num):
        begin_index = pred_sql[end_index:].index('[ent]') + end_index
        pred_sql = pred_sql[:begin_index] + content_dict['[ent]'][i] + pred_sql[begin_index + 5:]
        end_index = begin_index + len(content_dict['[ent]'][i]) + 1

    # replace [val]
    end_index = 0
    for i in range(val_num):
        begin_index = pred_sql[end_index:].index('[val]') + end_index
        pred_sql = pred_sql[:begin_index] + content_dict['[val]'][i] + pred_sql[begin_index + 5:]
        end_index = begin_index + len(content_dict['[val]'][i]) + 1

    # replace [rel]
    end_index = 0
    for i in range(rel_num):
        begin_index = pred_sql[end_index:].index('[rel]') + end_index
        pred_sql = pred_sql[:begin_index] + content_dict['[rel]'][i] + pred_sql[begin_index + 5:]
        end_index = begin_index + len(content_dict['[rel]'][i]) + 1

    # replace [con]
    end_index = 0
    for i in range(con_num):
        begin_index = pred_sql[end_index:].index('[con]') + end_index
        pred_sql = pred_sql[:begin_index] + content_dict['[con]'][i] + pred_sql[begin_index + 5:]
        end_index = begin_index + len(content_dict['[con]'][i]) + 1

    if pred_sql[0] == ' ':
        pred_sql = pred_sql[1:]

    pred_sql = [p for p in pred_sql.split() if p != ""]
    pred_sql = " ".join(pred_sql)
    return pred_sql


def store_schema_items(query: str, ent_pattern, rel_pattern):
    schemafactory = SchemaItemFactory()
    entity_matches =  ent_pattern.findall(query)
    for entity_match in set(entity_matches):
        match_entity_name = schemafactory[entity_match]
        query = re.sub(entity_match+"( )+", match_entity_name+" ", query)

    relation_matches = rel_pattern.findall(query)
    for relation_match in set(relation_matches):
        match_relation_name = schemafactory[relation_match]
        query = re.sub(relation_match+"( )+", match_relation_name+" ", query)

    return schemafactory, query


def store_strings(query):
    quoted_string_pattern_str = r'((?<![\\])[\'\"])((?:.(?!(?<![\\])\1))*.?)\1'
    quoted_string_pattern = re.compile(quoted_string_pattern_str)

    strings_with_langtag_pattern_str = quoted_string_pattern_str + r'(\ )*@[a-zA-Z]+(-[a-zA-Z0-9]+)*'
    strings_with_langtag_pattern = re.compile(strings_with_langtag_pattern_str)

    strings_with_datatype_pattern_str = quoted_string_pattern_str + r'(\ )*\^\^(\ )*[a-zA-Z0-9]+:[a-zA-Z0-9]+'
    strings_with_datatype_pattern = re.compile(strings_with_datatype_pattern_str)

    stringfactory = StringFactory()
    strings_with_langtag_matches = strings_with_langtag_pattern.finditer(query)
    if strings_with_langtag_matches:
        old_query = query
        for match in strings_with_langtag_matches:
            match_string = old_query[match.start(): match.end()]
            match_string_name = stringfactory[match_string]
            query = query.replace(match_string, match_string_name)

    strings_with_datatype_matches = strings_with_datatype_pattern.finditer(query)
    if strings_with_datatype_matches:
        old_query = query
        for match in strings_with_datatype_matches:
            match_string = old_query[match.start(): match.end()]
            match_string_name = stringfactory[match_string]
            query = query.replace(match_string, match_string_name)

    strings_matches = quoted_string_pattern.finditer(query)
    if strings_matches:
        old_query = query
        for match in strings_matches:
            match_string = old_query[match.start(): match.end()]
            match_string_name = stringfactory[match_string]
            query = query.replace(match_string, match_string_name)

    return stringfactory, query


def encode(normalized_sparql: str) -> str:

    normalized_sparql = '  '.join(normalized_sparql.split())

    stringfactory, normalized_sparql = store_strings(normalized_sparql)

    encoded_sparql = do_replacements(normalized_sparql)

    for key, value in stringfactory.v2c.items():
        if value[-1] not in ["'", '"']:
            if "'@" in value or '"@' in value:
                tag = " lang_at "
                index = value.index("'@")+1 if "'@" in value else value.index('"@')+1
                left_part = value[:index]
                right_part = value[index + 1:]
            if "'^^" in value or '"^^' in value:
                tag = " string_type "
                index = value.index("'^^")+1 if "'^^" in value else value.index('"^^')+1
                left_part = value[:index]
                right_part = value[index + 2:]

            if left_part[-1] == "'":
                left_part = "single_quote_begin "+left_part[1:-1]+" single_quote_end"
            elif left_part[-1] == '"':
                left_part = "quote_begin " + left_part[1:-1] + " quote_end"
            encoded_sparql = encoded_sparql.replace(key, left_part + tag + right_part)
        else:
            if value == "''":
                encoded_sparql = encoded_sparql.replace(key, "single_quote_begin single_quote_end")
            elif value == '""':
                encoded_sparql = encoded_sparql.replace(key, "quote_begin quote_end")
            else:
                if value[-1] == "'":
                    part = "single_quote_begin "+value[1:-1]+" single_quote_end"
                elif value[-1] == '"':
                    part = "quote_begin " + value[1:-1] + " quote_end"
                encoded_sparql = encoded_sparql.replace(key, part)

    encoded_sparql = ' '.join(encoded_sparql.split())

    return encoded_sparql


def decode(encoded_sparql: str) -> str:

    sparql = reverse_replacements(encoded_sparql)

    sparql = sparql.replace("single_quote_begin single_quote_end", "''")
    sparql = sparql.replace("quote_begin quote_end", '""')
    sparql = sparql.replace("single_quote_begin ", "'").replace(" single_quote_end", "'")
    sparql = sparql.replace("quote_begin ", '"').replace(" quote_end", '"')
    sparql = sparql.replace(" lang_at ", '@')
    sparql = sparql.replace(" string_type ", '^^')

    return ' '.join(sparql.split())


def reverse_replacements(sparql: str) -> str:
    for r in REVERSE_REPLACEMENTS:
        encoding = r[0]
        original = r[-1]
        sparql = sparql.replace(encoding, original)
        stripped_encoding = str.strip(encoding)
        sparql = sparql.replace(stripped_encoding, original)
        sparql = sparql.replace('{', ' { ').replace('}', ' } ')
    return sparql

def do_replacements(sparql: str) -> str:
    for r in REPLACEMENTS:
        encoding = r[-1]
        for original in r[:-1]:
            sparql = sparql.replace(original, encoding)
    return sparql


class VarNameFactory():
    def __init__(self, prefix="var_"):
        self.prefix = prefix
        self.idx = 1
        self.v2c = {} # var(0,1,...) to custom names
        self.c2v = {} # custom names to var(0,1,...)

    def __getitem__(self, name):
        if name not in self.c2v:
            if name == "*":
                self.v2c[self.prefix+str(0)] = name
                self.c2v[name] = self.prefix+str(0)
            else:
                self.v2c[self.prefix+str(self.idx)] = name
                self.c2v[name] = self.prefix+str(self.idx)
                self.idx += 1
        return self.c2v[name]

    def __len__(self):
        return len(self.v2c)

    def __contains__(self, name):
        return name in self.c2v


class StringFactory():
    def __init__(self, prefix="string_"):
        self.prefix = prefix
        self.idx = 0
        self.v2c = {} # string to custom names
        self.c2v = {} # custom names to string

    def __getitem__(self, string):
        if string not in self.c2v:
            self.v2c[self.prefix+str(self.idx)] = string
            self.c2v[string] = self.prefix+str(self.idx)
            self.idx += 1
        return self.c2v[string]

    def __len__(self):
        return len(self.v2c)

    def __contains__(self, name):
        return name in self.c2v


class SchemaItemFactory():
    def __init__(self, prefix="schema_"):
        self.prefix = prefix
        self.idx = 0
        self.v2c = {} # string to custom names
        self.c2v = {} # custom names to string

    def __getitem__(self, string):
        if string not in self.c2v:
            self.v2c[self.prefix+str(self.idx)] = string
            self.c2v[string] = self.prefix+str(self.idx)
            self.idx += 1
        return self.c2v[string]

    def __len__(self):
        return len(self.v2c)

    def __contains__(self, name):
        return name in self.c2v


def lower_(word: str) -> str:
    if word.upper() in SPARQL_KEYWORDS:
        return word.lower()
    else:
        return word