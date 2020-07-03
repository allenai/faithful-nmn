import sys
import json
import csv
from copy import deepcopy
import nltk
import argparse
from allennlp.data.tokenizers import WordTokenizer
from counts import COUNTS

COUNTS_REV = {str(COUNTS[word]): word for word in COUNTS}
tokenizer = WordTokenizer()
wnl = nltk.stem.wordnet.WordNetLemmatizer()


def _get_children(modules, index):
    children = []
    for i in range(len(modules)):
        j = i - 1
        while j >= 0:
            if (
                len(modules[j]) - len(modules[j].strip("."))
                == len(modules[i]) - len(modules[i].strip(".")) - 1
            ):
                if j == index:
                    children.append(i)
                break
            j -= 1
    return children


def _get_descendants(modules, index):
    descendants = []
    children = _get_children(modules, index)
    descendants += children
    for c in children:
        descendants += _get_descendants(modules, c)
    return descendants


def in_left_right_image(modules):
    right_image_module = "op_in_right_image"
    left_image_module = "op_in_left_image"
    while True:
        found_filter = False
        new_modules = []
        num_periods = []
        parents = []
        for j in range(len(modules)):
            num_periods.append(len(modules[j]) - len(modules[j].strip(".")))
            c = num_periods[-1]
            if c == 0:
                parents.append(-1)
                continue
            k = j - 1
            assert k >= 0
            while modules[k][:c] == "." * c:
                k -= 1
            assert modules[k][: c - 1] == "." * (c - 1) and modules[k][c - 1] != "."
            parents.append(k)
        new_num_periods = []
        for j in range(len(modules)):
            filter_right_sentences = [
                "in right image",
                "in the right image",
                "in the image on the right",
                "in the image to the right",
            ]
            filter_left_sentences = [
                "in left image",
                "in the left image",
                "in the image on the left",
                "in the image to the left",
            ]
            find_left_sentences = [
                "left image",
                "lefthand image",
                "left hand image",
                "the image to the left",
                "the image on the left side",
            ]
            find_right_sentences = [
                "right image",
                "righthand image",
                "right hand image",
                "the image to the right",
                "the image on the right side",
            ]
            if (
                j > 0
                and any([s in modules[j] for s in filter_right_sentences])
                and "[" not in modules[j]
            ):
                assert modules[parents[j]].split("[")[0].strip(".") == "op_filter"
                found_filter = True
                new_modules = (
                    new_modules[: parents[j]]
                    + [num_periods[parents[j]] * "." + right_image_module]
                    + new_modules[parents[j] + 1 :]
                    + [mod for mod in modules[j + 1 :]]
                )
                break
            elif (
                j > 0
                and any([s in modules[j] for s in filter_left_sentences])
                and "[" not in modules[j]
            ):
                assert modules[parents[j]].split("[")[0].strip(".") == "op_filter"
                found_filter = True
                new_modules = (
                    new_modules[: parents[j]]
                    + [num_periods[parents[j]] * "." + left_image_module]
                    + new_modules[parents[j] + 1 :]
                    + [mod for mod in modules[j + 1 :]]
                )
                break
            elif (
                j > 0
                and modules[j].split("[")[0].strip(".") == "op_find"
                and "[" in modules[j]
                and any(
                    [
                        s in modules[j].split("[")[1].split("]")[0]
                        for s in find_right_sentences
                    ]
                )
                and modules[parents[j]].split("[")[0].strip(".") == ["op_filter"]
            ):
                found_filter = True
                new_modules = (
                    new_modules[: parents[j]]
                    + [num_periods[parents[j]] * "." + right_image_module]
                    + [num_periods[j] * "." + "\n"]
                    + new_modules[parents[j] + 1 : -1]
                    + [mod for mod in modules[j + 1 :]]
                )
                break
            elif (
                j > 0
                and modules[j].split("[")[0].strip(".") == "op_find"
                and "[" in modules[j]
                and any(
                    [
                        s in modules[j].split("[")[1].split("]")[0]
                        for s in find_left_sentences
                    ]
                )
                and modules[parents[j]].split("[")[0].strip(".") == ["op_filter"]
            ):
                found_filter = True
                new_modules = (
                    new_modules[: parents[j]]
                    + [num_periods[parents[j]] * "." + left_image_module]
                    + [num_periods[j] * "." + "\n"]
                    + new_modules[parents[j] + 1 : -1]
                    + [mod for mod in modules[j + 1 :]]
                )
                break
            new_modules.append(modules[j])
        modules = new_modules
        if not found_filter:
            modules = []
            for mod in new_modules:
                if mod[-1] != "\n":
                    modules.append(mod)
            break
    while True:
        num_periods = []
        for j in range(len(modules)):
            num_periods.append(len(modules[j]) - len(modules[j].strip(".")))
        found_project = False
        new_modules = []
        for j in range(len(modules)):
            if (
                j > 0
                and modules[j - 1].split("[")[0].strip(".") == "op_project"
                and modules[j].split("[")[0].strip(".") == "op_find"
                and modules[j].split("]")[0].split("[")[1].strip(".")
                in {
                    "right image",
                    "the right image",
                    "image on the right",
                    "the image on the right",
                    "the image to the right",
                    "the image on the right side",
                    "the righthand image",
                }
            ):
                assert modules[j - 1].split("]")[0].split()[-1] in {"in", "of"}
                found_project = True
                new_modules = (
                    new_modules[:-1]
                    + [num_periods[j - 1] * "." + right_image_module]
                    + [
                        (num_periods[j - 1] + 1) * "."
                        + "op_find"
                        + "["
                        + " ".join(
                            modules[j - 1].split("]")[0].split("[")[1].split()[:-1]
                        )
                        + "]"
                    ]
                    + modules[j + 1 :]
                )
                break
            elif (
                j > 0
                and modules[j - 1].split("[")[0].strip(".") == "op_project"
                and modules[j].split("[")[0].strip(".") == "op_find"
                and modules[j].split("]")[0].split("[")[1].strip(".")
                in {
                    "left image",
                    "the left image",
                    "image on the left",
                    "the image on the left",
                    "the image to the left",
                    "the image on the left side",
                    "the lefthand image",
                }
            ):
                assert modules[j - 1].split("]")[0].split()[-1] in {"in", "of"}
                found_project = True
                new_modules = (
                    new_modules[:-1]
                    + [num_periods[j - 1] * "." + left_image_module]
                    + [
                        (num_periods[j - 1] + 1) * "."
                        + "op_find"
                        + "["
                        + " ".join(
                            modules[j - 1].split("]")[0].split("[")[1].split()[:-1]
                        )
                        + "]"
                    ]
                    + modules[j + 1 :]
                )
                break
            else:
                new_modules.append(modules[j])
        modules = new_modules
        if not found_project:
            break
    while True:
        num_periods = []
        for j in range(len(modules)):
            num_periods.append(len(modules[j]) - len(modules[j].strip(".")))
        found_if = False
        new_modules = []
        for j in range(len(modules)):
            if (
                modules[j].split("[")[0].strip(".") == "op_if"
                and "[" in modules[j]
                and " ".join(
                    modules[j].strip(".").split("[")[1].split("]")[0].split()[1:]
                )
                in {"in the right image", "in right image", "in the image on the right"}
            ):
                new_modules.append(num_periods[j] * "." + "op_exist")
                new_modules.append((num_periods[j] + 1) * "." + right_image_module)
                descendants = set(_get_descendants(modules, j))
                for k in range(j + 1, len(modules)):
                    if k in descendants:
                        new_modules.append("." + modules[k])
                    else:
                        new_modules.append(modules[k])
                found_if = True
                break
            elif (
                modules[j].split("[")[0].strip(".") == "op_if"
                and "[" in modules[j]
                and " ".join(
                    modules[j].strip(".").split("[")[1].split("]")[0].split()[1:]
                )
                in {"in the left image", "in left image", "in the image on the left"}
            ):
                new_modules.append(num_periods[j] * "." + "op_exist")
                new_modules.append((num_periods[j] + 1) * "." + left_image_module)
                descendants = set(_get_descendants(modules, j))
                for k in range(j + 1, len(modules)):
                    if k in descendants:
                        new_modules.append("." + modules[k])
                    else:
                        new_modules.append(modules[k])
                found_if = True
                break
            else:
                new_modules.append(modules[j])
        modules = new_modules
        if not found_if:
            break
    return modules


def num_quantifiers(modules):
    def get_module_list(index, info):
        if len(info) <= index:
            return []
        modules = [info[index][0]]
        if info[index][0].strip(".") == "op_if":
            if (
                len(info[index][1]) == 3
                and info[info[index][1][0]][0].strip(".") == "op_count"
                and info[info[index][1][-1]][0].strip(".") == "op_count"
            ):
                modules = [
                    info[index][0] + "[" + info[info[index][1][1]][0].strip(".") + "]"
                ]
                info[index][1] = [info[index][1][0], info[index][1][-1]]
        for c in info[index][1]:
            modules += get_module_list(c, info)
        return modules

    num_quant_change = True
    while num_quant_change:
        num_quant_change = False
        ex = modules
        module_info = []
        for i in range(len(ex)):
            mod = ex[i]
            children = []
            j = i + 1
            num_periods = len(ex[i]) - len(ex[i].strip("."))
            while j < len(ex) and len(ex[j]) - len(ex[j].strip(".")) >= num_periods + 1:
                if len(ex[j]) - len(ex[j].strip(".")) == num_periods + 1:
                    children.append(j)
                j += 1
            module_info.append([mod, children])
        new_ex = get_module_list(0, module_info)
        i = 0
        while i < len(new_ex):
            mod = new_ex[i]
            mod_without_periods = mod.strip(".")
            if mod_without_periods[:5] == "op_if":
                num_periods = mod.index("o")
                assert num_periods == len(mod) - len(mod_without_periods)
                j = i + 1
                count_flag = False
                count_num = 0
                while (
                    j < len(new_ex)
                    and len(new_ex[j]) - len(new_ex[j].strip(".")) >= num_periods + 1
                ):
                    without_periods_j = new_ex[j].strip(".")
                    if (
                        without_periods_j
                        in {"op_count", "op_sum", "op_difference", "op_division"}
                        and len(new_ex[j]) - len(without_periods_j) == num_periods + 1
                    ):
                        count_flag = True
                        count_num += 1
                    j += 1
                if count_num > 1:
                    if "is equal to" in mod or "is the same as" in mod:
                        new_ex[i] = mod.split("o")[0] + "op_equal"
                    elif "is higher than" in mod or "is more than" in mod:
                        new_ex[i] = mod.split("o")[0] + "op_greater"
                    elif "is less than" in mod:
                        new_ex[i] = mod.split("o")[0] + "op_less"
                mod = new_ex[i]
                if count_num == 1 and "[" in mod:
                    arg = mod.split("[")[1].split("]")[0]
                    module = "equal"
                    if "at least" in arg or "no less" in arg:
                        module = "greater_equal"
                    elif "at most" in arg or "no more" in arg:
                        module = "less_equal"
                    elif "is more" in arg or "is higher" in arg:
                        module = "greater"
                    elif "is less" in arg or "is lower" in arg:
                        module = "less"
                    tokens = arg.split()
                    num = None
                    for tok in tokens:
                        if tok.isnumeric():
                            num = int(tok)
                            break
                        elif tok in COUNTS:
                            num = COUNTS[tok]
                            break
                    if num is not None:
                        num_periods = len(new_ex[i]) - len(new_ex[i].strip("."))
                        if (
                            i + 2 < len(modules)
                            and new_ex[i + 2].strip(".") == "op_project[images of]"
                        ):
                            new_module = None
                            if (module == "greater_equal" and num == 1) or (
                                module == "equal" and num == 1
                            ):
                                new_module = "op_in_at_least_one_image"
                            elif module == "equal" and num == 2:
                                new_module = "op_in_each_image"
                            if new_module is not None:
                                j = i + 3
                                while (
                                    j < len(new_ex)
                                    and len(new_ex[j]) - len(new_ex[j].strip("."))
                                    > num_periods
                                ):
                                    new_ex[j] = new_ex[j][1:]
                                    j += 1
                                new_ex = (
                                    new_ex[:i]
                                    + ["." * num_periods + new_module]
                                    + ["." * (num_periods + 1) + "op_exist"]
                                    + new_ex[i + 3 :]
                                )
                                num_quant_change = True
                                continue
                        j = i + 1
                        while (
                            j < len(new_ex)
                            and len(new_ex[j]) - len(new_ex[j].strip(".")) > num_periods
                        ):
                            j += 1
                        new_ex = (
                            new_ex[:i]
                            + ["." * num_periods + "op_" + module]
                            + new_ex[i + 1 : j]
                            + ["." * (num_periods + 1) + "op_number[" + str(num) + "]"]
                            + new_ex[j:]
                        )
                        num_quant_change = True
                        break
            i += 1
        modules = new_ex
    return new_ex


def in_one_each_image(modules):
    while True:
        found_if_change = False
        for i in range(len(modules)):
            without_periods = modules[i].strip(".")
            num_periods = len(modules[i]) - len(without_periods)
            if (
                without_periods[:5] == "op_if"
                and "[" in without_periods
                and "num_images" in without_periods
            ):
                argument = without_periods.split("[")[1].split("]")[0]
                argument_parts = argument.split()
                if "one" in argument_parts:
                    new_module = "op_in_at_least_one_image"
                else:
                    new_module = "op_in_each_image"
                modules = (
                    modules[:i] + ["." * (num_periods) + new_module] + modules[i + 1 :]
                )
                found_if_change = True
                break
        if not found_if_change:
            break
    while True:
        found_project_change = False
        modules_without_periods = [module.strip(".") for module in modules]
        for i in range(len(modules)):
            without_periods = modules_without_periods[i]
            num_periods = len(modules[i]) - len(without_periods)
            if (
                i >= 3
                and without_periods.split("[")[0] == "op_project"
                and modules_without_periods[i - 1] == "is equal to"
                and "num_images" in without_periods
                and "op_find" in modules_without_periods[i - 2]
            ):
                find_parts = modules_without_periods[i - 2].split("[")
                find_argument = find_parts[1].split("]")[0]
                if (
                    find_parts[0] == "op_find"
                    and find_argument == "two"
                    and modules_without_periods[i - 3] == "op_if"
                ):
                    found_project_change = True
                    for j in range(i + 1, len(modules)):
                        if (
                            len(modules[j]) - len(modules_without_periods[j])
                            <= num_periods
                        ):
                            break
                        periods = "." * (
                            len(modules[j]) - len(modules_without_periods[j]) - 1
                        )
                        modules[j] = periods + modules_without_periods[j]
                    modules = (
                        modules[: i - 3]
                        + [
                            "."
                            * (
                                len(modules[i - 3])
                                - len(modules_without_periods[i - 3])
                            )
                            + "op_in_each_image"
                        ]
                        + modules[i + 1 :]
                    )
                    break
        if not found_project_change:
            break
    while True:
        found_string_match_change = False
        for i in range(len(modules)):
            without_periods = modules[i].strip(".")
            if without_periods.split("[")[0] == "op_if":
                num_periods = len(modules[i]) - len(without_periods)
                strings = [
                    "in at least one image",
                    "in an image",
                    "in one of the images",
                    "in at least one of the images",
                ]
                if any([s + "]" in without_periods for s in strings]):
                    j = i + 1
                    while (
                        j < len(modules)
                        and len(modules[j]) - len(modules[j].strip(".")) > num_periods
                    ):
                        modules[j] = "." + modules[j]
                        j += 1
                    modules = (
                        modules[:i]
                        + [
                            "." * num_periods + "op_in_at_least_one_image",
                            "." * (num_periods + 1) + "op_exist",
                        ]
                        + modules[i + 1 :]
                    )
                    found_string_match_change = True
                    break
        if not found_string_match_change:
            break
    return modules


def project_images_of_is_equal_to(modules):
    found_project_flag = True
    while found_project_flag:
        found_project_flag = False
        for i in range(len(modules)):
            without_periods = modules[i].strip(".")
            if without_periods[:5] == "op_if":
                num_periods = len(modules[i]) - len(without_periods)
                children = []
                children_indices = _get_children(modules, i)
                children = [modules[j] for j in children_indices]
                if i + 2 < len(modules) and "op_project[images of]" in modules[i + 2]:
                    if (
                        len(children) == 3
                        and "op_count" in children[0]
                        and ".is equal to" in children[1]
                        and (
                            "op_find[one]" in children[2]
                            or "op_find[two]" in children[2]
                        )
                    ):
                        assert children_indices[1] == children_indices[2] - 1
                        found_project_flag = True
                        new_module = "op_in_each_image"
                        if "op_find[one]" in children[2]:
                            new_module = "op_in_at_least_one_image"
                        for j in range(i + 1, len(modules)):
                            without_periods_j = modules[j].strip(".")
                            if len(modules[j]) - len(without_periods_j) > num_periods:
                                modules[j] = modules[j][1:]
                            else:
                                break
                        modules = (
                            modules[:i]
                            + [
                                "." * num_periods + new_module,
                                "." * (num_periods + 1) + "op_exist",
                            ]
                            + modules[i + 3 : children_indices[1]]
                            + modules[children_indices[2] + 1 :]
                        )
                        break
        if not found_project_flag:
            break
    return modules


def collect_attentions(modules):
    while True:
        filter_attentions_set = set()
        moved_attention = False
        for i in range(len(modules)):
            without_periods = modules[i].strip(".")
            num_periods = len(modules[i]) - len(without_periods)
            if without_periods[:3] == "op_":
                if "op_filter[" in without_periods:
                    filter_attentions_set.add(i)
                continue
            assert "[" not in without_periods and "]" not in without_periods
            for j in range(i - 1, -1, -1):
                without_periods_j = modules[j].strip(".")
                num_periods_j = len(modules[j]) - len(without_periods_j)
                if (
                    num_periods_j == num_periods - 1
                    and "op_filter" in without_periods_j
                ):
                    if (
                        j in filter_attentions_set
                        and without_periods_j != "op_filter[that]"
                        and (
                            i + 1 >= len(modules)
                            or len(modules[i + 1]) - len(modules[i + 1].strip("."))
                            >= num_periods
                        )
                    ):
                        for k in range(i - 1, -1, -1):
                            without_periods_k = modules[k].strip(".")
                            num_periods_k = len(modules[k]) - len(without_periods_k)
                            if num_periods_k == num_periods:
                                break
                        for k2 in range(k, len(modules)):
                            without_periods_k2 = modules[k2].strip(".")
                            num_periods_k2 = len(modules[k2]) - len(without_periods_k2)
                            modules[k2] = "." + modules[k2]
                            if num_periods_k2 == num_periods_k - 1:
                                modules[k2] = modules[k2][1:]
                                break
                        modules = (
                            modules[:k]
                            + [
                                "." * num_periods
                                + "op_filter"
                                + "["
                                + without_periods
                                + "]"
                            ]
                            + modules[k:i]
                            + modules[i + 1 :]
                        )
                        filter_attentions_set.add(k)
                        break
                if num_periods_j == num_periods - 1:
                    assert without_periods_j[:3] == "op_"
                    if "op_filter" in without_periods_j:
                        filter_attentions_set.add(j)
                    if "]" in modules[j]:
                        modules[j] = modules[j][:-1] + " " + without_periods + "]"
                    else:
                        modules[j] = modules[j] + "[" + without_periods + "]"
                    modules = modules[:i] + modules[i + 1 :]
                    break
            moved_attention = True
            break
        if not moved_attention:
            break
    return modules


def fix_one_other_image(modules):
    found_find = True
    added_top_module = False
    while found_find:
        found_find = False
        found_other = False
        for i in range(len(modules)):
            without_periods = modules[i].strip(".")
            if (
                without_periods[:7] == "op_find"
                and "[" in without_periods
                and without_periods.split("[")[1].split("]")[0] == "one image"
            ):
                if "op_project" in modules[i - 1]:
                    arg = modules[i - 1].split("[")[1].split("]")[0]
                    if arg[-3:] in {" in", " of"}:
                        arg = arg[:-3]
                    num_periods = len(modules[i]) - len(without_periods)
                    modules = (
                        modules[: i - 1]
                        + [
                            "." * (num_periods - 1) + "op_in_one_image",
                            "." * (num_periods) + "op_find[" + arg + "]",
                        ]
                        + modules[i + 1 :]
                    )
                    found_find = True
                    break
            elif (
                without_periods[:7] == "op_find"
                and "[" in without_periods
                and without_periods.split("[")[1].split("]")[0] == "other image"
            ):
                if "op_project" in modules[i - 1]:
                    arg = modules[i - 1].split("[")[1].split("]")[0]
                    if arg[-3:] in {" in", " of"}:
                        arg = arg[:-3]
                    num_periods = len(modules[i]) - len(without_periods)
                    modules = (
                        modules[: i - 1]
                        + [
                            "." * (num_periods - 1) + "op_in_other_image",
                            "." * (num_periods) + "op_find[" + arg + "]",
                        ]
                        + modules[i + 1 :]
                    )
                    found_find = True
                    found_other = True
                    break
        if found_other and not added_top_module:
            modules = ["op_in_one_other_image"] + ["." + mod for mod in modules]
            added_top_module = True
        if not found_find:
            break
    return modules


def format_attentions(
    modules, sentence, use_number_literals=False, add_number_attention=False
):
    parts = deepcopy(modules)
    tokenized_sentence = [str(tok) for tok in tokenizer.tokenize(sentence)]

    def join_indices(indices):
        return "[" + ",".join([str(num) for num in indices]) + "]"

    for i in range(len(parts)):
        op_name_with_periods = parts[i].split("[")[0]
        if use_number_literals and parts[i].strip(".").split("[")[0] == "op_number":
            argument = parts[i].split("[")[1].split("]")[0]
            op_name_with_periods = parts[i].split("o")[0] + argument
            if not add_number_attention:
                parts[i] = op_name_with_periods
                continue
        if "[" in parts[i]:
            words = parts[i].split("[")[1].split("]")[0]
            tokenized_words = [str(tok) for tok in tokenizer.tokenize(words)]
            num_words = len(tokenized_words)

            is_number_module = parts[i].strip(".").split("[")[0] == "op_number"
            word_in_sentence = words in sentence
            number_in_sentence = (
                is_number_module
                and words in COUNTS_REV
                and COUNTS_REV[words] in sentence
            )

            index = None
            if word_in_sentence or number_in_sentence:
                for i2 in range(len(tokenized_sentence)):
                    if i2 + num_words <= len(tokenized_sentence):
                        if (
                            tokenized_sentence[i2 : i2 + num_words] == tokenized_words
                            or number_in_sentence
                            and tokenized_sentence[i2 : i2 + num_words][0]
                            == COUNTS_REV[words]
                        ):
                            index = i2
                            break
                if index is not None:
                    parts[i] = op_name_with_periods + join_indices(
                        range(index, index + num_words)
                    )
            if index is None:
                indices = []
                for word in tokenized_words:
                    if word in tokenized_sentence:
                        indices.append(tokenized_sentence.index(word))
                    else:
                        lemma = wnl.lemmatize(word)
                        if lemma in tokenized_sentence:
                            indices.append(tokenized_sentence.index(lemma))
                if len(indices) > 0:
                    parts[i] = op_name_with_periods + join_indices(indices)
                elif is_number_module:
                    parts[i] = op_name_with_periods
                elif "op_filter" in parts[i] and len(_get_children(parts, i)) == 1:
                    descendants = _get_descendants(parts, i)
                    parts = (
                        parts[:i]
                        + [""]
                        + [parts[desc][1:] for desc in descendants]
                        + parts[max(descendants) + 1 :]
                    )
                else:
                    # print(sentence, parts, parts[i])
                    parts[i] = parts[i].split("[")[0]
        if "op_filter" in parts[i]:
            children = _get_children(parts, i)
            # if len(children) not in {1, 2}:
            #     print(sentence)
            # assert len(children) in {1, 2}
            if len(children) == 2:
                parts[i] = (
                    parts[i].split("op_filter")[0]
                    + "op_with_relation"
                    + parts[i].split("op_filter")[1]
                )
    new_parts = []
    for part in parts:
        if len(part) > 0:
            new_parts.append(part)
    parts = new_parts

    parts_tokens = deepcopy(parts)
    for i, p in enumerate(parts_tokens):
        attention = None if "[" not in p else p.split("[")[1].split("]")[0]
        if not attention:
            continue
        try:
            attention_tokens = [
                tokenized_sentence[int(j)] for j in attention.split(",")
            ]
        except ValueError:
            continue
        prefix = p.split("[")[0]
        parts_tokens[i] = prefix + "[" + ",".join(attention_tokens) + "]"
    return parts, parts_tokens, tokenized_sentence


def remove_op_prefix(modules):
    new_modules = []
    for mod in modules:
        without_periods = mod.strip(".")
        if without_periods[:3] == "op_":
            new_modules.append(
                "." * (len(mod) - len(without_periods)) + without_periods[3:]
            )
        else:
            new_modules.append(mod)
    return new_modules


def remove_sum_one_argument(modules):
    while True:
        found_single_arg_sum = False
        for i in range(len(modules)):
            if modules[i].strip(".")[:6] == "op_sum":
                children = _get_children(modules, i)
                if len(children) == 1:
                    descendants = _get_descendants(modules, i)
                    first_descendant = min(descendants)
                    last_descendant = max(descendants)
                    assert first_descendant == i + 1
                    assert set(descendants) == set(
                        range(first_descendant, last_descendant + 1)
                    )
                    modules = (
                        modules[:i]
                        + [
                            mod[1:]
                            for mod in modules[first_descendant : last_descendant + 1]
                        ]
                        + modules[last_descendant + 1 :]
                    )
                    found_single_arg_sum = True
                    break
        if not found_single_arg_sum:
            break
    return modules


def handle_remaining_ifs(modules):
    while True:
        found_if = False
        for i in range(len(modules)):
            without_periods = modules[i].strip(".")
            if without_periods[:5] == "op_if":
                children = _get_children(modules, i)
                if len(children) == 2:
                    descendants = _get_descendants(modules, i)
                    first_desc = min(descendants)
                    last_desc = max(descendants)
                    argument = ""
                    if "[" in without_periods:
                        argument = (
                            "[" + without_periods.split("[")[1].split("]")[0] + "]"
                        )
                    modules = (
                        modules[:i]
                        + ["." * (len(modules[i]) - len(without_periods)) + "op_exist"]
                        + [
                            "." * (len(modules[i]) - len(without_periods) + 1)
                            + "op_with_relation"
                            + argument
                        ]
                        + ["." + mod for mod in modules[first_desc : last_desc + 1]]
                        + modules[last_desc + 1 :]
                    )
                    found_if = True
                    break
                if "[" in modules[i]:
                    argument = modules[i].split("[")[1].split("]")[0]
                    if argument == "an image":
                        modules = (
                            modules[:i]
                            + [
                                "." * (len(modules[i]) - len(without_periods))
                                + "op_exist"
                            ]
                            + modules[i + 1 :]
                        )
                        found_if = True
                        break
                    elif "in one image" in argument:
                        modules = (
                            modules[:i]
                            + [
                                "." * (len(modules[i]) - len(without_periods))
                                + "op_in_at_least_one_image",
                                "." * (len(modules[i]) - len(without_periods) + 1)
                                + "op_exist",
                            ]
                            + ["." + mod for mod in modules[i + 1 :]]
                        )
                        found_if = True
                        break
                    elif "in each image" in argument:
                        modules = (
                            modules[:i]
                            + [
                                "." * (len(modules[i]) - len(without_periods))
                                + "op_in_each_image",
                                "." * (len(modules[i]) - len(without_periods) + 1)
                                + "op_exist",
                            ]
                            + ["." + mod for mod in modules[i + 1 :]]
                        )
                        found_if = True
                        break
                    else:
                        descendants = _get_descendants(modules, i)
                        first_desc = min(descendants)
                        last_desc = max(descendants)
                        argument = ""
                        if "[" in without_periods:
                            argument = (
                                "[" + without_periods.split("[")[1].split("]")[0] + "]"
                            )
                        modules = (
                            modules[:i]
                            + [
                                "." * (len(modules[i]) - len(without_periods))
                                + "op_exist"
                            ]
                            + [
                                "." * (len(modules[i]) - len(without_periods) + 1)
                                + "op_filter"
                                + argument
                            ]
                            + ["." + mod for mod in modules[first_desc : last_desc + 1]]
                            + modules[last_desc + 1 :]
                        )
                        found_if = True
                        break
                else:
                    modules = (
                        modules[:i]
                        + ["." * (len(modules[i]) - len(without_periods)) + "op_exist"]
                        + modules[i + 1 :]
                    )
                    found_if = True
                    break
        if not found_if:
            break
    return modules


def convert_number_modules_to_literals(sentence, modules):
    parts = deepcopy(modules)
    tokenized_sentence = [str(tok) for tok in tokenizer.tokenize(sentence)]
    number_failed = False
    for i in range(len(parts)):
        if "[" in parts[i] and parts[i].strip(".").split("[")[0] == "number":
            num_periods = len(parts[i]) - parts[i][::-1].index(".")
            number_tokens_ids = parts[i].split("[")[1].split("]")[0].split(",")
            number_tokens_ids = [int(n) for n in number_tokens_ids]
            number_tokens = " ".join([tokenized_sentence[n] for n in number_tokens_ids])
            if number_tokens in COUNTS_REV:
                parts[i] = num_periods * "." + number_tokens
            elif number_tokens in COUNTS:
                parts[i] = num_periods * "." + str(COUNTS[number_tokens])
            else:
                number_failed = True
    return parts if not number_failed else None


if __name__ == "__main__":
    f = open(sys.argv[1])
    reader = csv.DictReader(f)
    fout = open("nlvr2_processed_annotations.csv", "w")
    trainf = open("train_annotations_from_qdmr_rawnumber.tsv", "w")
    devf = open("dev_annotations_from_qdmr_rawnumber.tsv", "w")
    testf = open("test1_annotations_from_qdmr_rawnumber.tsv", "w")
    FIELDNAMES = [
        "sentence",
        "annotation",
        "annotation_tokens",
        "tokens",
        "qdmr",
        "problem",
    ]
    train_out = csv.DictWriter(trainf, fieldnames=FIELDNAMES, delimiter="\t")
    dev_out = csv.DictWriter(devf, fieldnames=FIELDNAMES, delimiter="\t")
    test_out = csv.DictWriter(testf, fieldnames=FIELDNAMES, delimiter="\t")
    train_out.writeheader()
    dev_out.writeheader()
    test_out.writeheader()
    for i, l in enumerate(reader):
        print(l.keys())
        sentence = l["question_text"][3:]
        program = l["program"]
        modules = program.split("\n")
        if modules == ["ERROR"]:
            continue
        if any(
            [
                ("[" in mod and "]" not in mod) or ("[" not in mod and "]" in mod)
                for mod in modules
            ]
        ):
            continue
        assert modules[-1] == ""
        # print(l[0])

        # try:
        print(modules)
        modules = modules[:-1]
        modules = in_left_right_image(modules)
        modules = num_quantifiers(modules)
        modules = in_one_each_image(modules)
        modules = project_images_of_is_equal_to(modules)
        modules = collect_attentions(modules)
        modules = remove_sum_one_argument(modules)
        modules = handle_remaining_ifs(modules)
        modules = fix_one_other_image(modules)
        modules, modules_tokens, tokens = format_attentions(
            modules, sentence, use_number_literals=True, add_number_attention=True
        )
        # except Exception:
        #     print("some error")
        #     continue
        tokens = [str(tok) for tok in tokenizer.tokenize(sentence)]
        modules = remove_op_prefix(modules)
        fout.write(
            json.dumps(
                {
                    "identifier": l["question_id"],
                    "sentence": sentence,
                    "program": modules,
                }
            )
            + "\n"
        )
        if "train" in l["question_id"]:
            train_out.writerow(
                {
                    "sentence": sentence,
                    "tokens": {j: tokens[j] for j in range(len(tokens))},
                    "annotation": "\n".join(modules),
                    "annotation_tokens": "\n".join(modules_tokens),
                    "qdmr": 1,
                }
            )
        elif "dev" in l["question_id"]:
            dev_out.writerow(
                {
                    "sentence": sentence,
                    "tokens": {j: tokens[j] for j in range(len(tokens))},
                    "annotation": "\n".join(modules),
                    "annotation_tokens": "\n".join(modules_tokens),
                    "qdmr": 1,
                }
            )
        elif "test" in l["question_id"]:
            test_out.writerow(
                {
                    "sentence": sentence,
                    "tokens": {j: tokens[j] for j in range(len(tokens))},
                    "annotation": "\n".join(modules),
                    "annotation_tokens": "\n".join(modules_tokens),
                    "qdmr": 1,
                }
            )
    fout.close()
    trainf.close()
    devf.close()
    testf.close()
