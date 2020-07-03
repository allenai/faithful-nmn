from typing import List

import numpy as np


def annotation_to_lisp_exp(annotation: str) -> str:
    # TODO: Remove this hard-coded fix
    annotation = annotation.replace("and\n", "bool_and\n")
    annotation = annotation.replace("or\n", "bool_or\n")

    expressions = annotation.split("\n")
    output_depth = 0
    output = []

    def count_depth(exp: str):
        """count the depth of this expression. Every dot in the prefix symbols a depth entry."""
        return len(exp) - len(exp.lstrip("."))

    def strip_attention(exp: str):
        """remove the [attention] part of the expression"""
        if "[" in exp:
            return exp[: exp.index("[")]
        else:
            return exp

    for i, exp in enumerate(expressions):
        depth = count_depth(exp)
        if i + 1 < len(expressions):
            next_expression_depth = count_depth(expressions[i + 1])
        else:
            next_expression_depth = 0

        output.append("(")

        exp = strip_attention(exp)
        exp = exp.lstrip(".")
        output.append(exp)

        if next_expression_depth <= depth:
            # current clause should be closed
            output.append(")")

        while next_expression_depth < depth:
            # close until currently opened depth
            output.append(")")
            depth -= 1

        output_depth = depth

    while 0 < output_depth:
        output.append(")")
        output_depth -= 1

    # now make sure there's no one-expression in a parentheses (e.g. "(exist (find))" which should be "(exist find)")
    i = 0
    new_output = []
    while i < len(output):
        exp = output[i]
        if i + 2 >= len(output):
            new_output.append(exp)
            i += 1
            continue

        exp1 = output[i + 1]
        exp2 = output[i + 2]

        if exp == "(" and exp1 not in ["(", ")"] and exp2 == ")":
            new_output.append(exp1)
            i += 2
        else:
            new_output.append(exp)

        i += 1

    output = " ".join(new_output)
    output = output.replace("( ", "(")
    output = output.replace(" )", ")")
    return output


def annotation_to_module_attention(annotation: str) -> List:
    """
    retrieves the raw annotation string and extracts the word indices attention for each module
    """
    lines = annotation.split("\n")
    attn_supervision = []
    for line in lines:
        # We assume valid input, that is, each line either has no brackets at all,
        # or has '[' before ']', where there are numbers separated by commas between.
        if "[" in line:
            start_i = line.index("[")
            end_i = line.index("]")
            module = line[:start_i].strip(".")
            sentence_indices = line[start_i + 1 : end_i].split(",")
            sentence_indices = [ind.strip() for ind in sentence_indices]

            attn_supervision.append((module, sentence_indices))
    return attn_supervision


def target_sequence_to_target_boxes(target_sequence, module_boxes, children) -> List:
    intersected_module_boxes = {}
    for j in range(len(module_boxes) - 1, -1, -1):
        if module_boxes[j][1] == "project":
            assert len(children[module_boxes[j][0]]) == 1
            child = children[module_boxes[j][0]][0]
            new_boxes = [[], []]
            for img in range(2):
                if len(intersected_module_boxes[child][2][img]) > 0:
                    new_boxes[img] = module_boxes[j][2][img]
            intersected_module_boxes[module_boxes[j][0]] = (
                module_boxes[j][0],
                module_boxes[j][1],
                new_boxes,
            )
            continue
        elif module_boxes[j][1] == "filter":
            assert len(children[module_boxes[j][0]]) == 1
        elif module_boxes[j][1] == "with_relation":
            assert len(children[module_boxes[j][0]]) == 2
        else:
            intersected_module_boxes[module_boxes[j][0]] = module_boxes[j]
            continue
        child = children[module_boxes[j][0]][0]
        new_boxes = []
        for img_num, img in enumerate(module_boxes[j][2]):
            new_boxes.append([])
            for box in img:
                if box in intersected_module_boxes[child][2][img_num]:
                    new_boxes[-1].append(box)
        intersected_module_boxes[module_boxes[j][0]] = (
            module_boxes[j][0],
            module_boxes[j][1],
            new_boxes,
        )
    keys = sorted(list(intersected_module_boxes.keys()))
    module_boxes = [intersected_module_boxes[j] for j in keys]
    target_boxes = [[] for _ in range(len(target_sequence))]
    target_counts = [-1 for _ in range(len(target_sequence))]
    module_box_i = 0
    for action_i, action in enumerate(target_sequence):
        rhs = action.split(" -> ")[1]
        if module_box_i < len(module_boxes):
            _, module_name, boxes = module_boxes[module_box_i]
            if rhs == module_name:
                if module_name == "count":
                    target_counts[action_i] = boxes
                else:
                    target_boxes[action_i] = [
                        [
                            [
                                float(box["left"]),
                                float(box["top"]),
                                float(box["left"]) + float(box["width"]),
                                float(box["top"]) + float(box["height"]),
                            ]
                            for box in img
                        ]
                        for img in boxes
                    ]
                module_box_i += 1
    # assert all([len(imgs) == 2 for imgs in target_boxes])
    return target_boxes, target_counts


def target_sequence_to_target_attn(
    target_sequence, module_attn, full_array=False
) -> List:
    """
    retrieves target sequence and word indices attention for each module,
    returns the attention for each one of the actions
    """
    target_attn = [[-1]] * len(target_sequence)
    if full_array:
        target_attn = [np.zeros((len(module_attn[0][1]))) for _ in target_sequence]
    module_attn_i = 0
    for action_i, action in enumerate(target_sequence):
        rhs = action.split(" -> ")[1]
        if module_attn_i < len(module_attn):
            module_name, attn = module_attn[module_attn_i]
            if rhs == module_name:
                if full_array:
                    target_attn[action_i] = attn
                else:
                    target_attn[action_i] = [int(a) for a in attn]
                module_attn_i += 1
    return target_attn
