# pyright: reportAttributeAccessIssue=false
# ruff: noqa: D100, D103
from medmodels import MedRecord
from medmodels.medrecord.querying import (
    NodeAttributesTreeOperand,
    NodeIndicesOperand,
    NodeMultipleAttributesWithIndexOperand,
    NodeMultipleValuesWithIndexOperand,
    NodeOperand,
    NodeSingleValueWithIndexOperand,
    NodeSingleValueWithoutIndexOperand,
)

medrecord = MedRecord().from_simple_example_dataset()


def query_node_attribute_names(node: NodeOperand) -> NodeAttributesTreeOperand:
    node.in_group("patient")

    return node.attributes()


medrecord.query_nodes(query_node_attribute_names)


def query_node_attributes_count(
    node: NodeOperand,
) -> NodeMultipleAttributesWithIndexOperand:
    node.in_group("patient")
    attributes = node.attributes()

    return attributes.count()


medrecord.query_nodes(query_node_attributes_count)


def query_node_max_age(
    node: NodeOperand,
) -> NodeSingleValueWithIndexOperand:
    age = node.attribute("age")

    return age.max()


medrecord.query_nodes(query_node_max_age)


# Advanced node query
def query_node_male_patient_under_mean(
    node: NodeOperand,
) -> tuple[NodeMultipleValuesWithIndexOperand, NodeSingleValueWithoutIndexOperand]:
    node.in_group("patient")
    node.index().contains("pat")

    gender = node.attribute("gender")
    gender.lowercase()  # Converts the string to lowercase
    gender.trim()  # Removes leading and trailing whitespaces
    gender.equal_to("m")

    node.has_attribute("age")
    mean_age = node.attribute("age").mean()
    mean_age.subtract(5)  # Subtract 5 from the mean age
    node.attribute("age").less_than(mean_age)

    return node.attribute("age"), mean_age


medrecord.query_nodes(query_node_male_patient_under_mean)


# Incorrect implementation because the querying methods are assigned to a variable
def query_operand_assigned(node: NodeOperand) -> NodeIndicesOperand:
    gender_lowercase = node.attribute(
        "gender"
    ).lowercase()  # Assigning the querying method to a variable
    gender_lowercase.equal_to("m")

    return node.index()


medrecord.query_nodes(query_operand_assigned)


# Incorrect implementation because the querying methods are concatenated
def query_operands_concatenated(node: NodeOperand) -> NodeIndicesOperand:
    gender = node.attribute("gender")
    gender.lowercase().trim()  # Concatenating the querying methods
    gender.equal_to("m")

    return node.index()


medrecord.query_nodes(query_operands_concatenated)


# Correct implementation
def query_correct_implementation(node: NodeOperand) -> NodeIndicesOperand:
    gender = node.attribute("gender")
    gender.lowercase()
    gender.trim()
    gender.equal_to("m")

    return node.index()


medrecord.query_nodes(query_correct_implementation)
