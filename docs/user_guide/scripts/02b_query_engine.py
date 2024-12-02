from medmodels import MedRecord
from medmodels.medrecord.querying import EdgeOperand, NodeOperand

medrecord = MedRecord().from_example_dataset()


# Basic node query
def query_node_in_patient(node: NodeOperand):
    node.in_group("patient")


medrecord.select_nodes(query_node_in_patient)


# Intermediate node query
def query_node_patient_older_than_30(node: NodeOperand):
    node.in_group("patient")
    node.index().contains("pat")

    node.has_attribute("age")
    node.attribute("age").greater_than(30)


medrecord.select_nodes(query_node_patient_older_than_30)


# Reusing node query
def query_node_reused(node: NodeOperand):
    query_node_in_patient(node)
    node.index().contains("pat")

    node.has_attribute("age")
    node.attribute("age").greater_than(30)


medrecord.select_nodes(query_node_reused)


# Advanced node query
def query_node_male_patient_under_mean(node: NodeOperand):
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


medrecord.select_nodes(query_node_male_patient_under_mean)


# Node query with neighbors function
def query_node_neighbors(node: NodeOperand):
    query_node_patient_older_than_30(node)

    description_neighbors = node.neighbors().attribute("description")
    description_neighbors.lowercase()
    description_neighbors.contains("fentanyl")


medrecord.select_nodes(query_node_neighbors)


# Basic edge query
def query_edge_patient_drug(edge: EdgeOperand):
    edge.in_group("patient_drug")


edges = medrecord.select_edges(query_edge_patient_drug)
edges[0:5]


# Advanced edge query
def query_edge_old_patient_cheap_insulin(edge: EdgeOperand):
    edge.in_group("patient_drug")
    edge.attribute("cost").less_than(200)

    edge.source_node().attribute("age").is_max()
    edge.target_node().attribute("description").contains("insulin")


medrecord.select_edges(query_edge_old_patient_cheap_insulin)


# Combined node and edge query
def query_edge_combined(edge: EdgeOperand):
    edge.in_group("patient_drug")
    edge.attribute("cost").less_than(200)
    edge.attribute("quantity").equal_to(1)


def query_node_combined(node: NodeOperand):
    node.in_group("patient")
    node.attribute("age").is_int()
    node.attribute("age").greater_than(30)
    node.attribute("gender").equal_to("M")

    query_edge_combined(node.edges())


medrecord.select_nodes(query_node_combined)


# Either/or query
def query_edge_either(edge: EdgeOperand):
    edge.in_group("patient_drug")
    edge.attribute("cost").less_than(200)
    edge.attribute("quantity").equal_to(1)


def query_edge_or(edge: EdgeOperand):
    edge.in_group("patient_drug")
    edge.attribute("cost").less_than(200)
    edge.attribute("quantity").equal_to(12)


def query_node_either_or(node: NodeOperand):
    node.in_group("patient")
    node.attribute("age").greater_than(30)

    node.edges().either_or(query_edge_either, query_edge_or)


medrecord.select_nodes(query_node_either_or)


# Exclude query
def query_node_exclude(node: NodeOperand):
    node.in_group("patient")
    node.exclude(query_node_either_or)


medrecord.select_nodes(query_node_exclude)


# Clone query
def query_node_clone(node: NodeOperand):
    node.in_group("patient")
    node.index().contains("pat")

    mean_age_original = node.attribute("age").mean()
    mean_age_clone = mean_age_original.clone()  # Clone the mean age

    # Subtract 5 fom the cloned mean age (original remains unchanged)
    mean_age_clone.subtract(5)

    node.attribute("age").less_than(mean_age_original)  # Mean age
    node.attribute("age").greater_than(mean_age_clone)  # Mean age minus 5


medrecord.select_nodes(query_node_clone)

# Node queries as function arguments
medrecord.node[query_node_either_or]
medrecord.groups_of_node(query_node_patient_older_than_30)
medrecord.edge_endpoints(query_edge_old_patient_cheap_insulin)
