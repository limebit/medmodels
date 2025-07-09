# Attributes & Values

One of the main strengths of the query engine is the possibility of diving deeper into the MedRecords' attributes and values. We can access them by using different return types - in the [Query Engine Introduction](index.md), we mainly used *indices* as the return type for each query.

## Inspecting Attributes Names

Each node can have a variety of different attributes, each one of the holding an assigned [`MedRecordValue`](medmodels.medrecord.types.MedRecordValue){target="_blank"}. We can look at the attributes of each node by using the method [`attributes()`](medmodels.medrecord.querying.NodeOperand.attributes){target="_blank"}:

```{exec-literalinclude} scripts/attributes_and_values.py
---
language: python
setup-lines: 1-14
lines: 17-23
---
```

:::{dropdown} Methods used in the snippet

- [`in_group()`](medmodels.medrecord.querying.NodeOperand.in_group){target="_blank"} : Query nodes that belong to that group.
- [`attributes()`](medmodels.medrecord.querying.NodeOperand.attributes){target="_blank"} : Query the attribute names of each node.
- [`query_nodes()`](medmodels.medrecord.medrecord.MedRecord.query_nodes){target="_blank"} : Retrieves information on the nodes from the MedRecord given the query.

:::

You can also do operations on them, like checking how many attributes each node has, thanks to the [`count()`](medmodels.medrecord.querying.NodeMultipleAttributesWithIndexOperand.count){target="_blank"} method:

```{exec-literalinclude} scripts/attributes_and_values.py
---
language: python
setup-lines: 1-14
lines: 26-35
---
```

:::{dropdown} Methods used in the snippet

- [`in_group()`](medmodels.medrecord.querying.NodeOperand.in_group){target="_blank"} : Query nodes that belong to that group.
- [`attributes()`](medmodels.medrecord.querying.NodeOperand.attributes){target="_blank"} : Query the attribute names of each node.
- [`count()`](medmodels.medrecord.querying.NodeMultipleAttributesWithIndexOperand.count){target="_blank"} : Query how many attributes each node has.
- [`query_nodes()`](medmodels.medrecord.medrecord.MedRecord.query_nodes){target="_blank"} : Retrieves information on the nodes from the MedRecord given the query.

:::


## Inspecting Attributes Values

As said before we can look for specific values within our MedRecord, using the method [`attribute()`](medmodels.medrecord.querying.NodeOperand.attribute){target="_blank"}. For instance, we can search for the maximum `age` in our patients, and we will get the node ID of the patient with the highest age, and what that age is.

```{exec-literalinclude} scripts/attributes_and_values.py
---
language: python
setup-lines: 1-14
lines: 38-46
---
```

:::{dropdown} Methods used in the snippet

- [`attribute()`](medmodels.medrecord.querying.NodeOperand.attribute){target="_blank"} : Returns a [`NodeMultipleValuesWithIndexOperand()`](medmodels.medrecord.querying.NodeMultipleValuesWithIndexOperand){target="_blank"} to query on the values of the nodes for that attribute.
- [`max()`](medmodels.medrecord.querying.NodeMultipleValuesWithIndexOperand.max){target="_blank"}: Returns a [`NodeSingleValueWithIndexOperand()`](medmodels.medrecord.querying.NodeSingleValueWithIndexOperand){target="_blank"} holding the node index and value pair holding the maximum value for that attribute.
- [`query_nodes()`](medmodels.medrecord.medrecord.MedRecord.query_nodes){target="_blank"} : Retrieves information on the nodes from the MedRecord given the query.

:::

## Advanced Query Operations

In case, for instance, that you do not know whether there are different ways to assign the `gender` attribute across the [`MedRecord`](medmodels.medrecord.medrecord.MedRecord){target="_blank"} (with leading/trailing whitespaces or formatted in lower/uppercase), you can modify the value of the attributes of a node/edge inside the query.

:::{note}

It is important to note that modifying these values **does not** change the actual value of the attributes within the [`MedRecord`](medmodels.medrecord.medrecord.MedRecord){target="_blank"}: it just changes the value of those variables in the query.

:::

You can also perform mathematical calculations like [`mean()`](medmodels.medrecord.querying.NodeMultipleValuesWithIndexOperand.mean){target="_blank"}, [`median()`](medmodels.medrecord.querying.NodeMultipleValuesWithIndexOperand.median){target="_blank"} or [`min()`](medmodels.medrecord.querying.NodeMultipleValuesWithIndexOperand.min){target="_blank"} and assign them to a variable. Also, you can keep manipulating the operand, like in the following example, where we are subtracting _5_ years from the `mean_age` to query on that value.

In the result, we can see the only patient whose age is less than five years under the mean age and what that value is.


```{exec-literalinclude} scripts/attributes_and_values.py
---
language: python
setup-lines: 1-14
lines: 50-69
---
```

:::{dropdown} Methods used in the snippet

- [`in_group()`](medmodels.medrecord.querying.NodeOperand.in_group){target="_blank"} : Query nodes that belong to that group.
- [`index()`](medmodels.medrecord.querying.NodeOperand.index){target="_blank"}: Returns a [`NodeIndicesOperand`](medmodels.medrecord.querying.NodeIndicesOperand){target="_blank"} representing the indices of the nodes queried.
- [`contains()`](medmodels.medrecord.querying.NodeIndexOperand.contains){target="_blank"} : Query node indices containing that argument.
- [`attribute()`](medmodels.medrecord.querying.NodeOperand.attribute){target="_blank"} : Returns a [`NodeMultipleValuesWithIndexOperand`](medmodels.medrecord.querying.NodeMultipleValuesWithIndexOperand){target="_blank"} to query on the values of the nodes for that attribute.
- [`lowercase()`](medmodels.medrecord.querying.NodeMultipleValuesWithIndexOperand.lowercase){target="_blank"} : Converts the values that are strings to lowercase.
- [`trim()`](medmodels.medrecord.querying.NodeMultipleValuesWithIndexOperand.trim){target="_blank"} : Removes leading and trailing whitespacing from the values.
- [`equal_to()`](medmodels.medrecord.querying.NodeMultipleValuesWithIndexOperand.equal_to){target="_blank"} : Query values equal to that value.
- [`has_attribute()`](medmodels.medrecord.querying.NodeOperand.has_attribute){target="_blank"} : Query nodes that have that attribute.
- [`mean()`](medmodels.medrecord.querying.NodeMultipleValuesWithIndexOperand.mean){target="_blank"}: Returns a [`NodeSingleValueWithoutIndexOperand`](medmodels.medrecord.querying.NodeSingleValueWithoutIndexOperand){target="_blank"} containing the mean of those values.
- [`subtract()`](medmodels.medrecord.querying.NodeSingleValueWithoutIndexOperand.subtract){target="_blank"} : Subtract the argument from the single value operand.
- [`less_than()`](medmodels.medrecord.querying.NodeMultipleValuesWithIndexOperand.less_than){target="_blank"} : Query values that are less than that value.
- [`query_nodes()`](medmodels.medrecord.medrecord.MedRecord.query_nodes){target="_blank"} : Retrieves information on the nodes from the MedRecord given the query.

:::

:::{note}
Query methods used for changing the operands cannot be assigned to variables for further querying, since their return type is `None`. The following code snippet shows an example, where the variable `gender_lowercase` evaluates to None. An `AttributeError` is thrown as a consequence when trying to further query with the `equal_to` querying method:

```{exec-literalinclude} scripts/attributes_and_values.py
---
language: python
setup-lines: 1-14
lines: 73-82
expect-error: PanicException
---
```

The concatenation of querying methods also throws an error:

```{exec-literalinclude} scripts/attributes_and_values.py
---
language: python
setup-lines: 1-14
lines: 86-94
expect-error: PanicException
---
```

**Correct implementation**:

```{exec-literalinclude} scripts/attributes_and_values.py
---
language: python
setup-lines: 1-14
lines: 98-107
---
```

:::

## Full example Code

The full code examples for this chapter can be found here:

```{literalinclude} scripts/attributes_and_values.py
---
language: python
lines: 3-107
---
```