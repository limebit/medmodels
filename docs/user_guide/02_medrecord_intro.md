# MedRecord

## Preface

Just like every major library has its core class — [PyTorch](https://pytorch.org/) has `torch.Tensor`, [Numpy](https://numpy.org/) has `np.array` — MedModels revolves around `mm.MedRecord` as its central object.

MedModels delivers advanced data analytics methods out-of-the-box by utilizing a structured approach to data storage. This is enabled by the MedRecord class, which organizes data of any complexity within a graph structure. With its Rust backend implementation, MedRecord guarantees high performance, even when working with extremely large datasets.


```{literalinclude} scripts/02_medrecord_intro.py
---
language: python
lines: 3
---
```

## Adding Nodes to a MedRecord

Let's begin by introducing some sample medical data:

:::{list-table} Patients
:widths: 15 15 15 15
:header-rows: 1

*   - ID
    - Age
    - Sex
    - Loc
*   - Patient 01
    - 72
    - M
    - USA
*   - Patient 02
    - 74
    - M
    - USA
*   - Patient 03
    - 64
    - F
    - GER
:::

This data, stored for example in a Pandas DataFrame, looks like this:

```{literalinclude} scripts/02_medrecord_intro.py
---
language: python
lines: 5-13
---
```

To work with this data in MedModels, we'll have to create a MedRecord. The MedRecord employes a builder pattern to easily construct complex datastructures. 
The Builder Pattern simplifies creating complex objects by constructing them step by step. It improves flexibility, readability, and consistency, making it easier to manage and configure objects in a controlled way.

```{literalinclude} scripts/02_medrecord_intro.py
---
language: python
lines: 43
---
```

The MedModels MedRecord object, `record`, now contains three patients. Each patient is identified by an unique index and has specific attributes, such as age, sex, and location. These patients serve as the initial nodes in the graph structure of our MedRecord, and are represented as follows:

```{image} https://raw.githubusercontent.com/limebit/medmodels-static/main/imgs/user_guide/02/02_medrecord_intro_01.png
:class: transparent-image
```
We can now proceed by adding additional data, such as the following medications.
```{literalinclude} scripts/02_medrecord_intro.py
---
language: python
lines: 15-18
---
```
Using the builder pattern to construct the MedRecord allows us to pass as many nodes and edges as needed. If nodes are not added during the initial graph construction, they can easily be added later to an existing MedRecord by calling `.add_nodes()`, where you provide the DataFrame and specify the column containing the node indices.

```{literalinclude} scripts/02_medrecord_intro.py
---
language: python
lines: 46
---
```

This will expand the MedRecord, adding several new nodes to the graph. However, these nodes are not yet connected, so let's establish relationships between them!

```{image} https://raw.githubusercontent.com/limebit/medmodels-static/main/imgs/user_guide/02/02_medrecord_intro_02.png
:class: transparent-image
```
## Adding Edges to a MedRecord

To capture meaningful relationships between nodes, such as linking patients to prescribed medications, we add edges to the MedRecord. These edges must be specified in a relation table, such as the one shown below:

:::{list-table} Patient-Medication Relation
:widths: 15 15 15
:header-rows: 1

*   - Pat_ID
    - Med_ID
    - time
*   - Patient 02
    - Med 01
    - 2020/06/07
*   - Patient 02
    - Med 02
    - 2018/02/02
*   - Patient 03
    - Med 02
    - 2019/03/02
:::

We can add these edges then to our MedRecord Graph:

```{literalinclude} scripts/02_medrecord_intro.py
---
language: python
lines: 54
---
```
This results in an enlarged Graph with more information.

```{image} https://raw.githubusercontent.com/limebit/medmodels-static/main/imgs/user_guide/02/02_medrecord_intro_03b.png
:class: transparent-image
```
## Adding Groups to a MedRecord

For certain analyses, we may want to define specific subcohorts within our MedRecord for easier access. We can do this by defining named groups withing our MedRecored.

```{literalinclude} scripts/02_medrecord_intro.py
---
language: python
lines: 58
---
```
This group will include all the defined nodes, allowing for easier access during complex analyses. Both nodes and edges can be added to a group, with no limitations on group size. Additionally, nodes and edges can belong to multiple groups without restriction.

```{image} https://raw.githubusercontent.com/limebit/medmodels-static/main/imgs/user_guide/02/02_medrecord_intro_04.png
:class: transparent-image
```

## Saving and Loading MedRecords

When build up a MedRecord you might want to save it in order to create a persistent version. This can be done by storing it as a RON (Rusty Object Notation) file. From there, it can also be loaded and a new MedRecord instance can be created from an existing RON file.

```{literalinclude} scripts/02_medrecord_intro.py
---
language: python
lines: 80-81
---
```

## Printing Overview Tables

The MedModels MedRecord class is designed to efficiently handle large datasets while maintaining a standardized data structure that supports complex analysis methods. As a result, the structure within the MedRecord can become intricate and difficult to manage. To address this, MedModels offers tools to help keep track of the graph-based data. One such tool is the `.print_node_overview()` method:

```{literalinclude} scripts/02_medrecord_intro.py
---
language: python
lines: 60
---
```

It will print an overview over all grouped nodes in the MedRecord.

```
------------------------------------------------------
Nodes Group Count Attribute Info                    
------------------------------------------------------
Diagnoses   1     ICD       Values: I.21             
                  ICDv      min: 10                  
                            max: 10                  
                            mean: 10.00              
Medications 2     Name      Values: Insulin, Wararin 
Patients    3     Age       min: 64                  
                            max: 74                  
                            mean: 70.00              
                  Loc       Values: GER, USA         
                  Sex       Values: F, M             
US-Patients 2     Age       min: 72                  
                            max: 74                  
                            mean: 73.00              
                  Loc       Values: USA              
                  Sex       Values: M                
------------------------------------------------------
```
As shown, we have two groups of nodes—Patients and Medications—created when adding the nodes. Additionally, there’s a group called 'US-Patients' that we created. For each group of nodes, we can view their attributes along with a brief statistical summary, such as the minimum, maximum, and mean for numeric variables.

We can do the same to get an overview over edges in our MedRecord by using the `.print_edge_overview()`:

```{literalinclude} scripts/02_medrecord_intro.py
---
language: python
lines: 62
---
```

```
------------------------------------------------------
Nodes Group Count Attribute Info                    
------------------------------------------------------
Diagnoses   1     ICD       Values: I.21             
                  ICDv      min: 10                  
                            max: 10                  
                            mean: 10.00              
Medications 2     Name      Values: Insulin, Wararin 
Patients    3     Age       min: 64                  
                            max: 74                  
                            mean: 70.00              
                  Loc       Values: GER, USA         
                  Sex       Values: F, M             
US-Patients 2     Age       min: 72                  
                            max: 74                  
                            mean: 73.00              
                  Loc       Values: USA              
                  Sex       Values: M                
------------------------------------------------------
```

## Accessing Elements in a MedRecord

Now that we have stored some structured data in our MedRecord, we might want to access certain elements of it. The main way to do this is by either selecting the data with their indicies or via groups that they are in.

```{literalinclude} scripts/02_medrecord_intro.py
---
language: python
lines: 64-78
---
```
## Advanced Querying on a MedRecord

The MedRecord can be queried in very advanced ways in order to finde very specific nodes based on time, relations, neighbors or other. These advanced querying methods are covered in the next session of the user guide.


## Full example Code

The full code examples for this chapter can be foud here: 
```{literalinclude} scripts/02_medrecord_intro.py
---
language: python
---
```