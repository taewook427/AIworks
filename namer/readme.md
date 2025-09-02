# ACH-namer
Archiving Content Helper namer version

### Series Rename Mode
option tgt rename real i

Rename file that contains certain string.

Program attatches serial number at front of file name.

Each identifier should independent.

### Auto Grouping Mode
option tgt real idf sf cos th

Program automatically groups file names by simillarity.

Using extended TF-IDF and Hierarchical Clustering.

Set threshold to cut merging, include tree.txt file.

### Options
Default values are `read-only`, `clustering`, `idf 0.01`, `sf 1.0`, `cos 2.5`, `th 0.2`.

`-tgt D` : designate folder to work

`-real` : change files. without this option, program only shows what will happen

`-rename` : set rename mode

`-i S` : add rename mode detecter string

`-idf f` : set Inverse Document Frequency weight. bigger it gets, program thinks scarce terms are important

`-sf f` : set Series Frequency weight. bigger it gets, program thinks frequent terms are series name, and important

`-cos f` : set cosine similarity amplifier. bigger it gets, cos distance deviations are increased

`-th f` : set merge cut threshold. bigger it gets, more groups are merged into 1 group

### ext TF-IDF
Program uses different weight algorithm from TF-IDF.

Most file names have common "series name" part and unique "own name" part.

Default TF-IDF can not detect "series name" and set too much weight on "own name".

$$
\text{ext TF-IDF} = tf(D, t) \times \log\left( \frac{A \cdot (1 + n)}{1 + df(t)} + B \cdot (1 + df(t)) \right)
$$

By using this ext TF-IDF, program can set weight both "series name" and "uncommon terms".
