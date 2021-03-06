# Approximate Byte-based n-gram Analysis

Can be used to compare a bunch of files.  
Makes use of mmap-ing, sparse vectors and all your CPUs to be decently fast.


## Examples

### Simple / stdout

Comparing some files:

```
$ ls
manhattan.jpg  shuffleparrot.gif  tokio.png  ultrafastparrot.gif

$ ngrm -n 8 -o /dev/null --stdout *
n = 8
m = 100000
Time (generate_basis): 39 ms
Time (build_file_vectors): 120 ms
Time (generate_similarity_matrix): 4 ms

       manhat shuffl tokio. ultraf 
manhat 1.0000,0.0000,0.0011,0.0000
shuffl 0.0000,1.0000,0.0000,0.9695
tokio. 0.0011,0.0000,1.0000,0.0000
ultraf 0.0000,0.9695,0.0000,1.0000

Wrote output to file: /dev/null
Time (total): 165 ms
```

As can be seen from the output similarity matrix, `ultrafastparrot.gif` and `shuffleparrot.gif` must be similar files.

### Using clustering algorithm

The following command runs the analysis on all files in ~/data/,
which have a combined size of 3.8 GB. A similarity matrix
is generated and written to matrix.csv.
```
$ ngrm -n 8 ~/data/*.* -o matrix.csv
```
This takes about 12.3 seconds to run on my machine (NVMe SSD, Ryzen 9 3900X with 24 threads @ 3.8 GHz).
The similarity matrix can then (for example) be used together with a clustering
algorithm like the one included in this repo:
```
$ python ward_cluster.py matrix.csv
```
...which produces this output:

![Output](figures/clusters.png)

In this plot, each point (x, y) has a brightness value
corresponding to how similar the files x and y are.
E.g. all points on the diagonal (x,x) should have a value of 1 and
are bright yellow, as each file is completely similar to itself.
Each cluster corresponds to a set of files that are similar or identical.

## How It Works

This approximates an [n-gram analysis](https://en.wikipedia.org/wiki/N-gram) by first selecting *M* (`--vector-length`) random
byte sequences of length *N* (`-n`) from all the supplied files. These are then used
to map all files into a vector space with dimension *M*, where their similarity is computed
using the [cosine similarity metric](https://en.wikipedia.org/wiki/Cosine_similarity).

## Usage

```
ngrm [FLAGS] [OPTIONS] -n <n> --output <output> [FILES]...

FLAGS:
    -h, --help       Prints help information
    -s, --silent     Silence output
        --stats      Print top 25 most common sequences
        --stdin      Read input file paths from stdin instead of the arguments
    -V, --version    Prints version information

OPTIONS:
    -m, --vector-length <m>    The desired vector size. Higher values give more precise results. The required working
                               memory will be O(n * m). This value should be chosen as high as your RAM permits (default
                               is 100,000)
    -M, --mapping <mapping>    Optional output file with a mapping of (file |-> index in matrix)
    -n <n>                     The n in n-gram (Byte sequence length)
    -o, --output <output>      Output file for similarity matrix

ARGS:
    <FILES>...    Input files
```
