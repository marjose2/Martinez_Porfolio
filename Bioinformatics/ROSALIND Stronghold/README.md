
# Problems
Most of these problems were solved using Biophython.

## 1.Counting Nucleotides
## 2.Trancribing DNA into RNA
Code
```
string = "GATGGAACTTGACTACGTAAATT"
print(string.replace("T", "U"))
```
## 3.Complementing Starnd of DNA 
code
```
import Bio
from Bio.Seq import Seq
seq = Seq("AAAACCCGGT")
print(seq.reverse_complement())
```
## 4.Rabbits and Recurrence Relations
code
```
def Fibonacci_Loop_Pythonic(months, offsprings):
    parrent, child = 1, 1
    for itr in range(months - 1):
        child, parrent = parrent, parrent + (child * offsprings)
    return child
print(Fibonacci_Loop_Pythonic(5, 3))
```
## 5. Computing GC Content
Code
```
#import libaries from Biopython, this makes the code more consise and easy to follow
import Bio
from Bio import SeqIO
#download the data from the folder (the folder has to be located in the directory where your working)
for seq_record in SeqIO.parse("Is_orchid.fasta", "fasta"):
#print the basic outline of the data to see whats going on
    print(seq_record.id)
    print(repr(seq_record.seq))
    print(len(seq_record)) 
#Imports a way to counting the GC content using one line of code
from Bio.SeqUtils import GC
my_seq = seq_record.seq
GC(my_seq)
#print the results
print(GC(my_seq))
print(seq_record.id)
```

## 6.Counting Point Mutations
Code
```
def hammingDist(a,b):
    hamD = 0
    for i in range(0,len(s)):
        if s[i] == t[i]:
            continue
        else:
            hamD += 1
    return hamD
  
a = 'GAGCCTACTAACGGGAT'
b = 'CATCGTAATGACGGCCT'
hammingDist(a,b)
```
