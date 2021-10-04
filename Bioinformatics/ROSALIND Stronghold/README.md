
# Problems

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
