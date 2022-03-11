# Bioinformatics, Data Science Research Project
## Visit My Other Projects!

[Data Science](https://github.com/marjose2/Martinez_Porfolio/tree/main/Data%20Science)

## What This Research Project Was All About
<details><summary>Abstract</summary>

>  Repetitive behaviors, such as grooming, are commonly linked to human neurodevelopmental disorders such as Fragile X syndrome (FSX) (Oakes A. et al., 2016).   Drosophila melanogaster has been previously used in studies to learn about neurodevelopmental disorders. Drosophila CASK (calcium/calmodulin-dependent serine protein kinase) lines display repetitive behavior such as consistently elevated grooming, despite their dramatic decrease in walking (Xingjie R. et al., 2013).  Drosophila CASK presents a loss of function (LOF) of the CASK gene, meaning they do not make the CASK protein, leading to increased grooming phenotype (Xingjie R. et al., 2013). Repetitive behavior, such as excessive grooming, is a crucial subject for researching disorders like FXS. By studying Drosophila CASK lines, neurodevelopmental disorders in humans, such as FXS, can be better understood (Oakes A. et al., 2016). In this study, the creation of knockout stocks, using the Elav>Cas9, Mef2>Cas9, gRNA CASK, and gRNA QUAS, will be used to study the relationship between the CASK gene and the nervous system. These stocks were created by crossing the Cas9/Gal4 lines, Elav>Cas9 and Mef2>Cas9, with either gRNA CASK or gRNA QUAS.  If the CASK gene is working in the nervous system, then knocking out the expression of these genes in all neurons during all phases of development will re-create the phenotype that we see in true null mutants. The Elav knockout stock and the control lines showed no statistical difference in the Grooming Index (GI), the number of grooming bouts, or the mean grooming bout length. The P-values for these parameters are 0.0859, 0.0469, and 0.7399, respectively. These numbers indicate that this experiment's results did not support the hypothesis that the CASK gene is working in the nervous system. In contrast, the Mef2>Cas9 x QUAS cross resulted in a recessive lethal cross. 
</details>
<details><summary> Graphs And Interpretations </summary>
  
<details><summary>1. Ethograms</summary>

<img src="https://github.com/marjose2/Martinez_Porfolio/blob/main/Research%20Project/images/Ethogram.PNG" width="500" />

> This ethogram from 13 individuals flies from the Elav>Cas9 x QUAS line. It should be noted that each row represents an individual fly’s behavior; the black color in the figure represents the time the fly spent grooming, while the grey portions of the figure represent the time the fly did not spend grooming though the 10 minute videos.
  
<img src="https://github.com/marjose2/Martinez_Porfolio/blob/main/Research%20Project/images/ethogram2.PNG" width="500" />

> This ethograms from 13 individual flies from the Elav>Cas9 x CASK line. It should be noted that each row represents an individual fly’s behavior; the black color in the figure represents the time the fly spent grooming, while the grey portions of the figure represent the time the fly did not spend grooming though the 10 minute videos.
 
</details>
  
<details><summary>2. Box Plot Charts Figures A-C</summary>
 <img src="https://github.com/marjose2/Martinez_Porfolio/blob/main/Research%20Project/images/Graph1.PNG" width="500" />

> Figure A shows the Box plots of the values of number of grooming index values of the Elav>Cas9 x CASK and Elav>Cas9 x QUAS represented using Box-and-whisker plots. The blue box shows the 25th-75th percentiles; the red line in the box shows the median. The black lines, extending form the blue box, with the whiskers represent the 9th and the 91th. (p-value=0.0859). This box plot was constructed with 17 Elav>Cas9 x CASK (11 females and 6 males) and 24 Elav>Cas9 x QUAS (12 females and 12 males).

<img src="https://github.com/marjose2/Martinez_Porfolio/blob/main/Research%20Project/images/Graph2.PNG" width="500" />

> Figure B shows Box plots of the values of the number of grooming bouts done by Elav>Cas9 x CASK and Elav>Cas9 x QUAS represented using Box-and-whisker plots. The blue box shows the 25th-75th percentiles; in the red line in the box shows the median. The black lines, extending form the blue box, with the whiskers represents the 9th and the 91th. (p-value 0.0469). This box plot was constructed with 17 Elav>Cas9 x CASK (11 females and 6 males) and 24 Elav>Cas9 x QUAS (12 females and 12 males).

<img src="https://github.com/marjose2/Martinez_Porfolio/blob/main/Research%20Project/images/Graph3.PNG" width="500" />

> Figure C shows Box plots of the values of the mean bout length done by Elav>Cas9  x CASK and Elav>Cas9 x QUAS represented using Box-and-whisker plots. The blue box shows the 25th-75th percentiles; in the red line in the box shows the median. The black lines, extending form the blue box, with the whiskers represents the 9th and the 91th. (p-value=0.7399). This box plot was constructed with 17 Elav>Cas9 x CASK (11 females and 6 males) and 24 Elav>Cas9 x QUAS (12 females and 12 males).
</details>
</details>

</details>
<details><summary>How The Data Was Generated Using Code</summary>
The data, of much each fly groomed in a 10 minute video, was extracted from using V-Code and Custom In Perl Scrips. V-code is an open-source video-annotation software that allows one to markdown and score when a fly groomed itself. When scoring the videos, each fly had a corresponding key, and whenever a fly started grooming, the key corresponding to that fly was pressed. The same key was pressed when the fly stopped grooming; this marked the beginnings and ends of a grooming bout, thus allowing for quantification of the amount of time spent grooming by each fly. 
Once the videos were scored using V-code they were turn into text files so the data, of much each fly groomed in a 10 minute video, could be extrated using custom, in-house Perl scripts. The custom scripts in Perl to parse the VCode output and quantify behavioral metrics for each fly, such as total grooming time, a number of grooming bouts, grooming index (GI, the percentage of time spent grooming during a given interval), and comparable indices for standing, walking and falling. For a detailed analysis of grooming, additional parameters were the type and frequency of grooming-site transitions within grooming bouts, and time spent grooming specific body parts. The Perl scripts produce tab-delimited output files that were imported to Microsoft Excel (Microsoft Corporation, Redmond, WA, USA) and MATLAB (MathWorks®, Natick, MA, USA) for further analysis.

</details>

</details>
<details><summary>Full Reasearch Paper Link</summary>
https://github.com/marjose2/Martinez_Porfolio/blob/main/Research%20Project/Bioinformatics%2C%20Data%20Science%20Research%20paper.docx
</details>
 
 
## Research At A Glance
### Assertion of motor-behavior abnormalities in Drosophila CASK mosaics

<img src="https://github.com/marjose2/Martinez_Porfolio/blob/main/Research%20Project/images/Poster.PNG" width="1500" />

## Contact Me:

+ Profesional Email: joseignacio1225@hotmail.com
+ Linkedin: https://www.linkedin.com/in/jose-martinez-10303b1aa/



