# Jupyter notebook

If you want to use a Jupyter notebook for your report, you need to make sure to:

- Name your notebook `report.ipynb`.
- Submit the notebook with **all cells executed** (outputs should be saved in the `.ipynb` file).
- Export the notebook as HTML or PDF (**including outputs**) and include it in the submitted ZIP file:
  - If you are using Jupyter's browser interface: `File > Download as > HTML`.
  - If you are using Google Colab, print the notebook as PDF: `File > Print > Save as PDF`.
  - Alternatively, you can use [`nbconvert`](https://nbconvert.readthedocs.io/):  
    `jupyter nbconvert --to html report.ipynb`
