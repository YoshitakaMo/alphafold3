# %%
from alphafold3.data.parsers import convert_stockholm_to_a3m

stofile = "/Users/YoshitakaM/Desktop/uniprot_raf.sto"

with open(stofile, "r") as f:
  stockholm_str = convert_stockholm_to_a3m(f, 10000)
print(stockholm_str)
# %%
