# %%
import datetime
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import gemmi
import numpy as np
import requests
from absl import logging

from alphafold3 import structure
from alphafold3.constants import mmcif_names
from alphafold3.data import msa_config, parsers, structure_stores
from alphafold3.data import templates as templates_lib
from alphafold3.structure import mmcif


def _parse_hit_metadata(
  structure_store: structure_stores.StructureStore,
  pdb_id: str,
  auth_chain_id: str,
) -> tuple[Any, str | None, Sequence[int] | None, Mapping[str, str] | None]:
  """Parse hit metadata by parsing mmCIF from structure store."""
  try:
    cif = mmcif.from_string(structure_store.get_mmcif_str(pdb_id))
  except structure_stores.NotFoundError:
    logging.warning(
      "Failed to get mmCIF for %s (author chain %s).", pdb_id, auth_chain_id
    )
    return None, None, None, None
  release_date = mmcif.get_release_date(cif)

  try:
    struc = structure.from_parsed_mmcif(
      cif,
      model_id=structure.ModelID.ALL,
      include_water=True,
      include_other=True,
      include_bonds=False,
    )
  except ValueError:
    struc = structure.from_parsed_mmcif(
      cif,
      model_id=structure.ModelID.FIRST,
      include_water=True,
      include_other=True,
      include_bonds=False,
    )

  sequence = struc.polymer_author_chain_single_letter_sequence(
    include_missing_residues=True,
    protein=True,
    dna=True,
    rna=True,
    other=True,
  )[auth_chain_id]

  unresolved_res_ids = struc.filter(
    chain_auth_asym_id=auth_chain_id
  ).unresolved_residues.id.tolist()

  mapping = struc.polymer_auth_asym_id_to_label_asym_id()

  return (release_date, sequence, unresolved_res_ids, mapping)


def _polymer_auth_asym_id_to_label_asym_id_with_gemmi(
  model: gemmi.Model,
) -> Mapping[str, str]:
  """
  Docstring for polymer_auth_asym_id_to_label_asym_id_with_gemmi

  :param model: Description
  :type model: gemmi.Model
  :return: Description
  :rtype: dict[str, str]
  """
  auth_asym_id_to_label_asym_id = {}
  for chain in model:
    polymer = chain.get_polymer()
    if polymer is None:
      continue
    auth_asym_id = chain.name
    label_asym_id = polymer.subchain_id()

    if auth_asym_id in auth_asym_id_to_label_asym_id:
      raise ValueError(
        f'Author chain ID "{auth_asym_id}" does not have a unique mapping '
        f'to internal chain ID "{label_asym_id}", it is already mapped to '
        f'"{auth_asym_id_to_label_asym_id[auth_asym_id]}".'
      )
    auth_asym_id_to_label_asym_id[auth_asym_id] = label_asym_id
  return auth_asym_id_to_label_asym_id


def _parse_hit_metadata_with_gemmi(
  structure_store: structure_stores.StructureStore,
  pdb_id: str,
  auth_chain_id: str,
) -> tuple[Any, str | None, Sequence[int] | None, Mapping[str, str] | None]:
  """Parse hit metadata by parsing mmCIF from structure store."""
  try:
    doc = gemmi.cif.read_string(structure_store.get_mmcif_str(pdb_id))
  except structure_stores.NotFoundError:
    logging.warning(
      "Failed to get mmCIF for %s (author chain %s).", pdb_id, auth_chain_id
    )
    return None, None, None, None
  # get release date from the 1st "_pdbx_audit_revision_history.revision_date"
  block = doc.sole_block()
  model = gemmi.read_structure_string(structure_store.get_mmcif_str(pdb_id))[0]
  auth_asym_id_to_label_asym_id = _polymer_auth_asym_id_to_label_asym_id_with_gemmi(
    model
  )
  release_date = block.find_values("_pdbx_audit_revision_history.revision_date")[0]
  # get sequence
  entity_polys = block.find(
    "_entity_poly.", ["type", "pdbx_seq_one_letter_code_can", "pdbx_strand_id"]
  )
  for entity_poly in entity_polys:
    if entity_poly["type"] != "'polypeptide(L)'":
      raise ValueError(f"Unexpected polymer type: {entity_poly['type']}")
    if auth_chain_id in entity_poly["pdbx_strand_id"].split(","):
      sequence = (
        entity_poly["pdbx_seq_one_letter_code_can"]
        .replace("'", "")
        .replace(";", "")
        .replace("\n", "")
      )
  # missing residues
  print(f"sequence length: {len(sequence)}")
  label_asym_residues = model[auth_chain_id].get_polymer()
  all_res_ids = np.arange(1, len(sequence) + 1)
  resolved_res_ids = np.array([res.label_seq for res in label_asym_residues])
  unresolved_res_ids = (
    np.isin(all_res_ids, resolved_res_ids, invert=True).nonzero()[0] + 1
  ).tolist()
  return (release_date, sequence, unresolved_res_ids, auth_asym_id_to_label_asym_id)


def download_ciffile(pdb_id: str, directory: str | Path) -> None:
  """Download mmCIF file from RCSB PDB."""
  url = f"https://files.rcsb.org/download/{pdb_id}.cif"
  response = requests.get(url)
  if response.status_code == 200:
    with open(Path(directory, f"{pdb_id}.cif"), "wb") as f:
      f.write(response.content)
    logging.info("Downloaded mmCIF file for %s.", pdb_id)
  else:
    logging.error(
      "Failed to download mmCIF file for %s. Status code: %d",
      pdb_id,
      response.status_code,
    )


# %%
directory = "/Users/YoshitakaM/Desktop/mmcif_files"
cif_dir = structure_stores.StructureStore(directory)
# pdb_id = "1A0I"
pdb_id = "2GKT"
auth_chain_id = "I"
release_date1, sequence1, unresolved_res_ids1, mapping1 = _parse_hit_metadata(
  cif_dir, pdb_id, auth_chain_id
)
release_date2, sequence2, unresolved_res_ids2, mapping2 = (
  _parse_hit_metadata_with_gemmi(cif_dir, pdb_id, auth_chain_id)
)
print(f"Checking {pdb_id}...")
print(f"Release date: {release_date1} vs {release_date2}")
print(f"Sequence: {sequence1} vs {sequence2}")
print(f"Unresolved residues: {unresolved_res_ids1} vs {unresolved_res_ids2}")
assert release_date1 == release_date2, f"Release date mismatch for {pdb_id}"
assert sequence1 == sequence2, f"Sequence mismatch for {pdb_id}"
assert unresolved_res_ids1 == unresolved_res_ids2, (
  f"Unresolved residues mismatch for {pdb_id}"
)


# %%
def check_results_diff():
  """Check that both implementations give the same results."""
  directory = "/Users/YoshitakaM/Desktop/mmcif_files"
  cif_dir = structure_stores.StructureStore(directory)
  pdb_ids = [
    "3RUR",
    "9HAQ",
    "2Z6F",
    "5BUT",
    "9K2J",
    "4HHB",
    "1GFL",
    "2KIC",
    "1A0I",
  ]
  auth_chain_id = "A"
  for pdb_id in pdb_ids:
    print(f"Processing {pdb_id} with auth_chain_id {auth_chain_id}...")
    if not Path(directory, f"{pdb_id}.cif").exists():
      logging.info(
        "Skipping %s as mmCIF file is not found. Download from RCSB PDB.", pdb_id
      )
      download_ciffile(pdb_id, directory)
    (
      release_date1,
      sequence1,
      unresolved_res_ids1,
      mapping1,
    ) = _parse_hit_metadata(cif_dir, pdb_id, auth_chain_id)
    (
      release_date2,
      sequence2,
      unresolved_res_ids2,
      mapping2,
    ) = _parse_hit_metadata_with_gemmi(cif_dir, pdb_id, auth_chain_id)
    print(f"Checking {pdb_id}...")
    print(f"Release date: {release_date1} vs {release_date2}")
    print(f"Sequence: {sequence1} vs {sequence2}")
    print(f"Unresolved residues: {unresolved_res_ids1} vs {unresolved_res_ids2}")
    print(f"Mapping: {mapping1} vs {mapping2}")
    assert release_date1 == release_date2, f"Release date mismatch for {pdb_id}"
    assert sequence1 == sequence2, f"Sequence mismatch for {pdb_id}"
    assert unresolved_res_ids1 == unresolved_res_ids2, (
      f"Unresolved residues mismatch for {pdb_id}"
    )
    assert mapping1 == mapping2, f"Mapping mismatch for {pdb_id}"
  print("All tests passed!")


check_results_diff()

# %%
a3mfile = "/Users/YoshitakaM/Desktop/orf5.a3m"
outstofile = "/Users/YoshitakaM/Desktop/orf5.sto"
mmcif_dir = "/Users/YoshitakaM/Desktop/mmcif_files"
database_path = "/Users/YoshitakaM/Desktop/pdb_seqres.txt"

filter_config = msa_config.TemplateFilterConfig(
  max_subsequence_ratio=0.95,
  min_align_ratio=0.1,
  min_hit_length=10,
  deduplicate_sequences=True,
  max_hits=4,
  max_template_date=datetime.date(2099, 6, 14),
)
# query sequence is the first sequence in the a3m file
with open(a3mfile, "r") as f:
  query_sequence = parsers.parse_fasta(f.read())[0][0]

msa_a3m: str = Path(a3mfile).read_text()

hmmsearch_config: msa_config.HmmsearchConfig = msa_config.HmmsearchConfig(
  hmmsearch_binary_path=shutil.which("hmmsearch"),
  hmmbuild_binary_path=shutil.which("hmmbuild"),
  e_value=100,
  inc_e=100,
  dom_e=100,
  incdom_e=100,
  alphabet="amino",
  filter_f1=0.1,
  filter_f2=0.1,
  filter_f3=0.1,
  filter_max=False,
)
# %%

protein_templates = templates_lib.Templates.from_seq_and_a3m(
  query_sequence=query_sequence,
  msa_a3m=msa_a3m,
  max_template_date=datetime.date(2099, 6, 14),
  database_path=database_path,
  hmmsearch_config=hmmsearch_config,
  max_a3m_query_sequences=None,
  chain_poly_type=mmcif_names.PROTEIN_CHAIN,
  structure_store=structure_stores.StructureStore(mmcif_dir),
  filter_config=filter_config,
)

# %%
for hit in protein_templates.hits:
  print(f"Hit: {hit.full_name}")
  print(f" Release date: {hit.release_date}")
  print(f" Length ratio: {hit.length_ratio:.3f}")
  print(f" Align ratio: {hit.align_ratio:.3f}")
  print(f" Is valid: {hit.is_valid}")
  print(f" HMMsearch sequence: {hit.hmmsearch_sequence}")
  print(f" Structure sequence: {hit.structure_sequence}")
  print(f" Output template sequence: {hit.output_templates_sequence}")
  print(f" Query to hit mapping: {hit.query_to_hit_mapping}")

# %%
