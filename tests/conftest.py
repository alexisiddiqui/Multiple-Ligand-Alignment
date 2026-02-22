from jaxtyping import install_import_hook

# All functions in ligand_neff will be checked at test time
install_import_hook("ligand_neff", "beartype.beartype")
