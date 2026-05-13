# Moral Machine Data

Downloaded from the OSF Moral Machine Dataset project:

https://osf.io/3hvt2/

The current raw download includes:

- `raw/moral_machine_data/`: the main Moral Machine response archives and small country/culture files.
- `raw/effect_sizes/`: paper effect-size `.rdata` files.
- `raw/external_data/`: small external country/culture support files.

The large OSF external file `csv_pus.zip` is skipped by default because it is not needed for the initial Moral Machine response prediction setup and adds about 2.5 GB. To include it, rerun:

```bash
DOWNLOAD_EXTERNAL_LARGE=1 moral_machine/scripts/download_moral_machine.sh
```

Local SHA-256 checksums and file sizes are in `metadata/`.
