import pandas as pd
from tqdm import tqdm
from .tng_api import get_json

BASE_URL = "https://www.tng-project.org/api/TNG100-1"
SNAPSHOT = 99


def fetch_subhalo_page(limit=100, offset=0, order_by="-mass_stars"):
    url = f"{BASE_URL}/snapshots/{SNAPSHOT}/subhalos/"
    params = {
        "limit": limit,
        "offset": offset,
        "order_by": order_by
    }
    return get_json(url, params=params)


def fetch_subhalo_detail(subhalo_url):
    return get_json(subhalo_url)


def fetch_parent_halo_info(parent_halo_url):
    return get_json(parent_halo_url + "info.json")


def build_initial_catalog(max_subhalos=5000, page_size=100):
    rows = []
    offset = 0

    while len(rows) < max_subhalos:
        page = fetch_subhalo_page(
            limit=page_size,
            offset=offset,
            order_by="-mass_stars"
        )

        results = page["results"]

        if len(results) == 0:
            break

        for item in tqdm(results, desc=f"offset={offset}"):
            if len(rows) >= max_subhalos:
                break

            sub = fetch_subhalo_detail(item["url"])

            parent_halo_url = sub["related"]["parent_halo"]
            parent = fetch_parent_halo_info(parent_halo_url)

            group = parent["Group"]

            row = {
                "simulation": "TNG100-1",
                "snapshot": SNAPSHOT,
                "redshift": 0.0,

                "subhalo_id": sub["id"],
                "halo_id": sub["grnr"],
                "is_central": int(sub.get("primary_flag", 0)),

                "subhalo_url": sub["meta"]["url"],
                "parent_halo_url": parent_halo_url,

                # Posición del subhalo
                "pos_x": sub.get("pos_x"),
                "pos_y": sub.get("pos_y"),
                "pos_z": sub.get("pos_z"),

                # Masas disponibles desde el subhalo
                "mass_log_msun": sub.get("mass_log_msun"),
                "stellar_mass_api": sub.get("mass_stars"),
                "gas_mass_api": sub.get("mass_gas"),
                "sfr": sub.get("sfr"),

                # Propiedades del halo padre.
                # El nombre exacto puede variar según endpoint/catálogo,
                # por eso guardamos el diccionario base también si hace falta.
                "group_m_crit200": group.get("Group_M_Crit200"),
                "group_m_mean200": group.get("Group_M_Mean200"),
                "group_first_sub": group.get("GroupFirstSub"),
                "group_nsubs": group.get("GroupNsubs"),

                "quality_flag": 1
            }

            rows.append(row)

        offset += page_size

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    df = build_initial_catalog(max_subhalos=5000, page_size=100)
    df.to_csv("data/raw/tng/metadata_raw/tng100_snapshot99_subhalos.csv", index=False)
    print(df.head())
    print(f"Total: {len(df)}")