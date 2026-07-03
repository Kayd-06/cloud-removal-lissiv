"""Export co-registered Sentinel-1 SAR + Sentinel-2 optical + cloud mask over a
LISS-IV footprint, straight from Google Earth Engine (free).

This removes most of the manual data-wrangling: GEE reprojects/clips everything to a
common grid and hands you analysis-ready GeoTIFFs. You then feed the S2 'clear'
composite to `src.preprocess.tiling` and the S1 scene as the SAR prior.

Why S2 as the 'clear' target?
  LISS-IV cloud-free scenes are scarce and slow to order. Sentinel-2 gives you an
  abundant, free clear-optical target with matching Green/Red/NIR bands, so you can
  build the whole training set NOW and fine-tune on LISS-IV later. Classic
  cross-sensor transfer -- and itself a novelty point.

--------------------------------------------------------------------------------
SETUP (one time, free):
  1. Register for Earth Engine: https://earthengine.google.com -> "Get Started",
     sign in with Google, create/choose a Cloud project (free, non-commercial use).
  2. Note your project id (e.g. "ee-yourname") -> pass as --project.

RUN INSIDE A KAGGLE NOTEBOOK (recommended, no local Python needed):
  cell 1:
      !pip install -q earthengine-api geemap
  cell 2:
      import ee; ee.Authenticate()      # prints a link -> approve -> paste token
  cell 3 (small area, downloads straight into Kaggle -- easiest):
      !python scripts/gee_export.py --project ee-yourname \
          --bbox 91.72 26.10 91.80 26.18 \   # ~0.08 deg test box near Guwahati
          --start 2023-11-01 --end 2024-02-28 \
          --to_local data/raw

For LARGER areas, drop --to_local; it batch-exports to your Google Drive instead
(no ~32MB download cap), then you copy those GeoTIFFs into Kaggle.

Produces three GeoTIFFs:
  s2_clear_*.tif  (G,R,NIR reflectance)   -> tiling.py --liss
  s1_sar_*.tif    (VV,VH)                 -> tiling.py --sar
  s2_cloudmask_*.tif                       -> cloud_mask.transfer_s2_mask
--------------------------------------------------------------------------------
"""
from __future__ import annotations
import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, help="Google Cloud / EE project id")
    ap.add_argument("--bbox", nargs=4, type=float, required=True,
                    metavar=("minLon", "minLat", "maxLon", "maxLat"))
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--out_folder", default="lissiv_ee", help="Drive folder name (drive mode)")
    ap.add_argument("--scale", type=int, default=10, help="export resolution (m); S2=10")
    ap.add_argument("--max_cloud", type=int, default=20, help="max scene cloud %% for clear target")
    ap.add_argument("--to_local", default=None,
                    help="if set, download GeoTIFFs straight to this local dir (Kaggle-friendly, "
                         "no Google Drive). Best for SMALL areas (<~0.1 deg) due to a ~32MB "
                         "download cap. Requires geemap.")
    args = ap.parse_args()

    try:
        import ee
    except ImportError:
        raise SystemExit("pip install earthengine-api geemap, then authenticate (see header)")

    # In a Kaggle/Colab notebook, run `ee.Authenticate()` once (paste the token) before this.
    ee.Initialize(project=args.project)
    region = ee.Geometry.Rectangle(args.bbox)

    # ---------------- Sentinel-2 cloud-free composite (the 'clear' target) ----------------
    def mask_s2_scl(img):
        scl = img.select("SCL")
        # keep vegetation/soil/water/etc; drop cloud (8,9,10), shadow (3), cirrus (10)
        clear = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
        return img.updateMask(clear)

    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(region)
          .filterDate(args.start, args.end)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", args.max_cloud))
          .map(mask_s2_scl))

    # LISS-IV band analogues: B3=Green, B4=Red, B8=NIR. Scale reflectance to [0,1].
    s2_clear = (s2.median()
                .select(["B3", "B4", "B8"], ["green", "red", "nir"])
                .divide(10000).clamp(0, 1).clip(region))

    # ---------------- Sentinel-2 cloud mask (probabilistic, for real cloudy scenes) --------
    # s2cloudless collection gives per-pixel cloud probability
    s2prob = (ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
              .filterBounds(region).filterDate(args.start, args.end))
    cloud_mask = (s2prob.mean().gt(40).rename("cloud").unmask(0).clip(region))

    # ---------------- Sentinel-1 SAR (cloud-penetrating prior) -----------------------------
    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
          .filterBounds(region)
          .filterDate(args.start, args.end)
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))
    s1_img = s1.select(["VV", "VH"]).median().clip(region)

    tag = f"{args.bbox[0]:.2f}_{args.bbox[1]:.2f}"
    tasks = [
        ("s2_clear_" + tag, s2_clear.toFloat()),
        ("s2_cloudmask_" + tag, cloud_mask.toFloat()),
        ("s1_sar_" + tag, s1_img.toFloat()),
    ]

    if args.to_local:
        # ---------- direct download to the local filesystem (Kaggle/Colab) ----------
        import os
        try:
            import geemap
        except ImportError:
            raise SystemExit("pip install geemap  (needed for --to_local)")
        os.makedirs(args.to_local, exist_ok=True)
        for name, img in tasks:
            out = os.path.join(args.to_local, name + ".tif")
            print(f"[gee] downloading {name} -> {out} ...")
            geemap.ee_export_image(img, filename=out, scale=args.scale,
                                   region=region, file_per_band=False)
        print(f"\n[done] GeoTIFFs saved to {args.to_local}/  "
              f"-> feed s2_clear_*.tif to tiling.py --liss and s1_sar_*.tif to --sar")
    else:
        # ---------- batch export to Google Drive (for larger areas) ----------
        for name, img in tasks:
            task = ee.batch.Export.image.toDrive(
                image=img, description=name, folder=args.out_folder,
                fileNamePrefix=name, region=region, scale=args.scale, maxPixels=1e10,
                fileFormat="GeoTIFF",
            )
            task.start()
            print(f"[gee] started export: {name}  (scale={args.scale}m) -> Drive/{args.out_folder}")
        print("\nTrack progress at https://code.earthengine.google.com/tasks "
              "or `earthengine task list`. Files land in your Google Drive when done.")


if __name__ == "__main__":
    main()
